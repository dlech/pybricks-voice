"""
The app on Windows: core.py's Core in a WinUI 3 window, with Whisper on the
Snapdragon's NPU.

asyncio runs as a guest of WinUI's DispatcherQueue (see loop.py), and the 3D
view presents into a SwapChainPanel (see scene_view.py).
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import logging
import pathlib
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, override

assert not TYPE_CHECKING or sys.platform == "win32"

import numpy as np
from winrt.microsoft.ui.interop import get_window_from_window_id
from winrt.microsoft.ui.windowing import OverlappedPresenter
from winrt.microsoft.ui.xaml import (
    Application,
    DispatcherShutdownMode,
    FocusState,
    LaunchActivatedEventArgs,
    RoutedEventArgs,
    SizeChangedEventArgs,
    Style,
    Visibility,
    WindowEventArgs,
)
from winrt.microsoft.ui.xaml.controls import (
    ScrollViewerViewChangedEventArgs,
    XamlControlsResources,
)
from winrt.microsoft.ui.xaml.input import KeyRoutedEventArgs, PointerRoutedEventArgs
from winrt.microsoft.ui.xaml.markup import IXamlMetadataProvider, IXamlType, XmlnsDefinition
from winrt.microsoft.ui.xaml.xamltypeinfo import XamlControlsXamlMetaDataProvider
from winrt.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import InitializeOptions, initialize
from winrt.runtime import ApartmentType, init_apartment
from winrt.system import Array, Object, box_string
from winrt.windows.foundation import IStringable
from winrt.windows.system import VirtualKey
from winrt.windows.ui.xaml.interop import TypeKind, TypeName

from ..core import Core, parse_args
from ..speech import Microphone
from ..whisper import Transcriber
from . import placement
from .loop import DispatcherQueueEventLoop
from .main_window import MainWindow
from .microphone import AudioGraphMicrophone
from .observable_list import ObservableList
from .scene_view import SceneView, use_d3d12
from .screenshot import save_window
from .whisper_qnn import Whisper

log = logging.getLogger(__name__)


@dataclasses.dataclass
class ChatMessage(IStringable):
    """
    A message in the chat: main_window.xaml's ChatItem. IStringable makes it a
    WinRT object, which an ObservableList's items must be.
    """

    text: str
    detail: str
    bubble_style: Style
    text_style: Style

    @property
    def detail_visibility(self) -> Visibility:
        return Visibility.VISIBLE if self.detail else Visibility.COLLAPSED

    @override
    def to_string(self) -> str:
        return self.text


class WindowsApp(Core):
    """main_window.xaml's code-behind: its handlers, and the values it binds to."""

    WHISPER_ON = "on the NPU"

    def __init__(self, application: Application, options: argparse.Namespace) -> None:
        super().__init__(options)
        self.application = application
        self.following_chat = False  # scrolling to the end: on_chat_view_changed
        self.chat_at_end = True  # where the chat was when it last stopped scrolling
        self.messages = ObservableList[ChatMessage]()  # the chat, main_window.xaml's py:ItemsSource

        # main_window.xaml, with its events wired to the on_* handlers below.
        self.ui = MainWindow(self)
        self.window = self.ui.window
        self.window.title = "Pybricks voice driving · Whisper on the NPU"
        self.window.app_window.resize((1600, 1000))
        # Where it was last time, but not when taking a screenshot, which
        # should be the same size every run.
        self.hwnd = get_window_from_window_id(self.window.app_window.id)
        self.remember_placement = self.options.screenshot is None
        maximize = self.remember_placement and placement.restore(self.hwnd)
        self.view = SceneView(self.ui.host, self.ui.panel, self.step)

        # Closing the window starts shutdown(), and the app exits once main() has
        # unwound, which takes the loop a few more iterations.
        application.dispatcher_shutdown_mode = DispatcherShutdownMode.ON_EXPLICIT_SHUTDOWN
        loop = DispatcherQueueEventLoop(self.window.dispatcher_queue)
        loop.start()
        self.view.on_first_frame = lambda: loop.call_soon_threadsafe(self.scene_ready.set)
        self.view.start()
        self.start(loop)
        self.window.activate()
        if maximize:
            self.window.app_window.presenter.as_(OverlappedPresenter).maximize()

    # ----- Core's -------------------------------------------------------------

    @override
    def make_microphone(self, on_samples: Callable[[np.ndarray], None]) -> Microphone:
        return AudioGraphMicrophone(on_samples)

    @override
    def load_whisper(self) -> Transcriber:
        return Whisper()

    @override
    def update_view(self) -> None:
        self.ui.update()

    @override
    def add_bubble(self, text: str, mine: bool, detail: str | None = None) -> None:
        look = "Mine" if mine else "Robot"
        styles = self.ui.content.resources
        self.messages.append(
            ChatMessage(
                text,
                detail or "",
                styles[box_string(f"{look}Bubble")].as_(Style),
                styles[box_string(f"{look}Text")].as_(Style),
            )
        )
        while len(self.messages) > 100:
            self.messages.pop(0)
        # Newest message at the bottom, as in a messaging app: follow the
        # conversation, once the new bubble is laid out to know its height.
        if self.ui.log.items_source is not None:  # set by the first update()
            self.ui.log.update_layout()
            self.scroll_chat_to_end()

    @override
    async def save_screenshot(self, path: pathlib.Path) -> None:
        await save_window(self.window, path)

    @override
    def exit(self) -> None:
        self.loop.stop()
        self.application.exit()

    # ----- the chat -----------------------------------------------------------

    def scroll_chat_to_end(self) -> None:
        self.following_chat = True
        self.ui.scroller.change_view(None, self.ui.scroller.scrollable_height, None)

    def on_chat_view_changed(self, sender: Object, args: ScrollViewerViewChangedEventArgs) -> None:
        if args.is_intermediate:
            return
        scroller = self.ui.scroller
        self.chat_at_end = scroller.vertical_offset >= scroller.scrollable_height - 0.5
        # The repeater makes only the bubbles in view, guessing the others'
        # heights, so the end moves as scrolling makes them: scroll on until
        # it's reached. Only while following, or the guesses changing as the
        # view moves would make it shake.
        if self.following_chat:
            if self.chat_at_end:
                self.following_chat = False
            else:
                scroller.change_view(None, scroller.scrollable_height, None)

    def on_chat_size_changed(self, sender: Object, args: SizeChangedEventArgs) -> None:
        # The status and queue lines take room from the chat as they grow;
        # keep the last message in view if it was.
        if self.chat_at_end:
            self.scroll_chat_to_end()

    # ----- main_window.xaml's handlers ------------------------------------------

    def submit_typed(self) -> None:
        if self.submit(self.ui.typed.text):
            self.ui.typed.text = ""

    def on_go(self, sender: Object, args: RoutedEventArgs) -> None:
        self.submit_typed()

    def on_typed_key(self, sender: Object, args: KeyRoutedEventArgs) -> None:
        if args.key == VirtualKey.ENTER:
            self.submit_typed()

    def on_key(self, sender: Object, args: KeyRoutedEventArgs) -> None:
        if args.key == VirtualKey.C and self.ui.typed.focus_state == FocusState.UNFOCUSED:
            self.toggle_camera()

    def on_camera(self, sender: Object, args: RoutedEventArgs) -> None:
        self.toggle_camera()

    def on_wheel(self, sender: Object, args: PointerRoutedEventArgs) -> None:
        self.zoom(closer=args.get_current_point(self.ui.hud).properties.mouse_wheel_delta > 0)

    def on_stop(self, sender: Object, args: RoutedEventArgs) -> None:
        self.stop()

    def on_reset(self, sender: Object, args: RoutedEventArgs) -> None:
        self.reset()

    def on_listen_toggled(self, sender: Object, args: RoutedEventArgs) -> None:
        self.set_listening(self.ui.listen.is_on)

    def on_closed(self, sender: Object, args: WindowEventArgs) -> None:
        if self.remember_placement:
            try:
                placement.save(self.hwnd)
            except OSError as e:
                log.warning("couldn't save the window's placement: %s", e)
        self.shutdown()


class App(Application, IXamlMetadataProvider):
    def __init__(self, options: argparse.Namespace) -> None:
        super().__init__()  # with constructor arguments of our own, this creates the WinRT object
        self._provider = XamlControlsXamlMetaDataProvider()
        self.options = options

    @override
    def _on_launched(self, args: LaunchActivatedEventArgs) -> None:
        self.resources.merged_dictionaries.append(XamlControlsResources())
        # Errors XAML reports here would otherwise go unseen.
        self.add_unhandled_exception(lambda s, e: log.error("XAML: %s (0x%08x)", e.message, e.exception & 0xFFFFFFFF))
        self.voice_app = WindowsApp(self, self.options)

    # ----- XAML metadata ------------------------------------------------------

    @override
    def get_xaml_type(self, type: TypeName | tuple[str, TypeKind]) -> IXamlType:
        return self._provider.get_xaml_type(type)

    @override
    def get_xaml_type_by_full_name(self, full_name: str) -> IXamlType:
        return self._provider.get_xaml_type_by_full_name(full_name)

    @override
    def get_xmlns_definitions(self) -> Array[XmlnsDefinition]:
        return self._provider.get_xmlns_definitions()


def main() -> None:
    options = parse_args()
    log.info("3D view on %s (D3D12)", use_d3d12())
    # python.exe declares no DPI awareness, so Windows would stretch the window.
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per monitor v2
    init_apartment(ApartmentType.SINGLE_THREADED)
    apps: list[App] = []
    with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
        Application.start(lambda _: apps.append(App(options)))
        for app in apps:
            app.voice_app.loop.close()
