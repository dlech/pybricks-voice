"""The window: a 3D simulated robot you drive by voice, with Whisper on the NPU.

Threads:
  UI thread     WinUI, a 60 Hz timer that steps the simulator and sends the
                pose to the WebView2 scene, and everything that touches XAML
  audio thread  AudioGraph quanta -> the voice-activity segmenter
  whisper       transcribes each utterance on the NPU and parses it, then
                hands the result to the UI thread through its DispatcherQueue
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import datetime
import json
import pathlib
import sys
import threading
import time
import traceback
import wave
from typing import Callable
from xml.sax.saxutils import quoteattr

import numpy as np
from typing_extensions import override
from winrt.microsoft.ui.xaml import (
    Application,
    ApplicationInitializationCallbackParams,
    FrameworkElement,
    LaunchActivatedEventArgs,
    UIElement,
    Window,
)
from winrt.microsoft.ui.xaml.controls import (
    Button,
    ScrollViewer,
    StackPanel,
    TextBlock,
    TextBox,
    ToggleSwitch,
    WebView2,
    XamlControlsResources,
)
from winrt.microsoft.ui.xaml.input import KeyRoutedEventArgs
from winrt.microsoft.ui.xaml.markup import IXamlMetadataProvider, IXamlType, XamlReader, XmlnsDefinition
from winrt.microsoft.ui.xaml.xamltypeinfo import XamlControlsXamlMetaDataProvider
from winrt.microsoft.web.webview2.core import CoreWebView2Environment, CoreWebView2HostResourceAccessKind
from winrt.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import InitializeOptions, initialize
from winrt.runtime import ApartmentType, init_apartment
from winrt.system import Array
from winrt.windows.foundation import AsyncStatus
from winrt.windows.graphics import SizeInt32
from winrt.windows.system import VirtualKey
from winrt.windows.ui.xaml.interop import TypeKind, TypeName

from .grammar import describe, suggest
from .robot import SimRobot
from .speech import Heard, Listener, Microphone, Segmenter
from .whisper import SAMPLE_RATE

ROOT = pathlib.Path(__file__).resolve().parent
WEB = ROOT / "web"
USER_DATA = ROOT.parent / ".webview2"
HOST = "robot.local"

XAML = """
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation">
    <Window.SystemBackdrop><MicaBackdrop/></Window.SystemBackdrop>
    <Grid>
        <Grid.ColumnDefinitions>
            <ColumnDefinition Width="*"/>
            <ColumnDefinition Width="380"/>
        </Grid.ColumnDefinitions>
        <WebView2 Name="scene"/>
        <Grid Grid.Column="1" Padding="16" RowSpacing="12">
            <Grid.RowDefinitions>
                <RowDefinition Height="Auto"/>
                <RowDefinition Height="Auto"/>
                <RowDefinition Height="Auto"/>
                <RowDefinition Height="Auto"/>
                <RowDefinition Height="*"/>
                <RowDefinition Height="Auto"/>
            </Grid.RowDefinitions>
            <StackPanel Spacing="4">
                <TextBlock Text="Talk to the robot" Style="{StaticResource SubtitleTextBlockStyle}"/>
                <TextBlock Name="status" Opacity="0.7" TextWrapping="Wrap"/>
            </StackPanel>
            <StackPanel Grid.Row="1" Orientation="Horizontal" Spacing="12">
                <ToggleSwitch Name="listen" OnContent="Listening" OffContent="Muted" IsOn="True"/>
                <TextBlock Name="level" VerticalAlignment="Center" Opacity="0.6" FontFamily="Consolas"/>
            </StackPanel>
            <Grid Grid.Row="2" ColumnSpacing="8">
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                </Grid.ColumnDefinitions>
                <TextBox Name="typed" PlaceholderText="…or type: forward 30 cm then turn left"/>
                <Button Name="go" Grid.Column="1" Content="Go" Style="{StaticResource AccentButtonStyle}"/>
            </Grid>
            <StackPanel Grid.Row="3" Orientation="Horizontal" Spacing="8">
                <Button Name="stop" Content="Stop"/>
                <Button Name="reset" Content="Reset"/>
                <Button Name="camera" Content="Camera"/>
            </StackPanel>
            <ScrollViewer Name="scroller" Grid.Row="4">
                <StackPanel Name="log" Spacing="8" Padding="0,0,12,0"/>
            </ScrollViewer>
            <TextBlock Name="queue" Grid.Row="5" Opacity="0.7" TextWrapping="Wrap" MaxLines="3"
                       TextTrimming="CharacterEllipsis"/>
        </Grid>
    </Grid>
</Window>
"""


def logged(fn: Callable) -> Callable:
    """XAML callbacks that raise end the process with no traceback; print it."""

    def wrapper(*args):
        try:
            return fn(*args)
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            raise

    return wrapper


class App(Application, IXamlMetadataProvider):
    def __init__(self, options: argparse.Namespace) -> None:
        super().__init__()  # with constructor arguments of our own, this creates the WinRT object
        self._provider = XamlControlsXamlMetaDataProvider()
        self.options = options
        self.robot = SimRobot()
        self.scene_ready = False
        self.chase = True
        self.last_tick = time.perf_counter()
        self.tick_gaps: list[float] = []
        self.shown_pending: list | None = None
        self.latencies: list[float] = []

    # ----- start up -----------------------------------------------------------

    @override
    @logged
    def _on_launched(self, args: LaunchActivatedEventArgs) -> None:
        self.resources.merged_dictionaries.append(XamlControlsResources())
        self.window = XamlReader.load(XAML).as_(Window)
        self.window.title = "Pybricks voice driving · Whisper on the NPU"
        self.window.app_window.resize(SizeInt32(1600, 1000))
        content = self.window.content.as_(FrameworkElement)
        find = content.find_name
        self.webview = find("scene").as_(WebView2)
        self.status = find("status").as_(TextBlock)
        self.level = find("level").as_(TextBlock)
        self.listen = find("listen").as_(ToggleSwitch)
        self.typed = find("typed").as_(TextBox)
        self.log = find("log").as_(StackPanel)
        self.scroller = find("scroller").as_(ScrollViewer)
        self.queue_text = find("queue").as_(TextBlock)
        # Newest message at the bottom, as in a messaging app: follow the
        # conversation once each new bubble has been measured.
        self.log.add_size_changed(lambda s, e: self.scroller.change_view(None, self.scroller.scrollable_height, None))

        find("go").as_(Button).add_click(lambda s, e: self.submit_typed())
        find("stop").as_(Button).add_click(lambda s, e: self.robot.stop())
        find("reset").as_(Button).add_click(lambda s, e: self.reset())
        find("camera").as_(Button).add_click(lambda s, e: self.toggle_camera())
        self.typed.add_key_down(self.on_typed_key)
        self.listen.add_toggled(lambda s, e: setattr(self.listener, "muted", not self.listen.is_on))
        self.window.add_closed(lambda s, e: self.shutdown())

        self.dispatcher = self.window.dispatcher_queue
        self.start_scene()
        self.start_speech()

        self.timer = self.dispatcher.create_timer()
        self.timer.interval = datetime.timedelta(milliseconds=16)
        self.timer.add_tick(lambda t, e: self.tick())
        self.timer.start()
        self.window.activate()
        self.start_script()

    def start_script(self) -> None:
        """--say, --screenshot and --quit-after: drive the app without a person."""
        if not (self.options.say or self.options.quit_after):
            return
        started = time.perf_counter()
        said = False

        @logged
        def check(timer, _) -> None:
            nonlocal said
            if not said and self.scene_ready:
                for text in self.options.say:
                    self.listener.say(text)
                said = True
            quit_after = self.options.quit_after
            if quit_after and time.perf_counter() - started >= quit_after:
                timer.stop()
                if self.options.screenshot:
                    self.screenshot(self.options.screenshot)
                self.shutdown()

        self.script_timer = self.dispatcher.create_timer()
        self.script_timer.interval = datetime.timedelta(milliseconds=250)
        self.script_timer.add_tick(check)
        self.script_timer.start()

    def screenshot(self, path: pathlib.Path) -> None:
        from PIL import ImageGrab

        pos, size = self.window.app_window.position, self.window.app_window.size
        path.parent.mkdir(parents=True, exist_ok=True)
        ImageGrab.grab((pos.x, pos.y, pos.x + size.width, pos.y + size.height), all_screens=True).save(path)

    def start_scene(self) -> None:
        USER_DATA.mkdir(exist_ok=True)
        op = CoreWebView2Environment.create_with_options_async("", str(USER_DATA), None)

        @logged
        def on_environment(op, status) -> None:
            if status != AsyncStatus.COMPLETED:
                self.set_status(f"WebView2 failed: {status.name}")
                return
            self.webview.ensure_core_webview2_async(op.get_results()).completed = on_core

        @logged
        def on_core(op, status) -> None:
            if status != AsyncStatus.COMPLETED:
                self.set_status(f"WebView2 failed: {status.name}")
                return
            core = self.webview.core_webview2
            core.set_virtual_host_name_to_folder_mapping(HOST, str(WEB), CoreWebView2HostResourceAccessKind.ALLOW)
            core.add_web_message_received(lambda s, e: self.on_scene_message(e.web_message_as_json))
            core.navigate(f"https://{HOST}/scene.html")

        op.completed = on_environment

    def start_speech(self) -> None:
        def post(callback: Callable[[], None]) -> None:
            self.dispatcher.try_enqueue(logged(callback))

        self.listener = Listener(
            on_heard=lambda heard: post(lambda: self.on_heard(heard)),
            on_status=lambda text: post(lambda: self.set_status(text)),
        )
        self.listener.record_dir = self.options.record
        self.listener.start()
        self.microphone = Microphone(lambda samples: self.listener.feed(samples, self.microphone.rate))
        if self.options.feed_wav:
            threading.Thread(target=self.feed_wavs, name="feed-wav", daemon=True).start()
        if self.options.no_mic:
            return

        def open_microphone() -> None:
            init_apartment(ApartmentType.MULTI_THREADED)
            try:
                asyncio.run(self.microphone.start())
            except Exception as e:
                post(lambda: self.set_status(f"no microphone: {e}"))

        threading.Thread(target=open_microphone, name="microphone", daemon=True).start()

    def feed_wavs(self) -> None:
        """--feed-wav: plays files into the listener at real-time pace, with room
        noise between them, so the segmenter and Whisper see what a mic would give."""
        rng = np.random.default_rng(0)
        chunk = SAMPLE_RATE // 100  # 10 ms

        def stream(audio: np.ndarray) -> None:
            for i in range(0, len(audio), chunk):
                self.listener.feed(audio[i : i + chunk], SAMPLE_RATE)
                time.sleep(chunk / SAMPLE_RATE)

        def room(seconds: float) -> np.ndarray:
            return (rng.standard_normal(int(SAMPLE_RATE * seconds)) * 10 ** (-60 / 20)).astype(np.float32)

        time.sleep(3)  # let Whisper load
        for path in self.options.feed_wav:
            with wave.open(str(path)) as wav:
                speech = np.frombuffer(wav.readframes(wav.getnframes()), np.int16).astype(np.float32) / 32768
            stream(room(1.5))
            self.fed_end = time.perf_counter() + len(speech) / SAMPLE_RATE
            stream(speech + room(len(speech) / SAMPLE_RATE))
        stream(room(2))

    # ----- every frame --------------------------------------------------------

    @logged
    def tick(self) -> None:
        now = time.perf_counter()
        self.tick_gaps.append(now - self.last_tick)
        dt, self.last_tick = min(now - self.last_tick, 0.1), now
        self.robot.step(dt)
        s = self.robot.state()
        if self.scene_ready:
            self.webview.core_webview2.post_web_message_as_json(json.dumps({"pose": {
                "x": s.x, "y": s.y, "heading": s.heading, "left": s.left_angle,
                "right": s.right_angle, "speed": s.speed}}))
        pending = ([s.current] if s.current else []) + s.queued
        if pending != self.shown_pending:  # only when it changes, not 60 times a second
            self.shown_pending = pending
            self.queue_text.text = ("Now: " + describe(pending)) if pending else "Waiting for a command"
        seg = self.listener.segmenter
        if seg is not None:
            bar = "█" * max(0, min(12, int((seg.level_db + 70) / 4)))
            self.level.text = f"{'● ' if seg.speaking else '  '}{bar:<12} {seg.level_db:5.0f} dB"

    # ----- events -------------------------------------------------------------

    def on_scene_message(self, message: str) -> None:
        if json.loads(message).get("ready"):
            self.scene_ready = True

    def on_heard(self, heard: Heard) -> None:
        self.robot.run(heard.parse.commands)
        if heard.audio_s:
            self.latencies.append(heard.latency_ms)
            pause_ms = Segmenter.END_MS
            detail = (f"Whisper {heard.timing.total_ms:.0f} ms on the NPU · "
                      f"moving {pause_ms + heard.latency_ms:.0f} ms after you stopped")
        else:
            detail = "typed"
        self.add_bubble(heard.text, mine=True, detail=detail)

        replies = list(heard.parse.remarks)
        if heard.parse.commands:
            replies.append(describe(heard.parse.commands) + ".")
        for segment in heard.parse.not_understood:
            tries = " or ".join(f"“{s}”" for s in suggest(segment))
            replies.append(f"I didn't get “{segment}”. Try {tries}.")
        if replies:  # "Oh my" on its own needs no answer
            self.add_bubble("\n".join(replies), mine=False)

    def add_bubble(self, text: str, mine: bool, detail: str | None = None) -> None:
        """A message bubble: yours on the right in the accent colour, the robot's on the left."""
        if mine:
            look = ('HorizontalAlignment="Right" CornerRadius="16,16,4,16" '
                    'Background="{ThemeResource AccentFillColorDefaultBrush}"')
            fore = 'Foreground="{ThemeResource TextOnAccentFillColorPrimaryBrush}"'
        else:
            look = ('HorizontalAlignment="Left" CornerRadius="16,16,16,4" BorderThickness="1" '
                    'Background="{ThemeResource CardBackgroundFillColorDefaultBrush}" '
                    'BorderBrush="{ThemeResource CardStrokeColorDefaultBrush}"')
            fore = ""
        extra = (f'<TextBlock Text={quoteattr(detail)} FontSize="11" Opacity="0.75" TextWrapping="Wrap" {fore}/>'
                 if detail else "")
        bubble = XamlReader.load(
            f'<Border xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation" {look} '
            f'MaxWidth="290" Padding="12,8,12,8">'
            f'<StackPanel Spacing="2"><TextBlock Text={quoteattr(text)} TextWrapping="Wrap" {fore}/>{extra}'
            f'</StackPanel></Border>'
        ).as_(UIElement)
        self.log.children.append(bubble)
        while len(self.log.children) > 100:
            self.log.children.pop(0)

    def submit_typed(self) -> None:
        text = self.typed.text.strip()
        if text:
            self.listener.say(text)
            self.typed.text = ""

    def on_typed_key(self, sender, e: KeyRoutedEventArgs) -> None:
        if e.key == VirtualKey.ENTER:
            self.submit_typed()

    def reset(self) -> None:
        self.robot.reset()
        if self.scene_ready:
            self.webview.core_webview2.post_web_message_as_json('{"reset": true}')

    def toggle_camera(self) -> None:
        self.chase = not self.chase
        if self.scene_ready:
            self.webview.core_webview2.post_web_message_as_json(
                json.dumps({"camera": "chase" if self.chase else "top"}))

    def set_status(self, text: str) -> None:
        self.status.text = text

    def shutdown(self) -> None:
        if getattr(self, "closing", False):
            return  # exit() closes the window, which calls this again
        self.closing = True
        self.timer.stop()
        gaps = np.array(self.tick_gaps[30:]) * 1000  # skip start-up
        if len(gaps):
            print(f"UI timer (16 ms): median gap {np.median(gaps):.1f} ms, p99 {np.percentile(gaps, 99):.1f} ms, "
                  f"max {gaps.max():.1f} ms, {int((gaps > 50).sum())} gaps over 50 ms", flush=True)
        if self.latencies:
            pause = Segmenter.END_MS
            print(f"end of speech -> commands: median {pause + np.median(self.latencies):.0f} ms "
                  f"({pause} ms pause + {np.median(self.latencies):.0f} ms), "
                  f"max {pause + max(self.latencies):.0f} ms over {len(self.latencies)} utterances", flush=True)
        self.microphone.stop()
        self.listener.close()
        self.exit()

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
    parser = argparse.ArgumentParser(prog="python -m pybricks_voice")
    parser.add_argument("--say", action="append", default=[],
                        help="type a command once the app is up (repeatable), for demos and tests")
    parser.add_argument("--screenshot", type=pathlib.Path, help="save the window to this PNG before quitting")
    parser.add_argument("--quit-after", type=float, help="quit after this many seconds")
    parser.add_argument("--feed-wav", action="append", default=[], type=pathlib.Path,
                        help="stream a 16 kHz WAV into the listener in real time, as if spoken (repeatable)")
    parser.add_argument("--no-mic", action="store_true", help="don't open the microphone")
    parser.add_argument("--record", type=pathlib.Path, help="save every utterance (WAV + text) to this folder")
    options = parser.parse_args()

    # python.exe declares no DPI awareness, so Windows would stretch the window.
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per monitor v2
    init_apartment(ApartmentType.SINGLE_THREADED)
    with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
        Application.start(lambda _: App(options))
