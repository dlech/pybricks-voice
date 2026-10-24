"""The window: a 3D simulated robot you drive by voice, with Whisper on the NPU.

The UI thread runs asyncio as a guest of WinUI's DispatcherQueue (see
loop.py), so the app is coroutines that touch XAML directly: one steps the
simulator 60 times a second and sends the pose to the WebView2 scene, another
reacts to what was heard. Two threads of their own do the rest:
  audio thread  AudioGraph quanta -> the voice-activity segmenter
  whisper       transcribes each utterance on the NPU (a daemon thread)
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import pathlib
import sys
import time
import traceback
import wave
from collections.abc import Callable
from typing import override
from xml.sax.saxutils import quoteattr

import numpy as np
from winrt.microsoft.ui.xaml import (
    Application,
    DispatcherShutdownMode,
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
from winrt.windows.graphics import SizeInt32
from winrt.windows.system import VirtualKey
from winrt.windows.ui.xaml.interop import TypeKind, TypeName

from .grammar import describe, suggest
from .loop import DispatcherQueueEventLoop
from .robot import SimRobot
from .speech import Heard, Listener, Microphone, Segmenter
from .whisper import SAMPLE_RATE

ROOT = pathlib.Path(__file__).resolve().parent
WEB = ROOT / "web"
USER_DATA = ROOT.parent / ".webview2"
HOST = "robot.local"
LEVEL_FALL_DB_S = 30.0

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
        self.scene_ready = asyncio.Event()
        self.whisper_ready = asyncio.Event()
        self.chase = True
        self.last_tick = time.perf_counter()
        self.tick_gaps: list[float] = []
        self.shown_pending: list | None = None
        self.latencies: list[float] = []
        self.level_db = -120.0

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

        # Closing the window starts shutdown(), and the app exits once main() has
        # unwound, which takes the loop a few more iterations.
        self.dispatcher_shutdown_mode = DispatcherShutdownMode.ON_EXPLICIT_SHUTDOWN
        self.loop = DispatcherQueueEventLoop(self.window.dispatcher_queue)
        self.loop.start()
        self.listener = Listener(self.loop)
        self.listener.record_dir = self.options.record
        self.microphone = Microphone(lambda samples: self.listener.feed(samples, self.microphone.rate))
        self.main_task = self.loop.create_task(self.main())
        self.main_task.add_done_callback(self.finished)
        self.window.activate()

    async def main(self) -> None:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(self.frames())
            tasks.create_task(self.scene())
            tasks.create_task(self.speech())
            if self.options.feed_wav:
                tasks.create_task(self.feed_wavs())
            if self.options.say or self.options.quit_after:
                tasks.create_task(self.script())

    async def script(self) -> None:
        """--say, --screenshot and --quit-after: drive the app without a person."""
        started = time.perf_counter()
        await self.scene_ready.wait()
        for text in self.options.say:
            self.listener.say(text)
        if self.options.quit_after:
            await asyncio.sleep(self.options.quit_after - (time.perf_counter() - started))
            if self.options.screenshot:
                self.screenshot(self.options.screenshot)
            self.shutdown()

    def screenshot(self, path: pathlib.Path) -> None:
        from PIL import ImageGrab

        pos, size = self.window.app_window.position, self.window.app_window.size
        path.parent.mkdir(parents=True, exist_ok=True)
        ImageGrab.grab((pos.x, pos.y, pos.x + size.width, pos.y + size.height), all_screens=True).save(path)

    async def scene(self) -> None:
        USER_DATA.mkdir(exist_ok=True)
        try:
            environment = await CoreWebView2Environment.create_with_options_async("", str(USER_DATA), None)
            await self.webview.ensure_core_webview2_async(environment)
        except OSError as e:
            self.set_status(f"WebView2 failed: {e}")
            return
        core = self.webview.core_webview2
        core.set_virtual_host_name_to_folder_mapping(HOST, str(WEB), CoreWebView2HostResourceAccessKind.ALLOW)
        core.add_web_message_received(lambda s, e: self.on_scene_message(e.web_message_as_json))
        core.navigate(f"https://{HOST}/scene.html")

    async def speech(self) -> None:
        self.set_status("loading Whisper on the NPU…")
        load_s = await self.listener.start()
        self.whisper_ready.set()
        self.set_status(f"listening (Whisper loaded in {load_s:.1f} s)")
        if not self.options.no_mic:
            try:
                await self.microphone.start()
            except Exception as e:
                self.set_status(f"no microphone: {e}")
        async for heard in self.listener.heard():
            self.on_heard(heard)

    async def feed_wavs(self) -> None:
        """--feed-wav: plays files into the listener at real-time pace, with room
        noise between them, so the segmenter and Whisper see what a mic would give."""
        rng = np.random.default_rng(0)

        async def stream(audio: np.ndarray) -> None:
            # Whatever is due by the clock, in pieces of at least 10 ms.
            start, fed = time.perf_counter(), 0
            while fed < len(audio):
                due = min(len(audio), int((time.perf_counter() - start) * SAMPLE_RATE))
                if due - fed >= SAMPLE_RATE // 100 or due == len(audio):
                    self.listener.feed(audio[fed:due], SAMPLE_RATE)
                    fed = due
                await asyncio.sleep(0.01)

        def room(seconds: float) -> np.ndarray:
            return (rng.standard_normal(int(SAMPLE_RATE * seconds)) * 10 ** (-60 / 20)).astype(np.float32)

        await self.whisper_ready.wait()
        for path in self.options.feed_wav:
            with wave.open(str(path)) as wav:
                speech = np.frombuffer(wav.readframes(wav.getnframes()), np.int16).astype(np.float32) / 32768
            await stream(room(1.5))
            await stream(speech + room(len(speech) / SAMPLE_RATE))
        await stream(room(2))

    # ----- every frame --------------------------------------------------------

    async def frames(self) -> None:
        """Steps the simulator and draws, 60 times a second, on the clock: a late
        frame makes the next one sooner rather than pushing them all back."""
        period = 1 / 60
        next_frame = time.perf_counter()
        while True:
            self.tick()
            next_frame = max(next_frame + period, time.perf_counter() - period)
            await asyncio.sleep(next_frame - time.perf_counter())

    def tick(self) -> None:
        now = time.perf_counter()
        self.tick_gaps.append(now - self.last_tick)
        dt, self.last_tick = min(now - self.last_tick, 0.1), now
        self.robot.step(dt)
        s = self.robot.state()
        self.post_to_scene(
            {
                "pose": {
                    "x": s.x,
                    "y": s.y,
                    "heading": s.heading,
                    "left": s.left_angle,
                    "right": s.right_angle,
                    "speed": s.speed,
                }
            }
        )
        pending = ([s.current] if s.current else []) + s.queued
        if pending != self.shown_pending:  # only when it changes, not 60 times a second
            self.shown_pending = pending
            self.queue_text.text = ("Now: " + describe(pending)) if pending else "Waiting for a command"
        seg = self.listener.segmenter
        if seg is not None:
            # Like a VU meter: up at once, back down at 30 dB/s, rather than
            # flickering with every 30 ms frame.
            self.level_db = max(seg.level_db, self.level_db - LEVEL_FALL_DB_S * dt)
            bar = "█" * max(0, min(12, int((self.level_db + 70) / 4)))
            text = f"{'● ' if seg.speaking else '  '}{bar:<12} {self.level_db:5.0f} dB"
            if text != self.level.text:
                self.level.text = text

    # ----- events -------------------------------------------------------------

    def on_scene_message(self, message: str) -> None:
        if json.loads(message).get("ready"):
            self.scene_ready.set()

    def on_heard(self, heard: Heard) -> None:
        self.robot.run(heard.parse.commands)
        if heard.audio_s:
            self.latencies.append(heard.latency_ms)
            pause_ms = Segmenter.END_MS
            detail = (
                f"Whisper {heard.timing.total_ms:.0f} ms on the NPU · "
                f"moving {pause_ms + heard.latency_ms:.0f} ms after you stopped"
            )
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
            look = (
                'HorizontalAlignment="Right" CornerRadius="16,16,4,16" '
                'Background="{ThemeResource AccentFillColorDefaultBrush}"'
            )
            fore = 'Foreground="{ThemeResource TextOnAccentFillColorPrimaryBrush}"'
        else:
            look = (
                'HorizontalAlignment="Left" CornerRadius="16,16,16,4" BorderThickness="1" '
                'Background="{ThemeResource CardBackgroundFillColorDefaultBrush}" '
                'BorderBrush="{ThemeResource CardStrokeColorDefaultBrush}"'
            )
            fore = ""
        extra = (
            f'<TextBlock Text={quoteattr(detail)} FontSize="11" Opacity="0.75" TextWrapping="Wrap" {fore}/>'
            if detail
            else ""
        )
        bubble = XamlReader.load(
            f'<Border xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation" {look} '
            f'MaxWidth="290" Padding="12,8,12,8">'
            f'<StackPanel Spacing="2"><TextBlock Text={quoteattr(text)} TextWrapping="Wrap" {fore}/>{extra}'
            f"</StackPanel></Border>"
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
        self.post_to_scene({"reset": True})

    def toggle_camera(self) -> None:
        self.chase = not self.chase
        self.post_to_scene({"camera": "chase" if self.chase else "top"})

    def post_to_scene(self, message: dict) -> None:
        if not self.scene_ready.is_set():
            return
        try:
            self.webview.core_webview2.post_web_message_as_json(json.dumps(message))
        except OSError:
            # The WebView2 closes as the window does, before the window's closed
            # event; a frame drawn in between mustn't take the app down with it.
            self.scene_ready.clear()

    def set_status(self, text: str) -> None:
        self.status.text = text

    def shutdown(self) -> None:
        """Stops everything; the app exits once main() has finished."""
        self.main_task.cancel()

    def finished(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            traceback.print_exception(error)
        gaps = np.array(self.tick_gaps[30:]) * 1000  # skip start-up
        if len(gaps):
            print(
                f"UI timer (16 ms): median gap {np.median(gaps):.1f} ms, p99 {np.percentile(gaps, 99):.1f} ms, "
                f"max {gaps.max():.1f} ms, {int((gaps > 50).sum())} gaps over 50 ms",
                flush=True,
            )
        if self.latencies:
            pause = Segmenter.END_MS
            print(
                f"end of speech -> commands: median {pause + np.median(self.latencies):.0f} ms "
                f"({pause} ms pause + {np.median(self.latencies):.0f} ms), "
                f"max {pause + max(self.latencies):.0f} ms over {len(self.latencies)} utterances",
                flush=True,
            )
        self.microphone.stop()
        self.listener.close()
        self.loop.stop()
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
    parser.add_argument(
        "--say", action="append", default=[], help="type a command once the app is up (repeatable), for demos and tests"
    )
    parser.add_argument("--screenshot", type=pathlib.Path, help="save the window to this PNG before quitting")
    parser.add_argument("--quit-after", type=float, help="quit after this many seconds")
    parser.add_argument(
        "--feed-wav",
        action="append",
        default=[],
        type=pathlib.Path,
        help="stream a 16 kHz WAV into the listener in real time, as if spoken (repeatable)",
    )
    parser.add_argument("--no-mic", action="store_true", help="don't open the microphone")
    parser.add_argument("--record", type=pathlib.Path, help="save every utterance (WAV + text) to this folder")
    options = parser.parse_args()

    # python.exe declares no DPI awareness, so Windows would stretch the window.
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per monitor v2
    init_apartment(ApartmentType.SINGLE_THREADED)
    apps: list[App] = []
    with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
        Application.start(lambda _: apps.append(App(options)))
        for app in apps:
            app.loop.close()
