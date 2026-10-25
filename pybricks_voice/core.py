"""
The app, less its window: what each platform's app shares.

The UI thread runs asyncio, as a guest of the platform's own event loop, so
the app is coroutines that touch the window directly: one updates the HUD 60
times a second, another reacts to what was heard. Threads of their own do the
rest:
  render        steps the simulator and the physics and draws them with pygfx,
                60 times a second
  audio thread  the microphone -> the voice-activity segmenter
  whisper       transcribes each utterance (a daemon thread)

A platform's app subclasses Core: it opens the window, its microphone and its
Whisper, shows what Core puts in the *_line attributes and add_bubble(), and
calls the handlers below from its controls.
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import logging
import math
import pathlib
import threading
import time
import wave
from collections.abc import Callable
from typing import Protocol

import numpy as np

from .grammar import describe, suggest
from .physics import Physics, World
from .robot import SimRobot, State
from .scene import RobotScene
from .speech import Heard, Listener, Microphone, Segmenter
from .whisper import SAMPLE_RATE, Transcriber

log = logging.getLogger(__name__)

# The level meter: how quickly it follows the microphone up and down, in
# seconds, and its range.
LEVEL_RISE_S, LEVEL_FALL_S = 0.05, 0.35
LEVEL_MIN_DB, LEVEL_MAX_DB = -70.0, -10.0


class SceneView(Protocol):
    """A platform's 3D view: RobotScene drawn by a thread of its own."""

    scene: RobotScene
    frame_gaps: list[float]  # between steps, in seconds
    frames_drawn: int

    def reset(self) -> None: ...

    def close(self) -> None: ...


class Core(abc.ABC):
    # Where Whisper runs, for the status line and the chat: "on the NPU".
    WHISPER_ON: str

    view: SceneView  # set by the subclass before start()

    def __init__(self, options: argparse.Namespace) -> None:
        self.options = options
        self.robot = SimRobot()
        self.physics = Physics()
        self.physics_lock = threading.Lock()  # stepped by the render thread, reset by the UI
        self.last_position = np.zeros(2)
        self.frame: tuple[World, State, float] | None = None  # the latest, from the render thread
        self.scene_ready = asyncio.Event()
        self.whisper_ready = asyncio.Event()
        self.whisper_tried = asyncio.Event()  # loaded or not: the microphone can open
        self.last_tick = time.perf_counter()
        self.tick_gaps: list[float] = []
        self.shown_pending: list | None = None
        self.latencies: list[float] = []
        self.level_db = -120.0
        self.level_text_at = 0.0
        self.listening = True
        # What the window shows; update_view() each tick shows whichever has changed.
        self.hud_line = ""
        self.status_line = ""
        self.listen_line = "Listening"
        self.level_value = LEVEL_MIN_DB
        self.level_line = ""
        self.queue_line = "Waiting for a command"

    # ----- what each platform provides ----------------------------------------

    @abc.abstractmethod
    def make_microphone(self, on_samples: Callable[[np.ndarray], None]) -> Microphone: ...

    @abc.abstractmethod
    def load_whisper(self) -> Transcriber:
        """On the Whisper thread."""

    @abc.abstractmethod
    def update_view(self) -> None:
        """Shows the *_line attributes and level_value, each tick."""

    @abc.abstractmethod
    def add_bubble(self, text: str, mine: bool, detail: str | None = None) -> None:
        """A message in the chat: yours on the right, the robot's on the left."""

    @abc.abstractmethod
    async def save_screenshot(self, path: pathlib.Path) -> None: ...

    @abc.abstractmethod
    def exit(self) -> None:
        """Once everything has stopped: stop the loop and close the app."""

    # ----- start up -----------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Once the window and the view are up, with the loop running."""
        self.loop = loop
        self.listener = Listener(loop, self.load_whisper)
        self.listener.record_dir = self.options.record
        self.microphone = self.make_microphone(lambda samples: self.listener.feed(samples, self.microphone.rate))
        self.main_task = loop.create_task(self.main())
        self.main_task.add_done_callback(self.finished)

    async def main(self) -> None:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(self.frames())
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
                async with asyncio.timeout(2):
                    await self.save_screenshot(self.options.screenshot)
            self.shutdown()

    async def speech(self) -> None:
        self.set_status(f"loading Whisper {self.WHISPER_ON}…")
        try:
            load_s = await self.listener.start()
        except Exception as e:
            # Typed commands still work.
            log.warning("no Whisper: %s", e)
            self.set_status(f"no Whisper: {e}")
        else:
            self.whisper_ready.set()
            self.set_status(f"Whisper loaded in {load_s:.1f} s")
        # Without Whisper, the level meter still shows what the microphone hears.
        self.whisper_tried.set()
        if self.listening:
            await self.open_microphone()
        async for heard in self.listener.heard():
            self.on_heard(heard)

    async def open_microphone(self) -> None:
        if self.options.no_mic or self.microphone.is_open:
            return
        try:
            await self.microphone.start()
        except Exception as e:
            self.set_status(f"no microphone: {e}")
            return
        if not self.listening:  # switched off while it was opening
            self.microphone.stop()

    def set_listening(self, on: bool) -> None:
        """
        Off closes the microphone, so the system no longer shows it in use,
        and drops whatever was half heard; on opens it again.
        """
        self.listening = on
        self.listen_line = "Listening" if on else "Muted"
        self.listener.muted = not on
        if on:
            if self.whisper_tried.is_set():
                self.microphone_task = self.loop.create_task(self.open_microphone())
            return
        self.microphone.stop()
        self.listener.segmenter = None
        self.level_db = -120.0
        self.level_value = LEVEL_MIN_DB
        self.level_line = ""

    async def feed_wavs(self) -> None:
        """
        --feed-wav: plays files into the listener at real-time pace, with room
        noise between them, so the segmenter and Whisper see what a mic would give.
        """
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

    def step(self, dt: float) -> World:
        """
        On the render thread, each frame: SimRobot turns commands into the
        angle each wheel should be at; the motors in the physics follow them,
        and what the robot really does comes from there.
        """
        self.robot.step(dt)
        s = self.robot.state()
        with self.physics_lock:
            self.physics.step(dt, s.left_angle, s.right_angle)
            world = self.physics.world()
        position = world.robot.position[:2] * 100  # cm
        speed = float(np.hypot(*(position - self.last_position))) / dt if dt else 0.0
        self.last_position = position
        self.frame = (world, s, speed)
        return world

    async def frames(self) -> None:
        """
        Updates the HUD and the level meter 60 times a second, on the clock: a
        late frame makes the next one sooner rather than pushing them all back.
        """
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
        seg = self.listener.segmenter
        if seg is not None and not self.listener.muted:
            self.show_level(seg.level_db, dt, now)
        if self.frame is not None:
            world, s, speed = self.frame
            position = world.robot.position[:2] * 100  # cm
            # Rounded first, then + 0.0, which turns -0.0 into 0.0: no "-0.0 cm" or "-0°".
            x, y = (round(float(v), 1) + 0.0 for v in position)
            self.hud_line = f"x {x:.1f} cm · y {y:.1f} cm · heading {round(world.heading)}° · {speed:.0f} cm/s"
            pending = ([s.current] if s.current else []) + s.queued
            if pending != self.shown_pending:  # described only when it changes, not 60 times a second
                self.shown_pending = pending
                self.queue_line = ("Now: " + describe(pending)) if pending else "Waiting for a command"
        self.update_view()

    def show_level(self, level_db: float, dt: float, now: float) -> None:
        """
        The microphone level, smoothed like a VU meter: it follows each 30 ms
        frame quickly on the way up and slowly on the way down, so room noise and
        the gaps between syllables don't make it flicker.
        """
        tau = LEVEL_RISE_S if level_db > self.level_db else LEVEL_FALL_S
        self.level_db += (max(level_db, -120.0) - self.level_db) * (1 - math.exp(-dt / tau))
        # In quarter decibels, which the bar can't show finer anyway.
        self.level_value = round(min(max(self.level_db, LEVEL_MIN_DB), LEVEL_MAX_DB) * 4) / 4
        # The number only a few times a second: digits changing 60 times a
        # second are unreadable.
        if now - self.level_text_at >= 0.25:
            self.level_text_at = now
            self.level_line = f"{round(self.level_db)} dBFS"

    # ----- events -------------------------------------------------------------

    def on_heard(self, heard: Heard) -> None:
        self.robot.run(heard.parse.commands)
        if heard.audio_s:
            self.latencies.append(heard.latency_ms)
            pause_ms = Segmenter.END_MS
            detail = (
                f"Whisper {heard.timing.total_ms:.0f} ms {self.WHISPER_ON} · "
                f"moving {pause_ms + heard.latency_ms:.0f} ms after you stopped"
                if self.options.timing
                else None
            )
        else:
            detail = "typed"
        self.add_bubble(heard.text, mine=True, detail=detail)

        # The robot answers remarks and what it didn't understand, without
        # repeating what was said: that is in the bubble just above, and a
        # command needs no echo, since the robot does it and the queue line
        # names it.
        replies = list(heard.parse.remarks)
        if heard.parse.not_understood:
            what = "all of that" if heard.parse.commands else "that"
            tries = " or ".join(f"“{s}”" for s in suggest(" ".join(heard.parse.not_understood)))
            replies.append(f"I didn't get {what}. Try {tries}.")
        if replies:  # "Oh my" on its own needs no answer
            self.add_bubble("\n".join(replies), mine=False)

    def submit(self, text: str) -> bool:
        """Typed input; whether there was any."""
        text = text.strip()
        if text:
            self.listener.say(text)
        return bool(text)

    def toggle_camera(self) -> None:
        self.view.scene.chase = not self.view.scene.chase

    def zoom(self, closer: bool) -> None:
        scene = self.view.scene
        scene.zoom = min(4.0, max(0.3, scene.zoom * (0.9 if closer else 1.1)))

    def stop(self) -> None:
        self.robot.stop()

    def reset(self) -> None:
        self.robot.reset()
        with self.physics_lock:
            self.physics.reset()
        self.view.reset()

    def set_status(self, text: str) -> None:
        self.status_line = text

    def shutdown(self) -> None:
        """Stops everything; the app exits once main() has finished."""
        self.main_task.cancel()

    def finished(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            log.error("the app stopped on an error", exc_info=error)
        self.view.close()
        for name, gaps in (("UI timer", self.tick_gaps), ("3D view", self.view.frame_gaps)):
            gaps = np.array(gaps[30:]) * 1000  # skip start-up
            if len(gaps):
                log.info(
                    "%s (16 ms): median gap %.1f ms, p99 %.1f ms, max %.1f ms, %d gaps over 50 ms",
                    name,
                    np.median(gaps),
                    np.percentile(gaps, 99),
                    gaps.max(),
                    (gaps > 50).sum(),
                )
        steps = len(self.view.frame_gaps)
        if steps:
            log.info("3D view: drew %d of %d frames, the rest unchanged", self.view.frames_drawn, steps)
        if self.latencies:
            pause, median = Segmenter.END_MS, float(np.median(self.latencies))
            log.info(
                "end of speech -> commands: median %.0f ms (%d ms pause + %.0f ms), max %.0f ms over %d utterances",
                pause + median,
                pause,
                median,
                pause + max(self.latencies),
                len(self.latencies),
            )
        self.microphone.stop()
        self.listener.close()
        self.exit()


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--timing", action="store_true", help="show Whisper's timings in the chat, for developing")
    parser.add_argument("--debug", action="store_true", help="log at DEBUG rather than INFO")
    options = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if options.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return options
