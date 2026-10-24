"""Microphone -> utterances -> Whisper -> commands, off the UI thread.

The microphone comes in through WinRT's AudioGraph; each AudioFrame's buffer
is read into NumPy through PyWinRT 4.0's buffer protocol. A small
voice-activity detector cuts the stream into utterances by loudness against
the room's own noise floor, and each utterance is transcribed once.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .grammar import Parse, parse
from .whisper import SAMPLE_RATE, Timing, Whisper


# ----- voice activity ------------------------------------------------------

class Segmenter:
    """Cuts a stream of samples into utterances.

    Speech starts after START_MS of frames louder than the noise floor by
    MARGIN_DB, and ends after END_MS of quieter ones. PREROLL_MS before the
    start is kept so the first syllable isn't clipped.
    """

    # START_MS spans 4 frames, so a click (which can touch 3) never starts one.
    FRAME_MS, START_MS, END_MS, PREROLL_MS, MAX_S = 30, 120, 600, 300, 12.0
    MARGIN_DB, MIN_DB = 12.0, -50.0

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.frame = rate * self.FRAME_MS // 1000
        self._pending = np.zeros(0, np.float32)
        self._preroll: deque[np.ndarray] = deque(maxlen=self.PREROLL_MS // self.FRAME_MS)
        self._history: deque[float] = deque(maxlen=3000 // self.FRAME_MS)  # 3 s of frame levels
        self._speech: list[np.ndarray] | None = None
        self._loud = self._quiet = 0
        self.level_db = -120.0
        self.threshold_db = self.MIN_DB

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        """Adds samples; returns any utterances that ended, at the input rate."""
        done = []
        self._pending = np.concatenate([self._pending, samples])
        while len(self._pending) >= self.frame:
            frame, self._pending = self._pending[: self.frame], self._pending[self.frame :]
            utterance = self._frame(frame)
            if utterance is not None:
                done.append(utterance)
        return done

    def _frame(self, frame: np.ndarray) -> np.ndarray | None:
        self.level_db = 20 * np.log10(np.sqrt(np.mean(frame * frame)) + 1e-9)
        if self._speech is None:
            self._history.append(self.level_db)
        floor = np.percentile(self._history, 20) if self._history else -120.0
        self.threshold_db = max(floor + self.MARGIN_DB, self.MIN_DB)
        loud = self.level_db > self.threshold_db

        if self._speech is None:
            self._preroll.append(frame)
            self._loud = self._loud + 1 if loud else 0
            if self._loud * self.FRAME_MS >= self.START_MS:
                self._speech = list(self._preroll)
                self._quiet = 0
            return None

        self._speech.append(frame)
        self._quiet = 0 if loud else self._quiet + 1
        length_s = len(self._speech) * self.FRAME_MS / 1000
        if self._quiet * self.FRAME_MS >= self.END_MS or length_s >= self.MAX_S:
            utterance = np.concatenate(self._speech)
            self._speech, self._loud = None, 0
            self._preroll.clear()
            return utterance
        return None

    @property
    def speaking(self) -> bool:
        return self._speech is not None


def resample(audio: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if rate == target:
        return audio.astype(np.float32)
    t = np.arange(0, len(audio) / rate, 1 / target)
    return np.interp(t, np.arange(len(audio)) / rate, audio).astype(np.float32)


# ----- microphone ----------------------------------------------------------

class Microphone:
    """The default capture device through AudioGraph, as mono float32."""

    def __init__(self, on_samples: Callable[[np.ndarray], None]) -> None:
        self.on_samples = on_samples
        self.graph = None
        self.rate = 0

    async def start(self) -> None:
        from winrt.windows.media import AudioBufferAccessMode  # noqa: F401 (used in _quantum)
        from winrt.windows.media.audio import (
            AudioDeviceNodeCreationStatus,
            AudioGraph,
            AudioGraphCreationStatus,
            AudioGraphSettings,
        )
        from winrt.windows.media.capture import MediaCategory
        from winrt.windows.media.render import AudioRenderCategory

        result = await AudioGraph.create_async(AudioGraphSettings(AudioRenderCategory.SPEECH))
        if result.status != AudioGraphCreationStatus.SUCCESS:
            raise RuntimeError(f"AudioGraph: {result.status.name}")
        self.graph = result.graph
        node = await self.graph.create_device_input_node_async(MediaCategory.SPEECH)
        if node.status != AudioDeviceNodeCreationStatus.SUCCESS:
            raise RuntimeError(f"microphone: {node.status.name}")
        self.output = self.graph.create_frame_output_node()
        node.device_input_node.add_outgoing_connection(self.output)
        props = self.output.encoding_properties
        self.rate, self.channels = props.sample_rate, props.channel_count
        self.graph.add_quantum_started(self._quantum)
        self.graph.start()

    def _quantum(self, graph, _) -> None:
        from winrt.windows.media import AudioBufferAccessMode

        frame = self.output.get_frame()
        with (
            frame.lock_buffer(AudioBufferAccessMode.READ) as buffer,
            buffer.create_reference() as reference,
            memoryview(reference) as view,  # the buffer protocol, new in PyWinRT 4.0
        ):
            # Copied out while the view is open: PyWinRT won't close a reference
            # whose memory an array still holds.
            mono = np.frombuffer(view, np.float32)[: buffer.length // 4].reshape(-1, self.channels).mean(axis=1)
        self.on_samples(mono)

    def stop(self) -> None:
        if self.graph is not None:
            self.graph.stop()
            self.graph.close()
            self.graph = None


# ----- the pipeline --------------------------------------------------------

_FILLERS = {"um", "uh", "hmm", "mm", "ah", "oh", "er", "erm", "you", "thank you", "thanks", "bye"}


def is_noise(text: str) -> bool:
    """What Whisper says for a room rather than a person: [BLANK_AUDIO],
    (clicking), "Um…", or the "Thank you." it is known to invent for silence."""
    words = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*", " ", text)
    words = re.sub(r"[^a-z' ]", " ", words.lower())
    words = " ".join(words.split())
    return not words or words in _FILLERS or all(w in _FILLERS for w in words.split())

@dataclass
class Heard:
    text: str
    parse: Parse
    audio_s: float
    timing: Timing
    latency_ms: float  # end of speech -> commands ready


class Listener:
    """Owns Whisper on a thread of its own; feed it audio, get Heard back."""

    def __init__(self, on_heard: Callable[[Heard], None], on_status: Callable[[str], None] = print) -> None:
        self.on_heard = on_heard
        self.on_status = on_status
        self._utterances: queue.Queue[tuple[np.ndarray, int, float] | str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="whisper", daemon=True)
        self.segmenter: Segmenter | None = None
        self.muted = False

    def start(self) -> None:
        self._thread.start()

    def feed(self, samples: np.ndarray, rate: int) -> None:
        """From the audio thread: cut into utterances and queue them."""
        if self.segmenter is None or self.segmenter.rate != rate:
            self.segmenter = Segmenter(rate)
        for utterance in self.segmenter.feed(samples):
            if not self.muted:
                self._utterances.put((utterance, rate, time.perf_counter()))

    def say(self, text: str) -> None:
        """Typed input takes the same path, minus Whisper."""
        self._utterances.put(text)

    def close(self) -> None:
        self._utterances.put(None)

    def _run(self) -> None:
        self.on_status("loading Whisper on the NPU…")
        whisper = Whisper()
        self.on_status(f"listening (Whisper loaded in {whisper.load_s:.1f} s)")
        while (item := self._utterances.get()) is not None:
            if isinstance(item, str):
                self.on_heard(Heard(item, parse(item), 0.0, Timing(0, 0, 0, 0), 0.0))
                continue
            audio, rate, ended = item
            text, timing = whisper.transcribe(resample(audio, rate))
            if not is_noise(text):
                self.on_heard(Heard(text, parse(text), len(audio) / rate, timing,
                                    (time.perf_counter() - ended) * 1000))
