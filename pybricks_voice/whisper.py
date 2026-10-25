"""
Whisper's features, decoding and tokens, the parts every backend shares: 30 s
of 16 kHz audio as log-mel features in, greedy decoding one token at a time,
token ids back to text out. Running the model is each platform's:
windows/whisper_qnn.py on the Snapdragon's NPU, macos/whisper_coreml.py
through Core ML.
"""

from __future__ import annotations

import functools
import json
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

MODEL = pathlib.Path(__file__).resolve().parent.parent / "models" / "whisper-small"

SAMPLE_RATE, N_FFT, HOP, N_MELS, FRAMES = 16000, 400, 160, 80, 3000
SOT, EN, TRANSCRIBE, NO_TIMESTAMPS, EOT, FIRST_TIMESTAMP = 50258, 50259, 50359, 50363, 50257, 50364
PROMPT = (SOT, EN, TRANSCRIBE, NO_TIMESTAMPS)  # English, transcribe, no timestamps


# ----- features ------------------------------------------------------------


@functools.cache
def mel_filters() -> np.ndarray:
    """librosa.filters.mel(sr=16000, n_fft=400, n_mels=80): Slaney scale and norm."""

    def hz_to_mel(f):
        f = np.asarray(f, np.float64)
        return np.where(f >= 1000, 15 + np.log(np.maximum(f, 1e-10) / 1000) / (np.log(6.4) / 27), f / (200 / 3))

    def mel_to_hz(m):
        m = np.asarray(m, np.float64)
        return np.where(m >= 15, 1000 * np.exp((np.log(6.4) / 27) * (m - 15)), m * (200 / 3))

    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    weights = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]))
    weights *= (2.0 / (mel_f[2 : N_MELS + 2] - mel_f[:N_MELS]))[:, None]
    return weights.astype(np.float32)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Whisper's features: 80 log-mel bins x 3000 frames (30 s, zero padded)."""
    audio = np.pad(audio[: SAMPLE_RATE * 30], (0, max(0, SAMPLE_RATE * 30 - len(audio))))
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP][:FRAMES]
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    log = np.log10(np.maximum(mel_filters() @ power.T, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)[None]


# ----- tokens --------------------------------------------------------------


@functools.cache
def _byte_decoder() -> dict[str, int]:
    """GPT-2's byte-level BPE alphabet: each byte has a printable stand-in."""
    printable = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(printable, chars, strict=True)}


class Vocabulary:
    """Token ids back to text, without the tokenizers package."""

    def __init__(self, path: pathlib.Path) -> None:
        table = json.loads(path.read_text(encoding="utf-8"))
        self.tokens = {i: token for token, i in table.items()}

    def decode(self, ids: list[int]) -> str:
        pieces = "".join(self.tokens.get(i, "") for i in ids if i < EOT)  # specials start at EOT
        decoder = _byte_decoder()
        return bytes(decoder[c] for c in pieces if c in decoder).decode("utf-8", errors="replace")


# ----- decoding --------------------------------------------------------------


def decode_greedy(step: Callable[[int, int], np.ndarray], max_len: int, max_tokens: int = 64) -> tuple[list[int], int]:
    """
    Whisper's greedy decoding, without timestamps, for a decoder that takes one
    token at a time and keeps its own caches: step(token, position) runs it
    and returns the logits. Returns the tokens, and how many steps it took.
    """
    tokens: list[int] = []
    steps = 0
    # The last prompt token's step gives the first token: max_tokens more steps from there.
    for n in range(min(max_len - 1, len(PROMPT) - 1 + max_tokens)):
        logits = step(PROMPT[n] if n < len(PROMPT) else tokens[-1], n)
        steps += 1
        if n < len(PROMPT) - 1:
            continue  # the prompt's own predictions don't matter
        logits = logits.reshape(-1).astype(np.float32)
        logits[FIRST_TIMESTAMP:] = -np.inf
        token = int(np.argmax(logits))
        if token == EOT:
            break
        tokens.append(token)
    return tokens, steps


# ----- model ---------------------------------------------------------------


@dataclass
class Timing:
    mel_ms: float
    encoder_ms: float
    decoder_ms: float
    steps: int
    device: str = ""  # where it ran, for showing: "on the NPU"

    @property
    def total_ms(self) -> float:
        return self.mel_ms + self.encoder_ms + self.decoder_ms


class Transcriber(Protocol):
    """A Whisper model loaded on some device, for speech.Listener."""

    load_s: float  # how long loading took
    device: str  # where it runs, for showing: "on the NPU"

    def transcribe(self, audio: np.ndarray) -> tuple[str, Timing]:
        """Text of up to 30 s of 16 kHz mono float32 audio."""
        ...
