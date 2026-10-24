"""Whisper-Small on the Hexagon NPU, through ONNX Runtime's QNN provider.

Qualcomm ships the model precompiled for the Snapdragon X Elite (QAIRT 2.50):
an encoder that turns 30 s of log-mel features into per-layer cross-attention
caches, and a single-step decoder with fixed-size self-attention caches.
Everything at the edges is quantized; the scales are in metadata.json.

Only the onnxruntime-qnn wheel (QNN 2.51) loads these binaries today; Windows
ML's QNN provider (QNN 2.48) refuses them.
"""

from __future__ import annotations

import functools
import json
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn

MODEL = pathlib.Path(__file__).resolve().parent.parent / "models" / "whisper-small"

SAMPLE_RATE, N_FFT, HOP, N_MELS, FRAMES = 16000, 400, 160, 80, 3000
SOT, EN, TRANSCRIBE, NO_TIMESTAMPS, EOT, FIRST_TIMESTAMP = 50258, 50259, 50359, 50363, 50257, 50364
MASK_NEG = -100.0


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
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(printable, chars)}


class Vocabulary:
    """Token ids back to text, without the tokenizers package."""

    def __init__(self, path: pathlib.Path) -> None:
        table = json.loads(path.read_text(encoding="utf-8"))
        self.tokens = {i: token for token, i in table.items()}

    def decode(self, ids: list[int]) -> str:
        pieces = "".join(self.tokens.get(i, "") for i in ids if i < EOT)  # specials start at EOT
        decoder = _byte_decoder()
        return bytes(decoder[c] for c in pieces if c in decoder).decode("utf-8", errors="replace")


# ----- model ---------------------------------------------------------------

@dataclass
class Timing:
    mel_ms: float
    encoder_ms: float
    decoder_ms: float
    steps: int

    @property
    def total_ms(self) -> float:
        return self.mel_ms + self.encoder_ms + self.decoder_ms


class Whisper:
    def __init__(self, model: pathlib.Path = MODEL) -> None:
        try:
            ort.register_execution_provider_library(onnxruntime_qnn.EP_NAME, onnxruntime_qnn.get_library_path())
        except Exception as e:  # already registered in this process
            if "already registered" not in str(e):
                raise
        npu = [d for d in ort.get_ep_devices()
               if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.NPU]
        if not npu:
            raise RuntimeError("no NPU found for the QNN execution provider")

        meta = json.loads((model / "metadata.json").read_text())["model_files"]
        specs = {name: spec for part in meta.values() for side in ("inputs", "outputs")
                 for name, spec in part[side].items()}
        self.quant = {name: (s["quantization_parameters"]["scale"], s["quantization_parameters"]["zero_point"])
                      for name, s in specs.items() if "quantization_parameters" in s}
        self.dtypes = {name: s["dtype"] for name, s in specs.items()}
        self.decoder_inputs = meta["decoder.onnx"]["inputs"]
        self.max_len = self.decoder_inputs["attention_mask"]["shape"][-1]

        opts = ort.SessionOptions()
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        opts.add_provider_for_devices(npu, {"htp_performance_mode": "burst"})
        t = time.perf_counter()
        self.encoder = ort.InferenceSession(str(model / "encoder.onnx"), sess_options=opts)
        self.decoder = ort.InferenceSession(str(model / "decoder.onnx"), sess_options=opts)
        self.load_s = time.perf_counter() - t
        self.encoder_outputs = [o.name for o in self.encoder.get_outputs()]
        self.decoder_outputs = [o.name for o in self.decoder.get_outputs()]
        self.vocabulary = Vocabulary(model / "vocab.json")

    def _q(self, name: str, x: np.ndarray) -> np.ndarray:
        scale, zero = self.quant[name]
        info = np.iinfo(np.dtype(self.dtypes[name]))
        return np.clip(np.round(x / scale + zero), info.min, info.max).astype(self.dtypes[name])

    def _dq(self, name: str, x: np.ndarray) -> np.ndarray:
        scale, zero = self.quant[name]
        return (x.astype(np.float32) - zero) * scale

    def _carry(self, src: str, dst: str, x: np.ndarray) -> np.ndarray:
        return x if self.quant[src] == self.quant[dst] else self._q(dst, self._dq(src, x))

    def transcribe(self, audio: np.ndarray, max_tokens: int = 64) -> tuple[str, Timing]:
        """Text of up to 30 s of 16 kHz mono float32 audio."""
        t0 = time.perf_counter()
        mel = log_mel(audio)
        t1 = time.perf_counter()
        enc = dict(zip(self.encoder_outputs,
                       self.encoder.run(None, {"input_features": self._q("input_features", mel)})))
        t2 = time.perf_counter()

        feeds = {name: self._carry(name, name, value) for name, value in enc.items()}
        for name, spec in self.decoder_inputs.items():
            if name.startswith(("k_cache_self", "v_cache_self")):
                feeds[name] = np.full(spec["shape"], self.quant[name][1], np.uint8)  # zeros

        mask = np.full((1, 1, 1, self.max_len), MASK_NEG, np.float32)
        prompt = [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
        tokens: list[int] = []
        steps = 0
        for n in range(min(self.max_len - 1, len(prompt) + max_tokens)):
            token = prompt[n] if n < len(prompt) else tokens[-1]
            mask[..., self.max_len - n - 1] = 0.0
            feeds["input_ids"] = np.array([[token]], np.int32)
            feeds["position_ids"] = np.array([n], np.int32)
            feeds["attention_mask"] = self._q("attention_mask", mask)
            out = dict(zip(self.decoder_outputs, self.decoder.run(None, feeds)))
            steps += 1
            for name, value in out.items():
                if name.endswith("_out"):
                    target = name.removesuffix("_out") + "_in"
                    feeds[target] = self._carry(name, target, value)
            if n < len(prompt) - 1:
                continue
            logits = self._dq("logits", out["logits"]).reshape(-1)
            logits[FIRST_TIMESTAMP:] = -np.inf
            nxt = int(np.argmax(logits))
            if nxt == EOT:
                break
            tokens.append(nxt)
        t3 = time.perf_counter()
        timing = Timing((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, steps)
        return self.vocabulary.decode(tokens).strip(), timing
