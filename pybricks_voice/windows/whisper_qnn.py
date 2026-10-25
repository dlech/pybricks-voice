"""
Whisper-Small on the Hexagon NPU, through ONNX Runtime's QNN provider.

Qualcomm ships the model precompiled for the Snapdragon X Elite (QAIRT 2.50):
an encoder that turns 30 s of log-mel features into per-layer cross-attention
caches, and a single-step decoder with fixed-size self-attention caches.
Everything at the edges is quantized; the scales are in metadata.json.

Only the onnxruntime-qnn wheel (QNN 2.51) loads these binaries today; Windows
ML's QNN provider (QNN 2.48) refuses them.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import TYPE_CHECKING

assert not TYPE_CHECKING or sys.platform == "win32"

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn

from ..whisper import MODEL, Timing, Vocabulary, decode_greedy, log_mel

MASK_NEG = -100.0


def _run(session: ort.InferenceSession, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    All of a session's outputs by name. ONNX Runtime types them as arrays,
    sparse tensors, sequences or maps; Whisper's are all plain arrays.
    """
    names = [o.name for o in session.get_outputs()]
    outputs = session.run(names, feeds)
    arrays = {}
    for name, value in zip(names, outputs, strict=True):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} is a {type(value).__name__}, not an array")
        arrays[name] = value
    return arrays


class Whisper:
    device = "on the NPU"

    def __init__(self, model: pathlib.Path = MODEL) -> None:
        try:
            ort.register_execution_provider_library(onnxruntime_qnn.EP_NAME, onnxruntime_qnn.get_library_path())
        except Exception as e:  # already registered in this process
            if "already registered" not in str(e):
                raise
        npu = [
            d
            for d in ort.get_ep_devices()
            if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.NPU
        ]
        if not npu:
            raise RuntimeError("no NPU found for the QNN execution provider")

        meta = json.loads((model / "metadata.json").read_text())["model_files"]
        specs = {
            name: spec for part in meta.values() for side in ("inputs", "outputs") for name, spec in part[side].items()
        }
        self.quant = {
            name: (s["quantization_parameters"]["scale"], s["quantization_parameters"]["zero_point"])
            for name, s in specs.items()
            if "quantization_parameters" in s
        }
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
        enc = _run(self.encoder, {"input_features": self._q("input_features", mel)})
        t2 = time.perf_counter()

        feeds = {name: self._carry(name, name, value) for name, value in enc.items()}
        for name, spec in self.decoder_inputs.items():
            if name.startswith(("k_cache_self", "v_cache_self")):
                feeds[name] = np.full(spec["shape"], self.quant[name][1], np.uint8)  # zeros

        mask = np.full((1, 1, 1, self.max_len), MASK_NEG, np.float32)

        def step(token: int, n: int) -> np.ndarray:
            mask[..., self.max_len - n - 1] = 0.0
            feeds["input_ids"] = np.array([[token]], np.int32)
            feeds["position_ids"] = np.array([n], np.int32)
            feeds["attention_mask"] = self._q("attention_mask", mask)
            out = _run(self.decoder, feeds)
            for name, value in out.items():
                if name.endswith("_out"):
                    target = name.removesuffix("_out") + "_in"
                    feeds[target] = self._carry(name, target, value)
            return self._dq("logits", out["logits"])

        tokens, steps = decode_greedy(step, self.max_len, max_tokens)
        t3 = time.perf_counter()
        timing = Timing((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, steps, self.device)
        return self.vocabulary.decode(tokens).strip(), timing
