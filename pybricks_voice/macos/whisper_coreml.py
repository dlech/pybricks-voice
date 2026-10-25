"""
Whisper-Small through Core ML, with WhisperKit's models (argmaxinc/whisperkit-coreml,
MIT): an encoder, and a decoder that takes one token at a time and leaves its
caches to the caller. whisper.py's features, greedy decoding and vocabulary do
the rest.

Loading for the Neural Engine compiles the model for this Mac's Neural Engine
the first time, which takes minutes (97 to 148 s for the encoder alone on an
M1 Pro); Core ML keeps the result, and later loads take about 0.3 s. So
CoreMLWhisper starts on the CPU, which loads in about 2 s, prepares the
Neural Engine on a thread of its own, and switches to it when it's ready.

Core ML is called through rubicon-objc, since coremltools has no Python 3.14
build; the arrays are NumPy's memory, which MLMultiArray wraps without copying.
"""

from __future__ import annotations

import ctypes
import logging
import pathlib
import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

assert not TYPE_CHECKING or sys.platform == "darwin"

import numpy as np
from rubicon.objc import ObjCClass, ObjCInstance, objc_id
from rubicon.objc.runtime import load_library

from ..whisper import MODEL as VOCABULARY
from ..whisper import Timing, Vocabulary, decode_greedy, log_mel

log = logging.getLogger(__name__)

MODEL = pathlib.Path(__file__).resolve().parent.parent.parent / "models" / "whisper-small-coreml"

load_library("CoreML")
MLDictionaryFeatureProvider = ObjCClass("MLDictionaryFeatureProvider")
MLModel = ObjCClass("MLModel")
MLModelConfiguration = ObjCClass("MLModelConfiguration")
MLMultiArray = ObjCClass("MLMultiArray")
NSURL = ObjCClass("NSURL")

# From the SDK's headers, by hand until generated bindings have them.
MLComputeUnitsCPUOnly, MLComputeUnitsCPUAndNeuralEngine = 0, 3
_DATA_TYPES = {np.dtype(np.float16): 0x10000 | 16, np.dtype(np.int32): 0x20000 | 32}  # MLMultiArrayDataType

# The decoder's caches: 12 layers × 768 channels, for 224 positions.
_CACHE_CHANNELS, _CACHE_LENGTH = 9216, 224
_NO_ATTENTION = -10000.0  # a padding mask's value for a position not yet decoded


def _error_text(error: objc_id) -> str:
    return str(ObjCInstance(error).localizedDescription) if error else "unknown error"


def load(path: pathlib.Path, units: int) -> Any:
    """A compiled model (.mlmodelc), on these compute units."""
    configuration = MLModelConfiguration.alloc().init()
    configuration.computeUnits = units
    error = objc_id()
    model = MLModel.modelWithContentsOfURL(
        NSURL.fileURLWithPath(str(path)), configuration=configuration, error=ctypes.byref(error)
    )
    if model is None:
        raise RuntimeError(f"couldn't load {path.name}: {_error_text(error)}")
    return model


def _wrap(array: np.ndarray) -> Any:
    """An MLMultiArray over array's memory, which must outlive it."""
    error = objc_id()
    wrapped = MLMultiArray.alloc().initWithDataPointer(
        array.ctypes.data,
        shape=list(array.shape),
        dataType=_DATA_TYPES[array.dtype],
        strides=[s // array.itemsize for s in array.strides],
        deallocator=None,
        error=ctypes.byref(error),
    )
    if wrapped is None:
        raise RuntimeError(f"MLMultiArray: {_error_text(error)}")
    return wrapped


def _unwrap(multi_array: Any) -> np.ndarray:
    """A float16 MLMultiArray's contents, copied out."""
    shape = [int(n.integerValue) for n in multi_array.shape]
    strides = [int(n.integerValue) for n in multi_array.strides]
    span = sum((n - 1) * s for n, s in zip(shape, strides, strict=True)) + 1
    pointer = multi_array.dataPointer
    address = pointer if isinstance(pointer, int) else pointer.value
    flat = np.frombuffer((ctypes.c_uint16 * span).from_address(address), np.float16)
    return np.lib.stride_tricks.as_strided(flat, shape, [s * 2 for s in strides]).copy()


def _predict(model: Any, features: Any) -> Any:
    error = objc_id()
    result = model.predictionFromFeatures(features, error=ctypes.byref(error))
    if result is None:
        raise RuntimeError(f"prediction: {_error_text(error)}")
    return result


def _features(arrays: dict[str, Any]) -> Any:
    error = objc_id()
    provider = MLDictionaryFeatureProvider.alloc().initWithDictionary(arrays, error=ctypes.byref(error))
    if provider is None:
        raise RuntimeError(f"features: {_error_text(error)}")
    return provider


class _Models:
    """The encoder and the decoder, loaded for one set of compute units."""

    def __init__(self, model: pathlib.Path, units: int, device: str) -> None:
        self.device = device
        self.encoder = load(model / "AudioEncoder.mlmodelc", units)
        self.decoder = load(model / "TextDecoder.mlmodelc", units)

    def transcribe(self, audio: np.ndarray, vocabulary: Vocabulary, max_tokens: int) -> tuple[str, Timing]:
        t0 = time.perf_counter()
        mel = log_mel(audio)[:, :, None, :].astype(np.float16)  # (1, 80, 1, 3000)
        t1 = time.perf_counter()
        encoded = _predict(self.encoder, _features({"melspectrogram_features": _wrap(mel)}))
        embeds = encoded.featureValueForName("encoder_output_embeds").multiArrayValue
        t2 = time.perf_counter()

        # Each step's inputs, changed in place: the features wrap their memory.
        token = np.zeros(1, np.int32)
        position = np.zeros(1, np.int32)
        keys = np.zeros((1, _CACHE_CHANNELS, 1, _CACHE_LENGTH), np.float16)
        values = np.zeros_like(keys)
        update = np.zeros((1, _CACHE_LENGTH), np.float16)  # 1 where this step's keys and values go
        padding = np.full((1, _CACHE_LENGTH), _NO_ATTENTION, np.float16)
        inputs = {
            "input_ids": token,
            "cache_length": position,
            "key_cache": keys,
            "value_cache": values,
            "kv_cache_update_mask": update,
            "decoder_key_padding_mask": padding,
        }
        features = _features({**{name: _wrap(a) for name, a in inputs.items()}, "encoder_output_embeds": embeds})

        def step(t: int, n: int) -> np.ndarray:
            token[0], position[0] = t, n
            update[:] = 0
            update[0, n] = 1
            padding[0, n] = 0
            out = _predict(self.decoder, features)
            keys[..., n] = _unwrap(out.featureValueForName("key_cache_updates").multiArrayValue)[..., 0]
            values[..., n] = _unwrap(out.featureValueForName("value_cache_updates").multiArrayValue)[..., 0]
            return _unwrap(out.featureValueForName("logits").multiArrayValue)

        tokens, steps = decode_greedy(step, _CACHE_LENGTH, max_tokens)
        t3 = time.perf_counter()
        timing = Timing((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, steps, self.device)
        return vocabulary.decode(tokens).strip(), timing


class CoreMLWhisper:
    """
    Whisper on the CPU at once, and on the Neural Engine once it's ready:
    on_neural_engine(seconds) is called then, from a thread of its own.
    """

    def __init__(
        self, model: pathlib.Path = MODEL, on_neural_engine: Callable[[float], object] = lambda seconds: None
    ) -> None:
        self.model = model
        self.on_neural_engine = on_neural_engine
        t = time.perf_counter()
        self.vocabulary = Vocabulary(VOCABULARY / "vocab.json")
        self._models = _Models(model, MLComputeUnitsCPUOnly, "on the CPU")
        self.load_s = time.perf_counter() - t
        threading.Thread(target=self._prepare_neural_engine, name="neural-engine", daemon=True).start()

    @property
    def device(self) -> str:
        return self._models.device

    def transcribe(self, audio: np.ndarray, max_tokens: int = 64) -> tuple[str, Timing]:
        """Text of up to 30 s of 16 kHz mono float32 audio."""
        return self._models.transcribe(audio, self.vocabulary, max_tokens)

    def _prepare_neural_engine(self) -> None:
        t = time.perf_counter()
        try:
            models = _Models(self.model, MLComputeUnitsCPUAndNeuralEngine, "on the Neural Engine")
        except RuntimeError:
            log.exception("Whisper stays on the CPU")
            return
        self._models = models  # the next transcription uses it
        seconds = time.perf_counter() - t
        log.info("Whisper on the Neural Engine, prepared in %.1f s", seconds)
        self.on_neural_engine(seconds)
