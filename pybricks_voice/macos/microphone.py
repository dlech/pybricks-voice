"""
The microphone on macOS, through AVAudioEngine: a tap on its input node, with
voice processing on, as Windows' speech category has it (echo cancellation,
noise suppression and gain control). Each buffer's samples are read into
NumPy through the pointers floatChannelData gives.

Opening it takes a few hundred milliseconds with voice processing (341 to
473 ms measured on an M1 Pro), so it opens on a thread of its own rather
than stalling the UI. AVAudioEngine stops when the input device changes,
such as when headphones are connected; the microphone then opens again on
whatever is now the default.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

assert not TYPE_CHECKING or sys.platform == "darwin"

import numpy as np
from rubicon.objc import Block, ObjCClass, ObjCInstance, objc_const, objc_id
from rubicon.objc.runtime import load_library

log = logging.getLogger(__name__)

_avfoundation = load_library("AVFoundation")
AVAudioEngine = ObjCClass("AVAudioEngine")
AVCaptureDevice = ObjCClass("AVCaptureDevice")
NSNotificationCenter = ObjCClass("NSNotificationCenter")
AVMediaTypeAudio = objc_const(_avfoundation, "AVMediaTypeAudio")
AVAudioEngineConfigurationChangeNotification = objc_const(_avfoundation, "AVAudioEngineConfigurationChangeNotification")

# From the SDK's headers, by hand until generated bindings have them.
AVAuthorizationStatusNotDetermined, AVAuthorizationStatusAuthorized = 0, 3
AVAudioVoiceProcessingOtherAudioDuckingLevelMin = 10

# Asked for, but the tap gives 100 ms buffers at 44.1 kHz whatever this says
# (measured on an M1 Pro, macOS 26).
_BUFFER_FRAMES = 1024


def _error_text(error: objc_id) -> str:
    return str(ObjCInstance(error).localizedDescription) if error else "unknown error"


class AVAudioEngineMicrophone:
    """The default input device, as mono float32 at its own rate."""

    # Off, the mic hears what this Mac plays too, such as `say` for testing.
    voice_processing = True

    def __init__(self, on_samples: Callable[[np.ndarray], None]) -> None:
        self.on_samples = on_samples
        self.rate = 0
        self._engine: Any = None
        self._opening = False
        self._observer: Any = None
        self._reopening: asyncio.Task[None] | None = None
        self._energy = np.zeros(0)
        # CoreAudio and Foundation hold only pointers to these.
        self._tap = Block(self._buffer, None, objc_id, objc_id)
        self._changed = Block(self._configuration_changed, None, objc_id)

    @property
    def is_open(self) -> bool:
        return self._engine is not None or self._opening

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._opening = True
        try:
            await self._authorize()
            await asyncio.to_thread(self._open)
        finally:
            self._opening = False

    def stop(self) -> None:
        self._close()

    async def _authorize(self) -> None:
        """Asks for the Microphone permission, the first time."""
        status = AVCaptureDevice.authorizationStatusForMediaType(AVMediaTypeAudio)
        if status == AVAuthorizationStatusAuthorized:
            return
        if status != AVAuthorizationStatusNotDetermined:
            raise RuntimeError("access is off in System Settings, Privacy & Security, Microphone")
        granted = self._loop.create_future()

        def answered(yes: bool) -> None:  # on a queue of AVFoundation's
            self._loop.call_soon_threadsafe(lambda: granted.done() or granted.set_result(yes))

        self._answered = Block(answered, None, ctypes.c_bool)
        AVCaptureDevice.requestAccessForMediaType(AVMediaTypeAudio, completionHandler=self._answered)
        if not await granted:
            raise RuntimeError("access was denied")

    def _open(self) -> None:
        engine = AVAudioEngine.alloc().init()
        node = engine.inputNode
        error = objc_id()
        if self.voice_processing:
            if node.setVoiceProcessingEnabled(True, error=ctypes.byref(error)):
                # It turns other apps' sound down while it runs: as little as it allows.
                node.voiceProcessingOtherAudioDuckingConfiguration = (
                    True,
                    AVAudioVoiceProcessingOtherAudioDuckingLevelMin,
                )
            else:
                log.warning("no voice processing: %s", _error_text(error))
        audio_format = node.outputFormatForBus(0)
        self.rate, channels = round(audio_format.sampleRate), int(audio_format.channelCount)
        self._energy = np.zeros(channels)
        log.info("microphone: %d Hz × %d channels", self.rate, channels)
        node.installTapOnBus(0, bufferSize=_BUFFER_FRAMES, format=None, block=self._tap)
        engine.prepare()
        if not engine.startAndReturnError(ctypes.byref(error)):
            node.removeTapOnBus(0)
            raise RuntimeError(_error_text(error))
        self._engine = engine
        # Only this engine's changes, from now on: setting it up changes it too.
        self._observer = NSNotificationCenter.defaultCenter.addObserverForName(
            AVAudioEngineConfigurationChangeNotification, object=engine, queue=None, usingBlock=self._changed
        )

    def _close(self) -> None:
        if self._observer is not None:
            NSNotificationCenter.defaultCenter.removeObserver(self._observer)
            self._observer = None
        if self._engine is not None:
            self._engine.inputNode.removeTapOnBus(0)
            self._engine.stop()
            self._engine = None

    def _buffer(self, buffer: objc_id, when: objc_id) -> None:
        """On a thread of AVAudioEngine's, for each buffer the tap gives."""
        pcm = ObjCInstance(buffer)
        frames, data = int(pcm.frameLength), pcm.floatChannelData
        if not frames or len(self._energy) == 0:
            return
        samples = np.stack([np.ctypeslib.as_array(data[c], shape=(frames,)) for c in range(len(self._energy))])
        # Voice processing gives the same signal on every channel (9 of them on
        # an M1 Pro); another device may have the voice on one. The channel with
        # the most energy is the one with the voice, as on Windows.
        self._energy = 0.95 * self._energy + 0.05 * np.mean(samples * samples, axis=1)
        self.on_samples(samples[int(np.argmax(self._energy))].copy())

    def _configuration_changed(self, notification: objc_id) -> None:
        """The input device changed, and the engine stopped: open the new default."""
        self._loop.call_soon_threadsafe(self._reopen_soon)

    def _reopen_soon(self) -> None:
        if self._engine is not None and self._reopening is None:
            self._reopening = self._loop.create_task(self._reopen())

    async def _reopen(self) -> None:
        log.info("the input device changed")
        self._close()
        self._opening = True
        try:
            await asyncio.to_thread(self._open)
        except RuntimeError as e:
            log.warning("couldn't open the microphone again: %s", e)
        finally:
            self._opening = False
            self._reopening = None
