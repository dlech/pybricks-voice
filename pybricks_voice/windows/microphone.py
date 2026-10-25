"""
The microphone on Windows, through WinRT's AudioGraph: each AudioFrame's
buffer is read into NumPy through PyWinRT 4.0's buffer protocol.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

assert not TYPE_CHECKING or sys.platform == "win32"

import numpy as np

log = logging.getLogger(__name__)


class AudioGraphMicrophone:
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
        self.energy = np.zeros(self.channels)
        log.info("microphone: %s %d Hz × %d channels", props.subtype, self.rate, self.channels)
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
            frames = np.frombuffer(view, np.float32)[: buffer.length // 4].reshape(-1, self.channels)
            # With MediaCategory.SPEECH, Windows' voice processing puts its output on
            # one channel of the array and leaves the rest silent (channel 2 of 8 on a
            # Surface Pro), so averaging divided the voice by 8. Use the channel with
            # the most energy, which is the processed one.
            if len(frames):  # the first quanta can be empty
                self.energy = 0.95 * self.energy + 0.05 * np.mean(frames * frames, axis=0)
            mono = frames[:, int(np.argmax(self.energy))].copy()
            del frames  # it still exports the view, which must be released before it closes
        if len(mono):
            self.on_samples(mono)

    @property
    def is_open(self) -> bool:
        return self.graph is not None

    @property
    def channel(self) -> int:
        return int(np.argmax(self.energy))

    def stop(self) -> None:
        if self.graph is not None:
            self.graph.stop()
            self.graph.close()
            self.graph = None
