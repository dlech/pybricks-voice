"""
Prints the level of each microphone channel for a few seconds.

Usage: uv run tools/mic_channels.py [seconds]
"""

import asyncio
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from winrt.windows.media import AudioBufferAccessMode
from winrt.windows.media.audio import AudioGraph, AudioGraphSettings
from winrt.windows.media.capture import MediaCategory
from winrt.windows.media.render import AudioRenderCategory


async def capture(category: MediaCategory, seconds: float) -> tuple[np.ndarray, int]:
    """seconds of every channel of the default microphone, and the sample rate."""
    graph = (await AudioGraph.create_async(AudioGraphSettings(AudioRenderCategory.SPEECH))).graph
    node = (await graph.create_device_input_node_async(category)).device_input_node
    output = graph.create_frame_output_node()
    node.add_outgoing_connection(output)
    channels = output.encoding_properties.channel_count
    rate = output.encoding_properties.sample_rate
    chunks: list[np.ndarray] = []

    def quantum(g, _):
        frame = output.get_frame()
        with (
            frame.lock_buffer(AudioBufferAccessMode.READ) as buffer,
            buffer.create_reference() as ref,
            memoryview(ref) as view,
        ):
            chunks.append(np.frombuffer(view, np.float32)[: buffer.length // 4].reshape(-1, channels).copy())

    graph.add_quantum_started(quantum)
    graph.start()
    await asyncio.sleep(seconds)
    graph.stop()
    graph.close()
    return np.concatenate(chunks), rate


async def main(seconds: float) -> None:
    for category in (MediaCategory.SPEECH, MediaCategory.OTHER):
        audio, rate = await capture(category, seconds)
        channels = audio.shape[1]
        rms = 20 * np.log10(np.sqrt(np.mean(audio**2, axis=0)) + 1e-12)
        peak = 20 * np.log10(np.abs(audio).max(axis=0) + 1e-12)
        print(f"{category.name}: {channels} channels, {len(audio) / rate:.1f} s")
        for c in range(channels):
            print(f"  channel {c}: rms {rms[c]:7.1f} dBFS  peak {peak[c]:7.1f} dBFS")
        corr = np.corrcoef(audio.T) if channels > 1 else np.ones((1, 1))
        print(f"  correlation of channel 0 with the others: {np.round(corr[0], 2).tolist()}")


asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 5))
