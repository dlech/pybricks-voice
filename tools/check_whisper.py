"""End-to-end check on the NPU: Windows speech synthesis -> Whisper -> grammar.

Synthesizes each phrase with Windows' own voice through PyWinRT, transcribes it
on the NPU and parses it, so the whole pipeline is tested without a microphone.

Usage: uv run tools/check_whisper.py
"""

import asyncio
import io
import pathlib
import sys
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from winrt.windows.media.speechsynthesis import SpeechSynthesizer
from winrt.windows.storage.streams import DataReader

from pybricks_voice.grammar import parse
from pybricks_voice.whisper import SAMPLE_RATE, Whisper

PHRASES = [
    "Forward 50 centimeters, turn right 90 degrees, then back 20.",
    "Turn left.",
    "Drive forward one meter and then turn around.",
    "Spin left for three seconds.",
    "Stop!",
]


async def synthesize(text: str, voice=None) -> np.ndarray:
    synthesizer = SpeechSynthesizer()
    if voice is not None:
        synthesizer.voice = voice
    stream = await synthesizer.synthesize_text_to_stream_async(text)
    reader = DataReader(stream.get_input_stream_at(0))
    await reader.load_async(stream.size)
    data = bytearray(stream.size)
    reader.read_bytes(data)
    with wave.open(io.BytesIO(bytes(data))) as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), np.int16).astype(np.float32) / 32768
    pcm = pcm.reshape(-1, channels).mean(axis=1)
    t = np.arange(0, len(pcm) / rate, 1 / SAMPLE_RATE)
    return np.interp(t, np.arange(len(pcm)) / rate, pcm).astype(np.float32)


async def main() -> None:
    whisper = Whisper()
    print(f"Whisper loaded in {whisper.load_s:.1f} s")
    for phrase in PHRASES:
        audio = await synthesize(phrase)
        text, timing = whisper.transcribe(audio)
        result = parse(text)
        print(f"\nsaid:  {phrase}  ({len(audio) / SAMPLE_RATE:.1f} s)")
        print(
            f"heard: {text}  ({timing.total_ms:.0f} ms: mel {timing.mel_ms:.0f}, "
            f"encoder {timing.encoder_ms:.0f}, decoder {timing.decoder_ms:.0f} / {timing.steps} steps)"
        )
        print(
            f"runs:  {', '.join(map(str, result.commands)) or '-'}"
            + (f"   not understood: {result.not_understood}" if result.not_understood else "")
        )


if __name__ == "__main__":
    asyncio.run(main())
