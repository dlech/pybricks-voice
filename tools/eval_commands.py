"""Scores the speech pipeline on 20 spoken commands (milestone 1: at least 18).

Each phrase is spoken by Windows' speech synthesis, cycling through every
installed voice, and saved to out/eval/ so later runs (and other machines)
hear exactly the same audio. Whisper transcribes each on the NPU and the
grammar parses it; a phrase passes when the commands match exactly.

Usage: uv run tools/eval_commands.py [--resynthesize]
"""

import asyncio
import pathlib
import sys
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from check_whisper import synthesize
from winrt.windows.media.speechsynthesis import SpeechSynthesizer

from pybricks_voice.grammar import parse
from pybricks_voice.robot import Drive, SetSpeed, Stop, Straight, Turn
from pybricks_voice.whisper import SAMPLE_RATE, Whisper

OUT = pathlib.Path(__file__).resolve().parent.parent / "out" / "eval"

CASES = [
    ("Forward 50 centimeters, turn right 90 degrees, then back 20.", [Straight(500), Turn(90), Straight(-200)]),
    ("Go forward 30 centimeters.", [Straight(300)]),
    ("Turn left.", [Turn(-90)]),
    ("Turn right 45 degrees.", [Turn(45)]),
    ("Back up 10 centimeters.", [Straight(-100)]),
    ("Drive forward one meter.", [Straight(1000)]),
    ("Turn around.", [Turn(180)]),
    ("Stop!", [Stop()]),
    ("Spin left for three seconds.", [Drive(0, -90, 3)]),
    ("Drive forward for two seconds.", [Drive(150, 0, 2)]),
    ("Faster.", [SetSpeed(1.5)]),
    ("Slow down.", [SetSpeed(1 / 1.5)]),
    ("Move ahead 150 millimeters.", [Straight(150)]),
    ("Reverse half a meter.", [Straight(-500)]),
    ("Turn left 180 degrees.", [Turn(-180)]),
    ("Forward 1 meter and then turn left and then forward 50.", [Straight(1000), Turn(-90), Straight(500)]),
    ("Go forward 20 centimeters, then turn right.", [Straight(200), Turn(90)]),
    ("Rotate right thirty degrees.", [Turn(30)]),
    ("Back up 10 inches.", [Straight(-254)]),
    ("Turn right, forward 10 centimeters, turn left.", [Turn(90), Straight(100), Turn(-90)]),
]


def read_wav(path: pathlib.Path) -> np.ndarray:
    with wave.open(str(path)) as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), np.int16).astype(np.float32) / 32768


def write_wav(path: pathlib.Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


async def make_audio(resynthesize: bool) -> list[tuple[pathlib.Path, str]]:
    OUT.mkdir(parents=True, exist_ok=True)
    voices = list(SpeechSynthesizer.all_voices)
    files = []
    for i, (phrase, _) in enumerate(CASES):
        voice = voices[i % len(voices)]
        path = OUT / f"{i:02d}.wav"
        if resynthesize or not path.exists():
            write_wav(path, await synthesize(phrase, voice))
        files.append((path, voice.display_name))
    return files


async def main() -> None:
    files = await make_audio("--resynthesize" in sys.argv)
    whisper = Whisper()
    passed, times = 0, []
    for (phrase, expected), (path, voice) in zip(CASES, files, strict=True):
        text, timing = whisper.transcribe(read_wav(path))
        got = parse(text).commands
        ok = got == expected
        passed += ok
        times.append(timing.total_ms)
        print(f"{'PASS' if ok else 'FAIL'} {timing.total_ms:4.0f} ms  [{voice}] {phrase}")
        if not ok:
            print(f"     heard {text!r} -> {[str(c) for c in got]}, wanted {[str(c) for c in expected]}")
    print(
        f"\n{passed}/{len(CASES)} passed (milestone: 18); Whisper median {np.median(times):.0f} ms, "
        f"max {max(times):.0f} ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
