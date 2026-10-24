# pybricks-voice

Drive a LEGO robot by talking to it. Whisper runs on the laptop's NPU, and for
now the robot is a 3D simulation of a SPIKE Prime driving base, so the voice
side can be built and tested without Bluetooth or a real table.

```
"forward 50 centimeters, turn right 90 degrees, then back 20"
  → Whisper on the Hexagon NPU (~150–250 ms)
  → straight(500 mm), turn(90°), straight(-200 mm)
  → the robot drives it
```

Built with [PyWinRT](https://github.com/pywinrt/pywinrt) 4: WinUI 3 for the
window, WebView2 + three.js for the 3D view, AudioGraph for the microphone,
and ONNX Runtime's QNN provider for the NPU.

## Requirements

- A Snapdragon X Elite PC (the Whisper model is precompiled for its NPU)
- [uv](https://docs.astral.sh/uv/) and an ARM64 CPython 3.14 (pinned in `.python-version`)

## Setup

```
uv sync
uv run tools/get_models.py
```

## Run

```
uv run python -m pybricks_voice
```

Say things like "go forward 30 centimeters", "turn left", "turn around",
"back up 10 inches", "spin right for 2 seconds", "faster", "stop", and chain
them with "then". You can also type commands into the box.

## Check

```
uv run python -m unittest          # robot, grammar, segmenter, vocabulary
uv run tools/check_whisper.py      # Windows speech synthesis -> Whisper on the NPU -> commands
```

## Layout

| | |
|---|---|
| `pybricks_voice/robot.py` | Pybricks `DriveBase`-style commands and the simulated robot |
| `pybricks_voice/grammar.py` | what Whisper heard → commands |
| `pybricks_voice/whisper.py` | Whisper-Small on the NPU, with a pure-Python token decoder |
| `pybricks_voice/speech.py` | microphone, voice-activity detection, the Whisper worker thread |
| `pybricks_voice/app.py` | the WinUI 3 window |
| `pybricks_voice/web/scene.html` | the three.js scene (three.js 0.169, MIT, vendored) |
