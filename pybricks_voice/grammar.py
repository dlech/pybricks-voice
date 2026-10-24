"""Turns what Whisper heard into robot commands.

A small grammar rather than a language model: plain driving instructions are
a handful of shapes, and anything that doesn't fit is reported back instead of
guessed at.

    "forward 50 centimeters, turn right 90 degrees, then back 20"
    -> [Straight(500), Turn(90), Straight(-200)]

Distances without a unit are centimetres; turns without an angle are 90°.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .robot import Command, Drive, SetSpeed, Stop, Straight, Turn

DEFAULT_DISTANCE_MM = 200.0
DEFAULT_ANGLE = 90.0
TIMED_SPEED = 150.0  # mm/s for "drive forward for 2 seconds"
TIMED_TURN_RATE = 90.0  # deg/s for "spin for 3 seconds"

_UNITS_MM = {
    "mm": 1.0, "millimeter": 1.0, "millimeters": 1.0, "millimetre": 1.0, "millimetres": 1.0,
    "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0, "centimetre": 10.0, "centimetres": 10.0,
    "m": 1000.0, "meter": 1000.0, "meters": 1000.0, "metre": 1000.0, "metres": 1000.0,
    "in": 25.4, "inch": 25.4, "inches": 25.4, "ft": 304.8, "foot": 304.8, "feet": 304.8,
}
_SECONDS = {"s", "sec", "secs", "second", "seconds"}
_DEGREES = {"degree", "degrees", "deg", "°"}

_FORWARD = {"forward", "forwards", "ahead", "straight", "go", "drive", "move"}
_BACKWARD = {"back", "backward", "backwards", "reverse"}
_STOP = {"stop", "halt", "freeze", "brake", "wait"}

_ONES = {
    "zero": 0, "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}


@dataclass
class Parse:
    commands: list[Command]
    not_understood: list[str]


def _words_to_numbers(text: str) -> str:
    """"one hundred twenty five" -> "125", "a half" -> "0.5", "2 0" stays."""
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w == "half" or (w in ("a", "one") and i + 1 < len(words) and words[i + 1] == "half"):
            out.append("0.5")
            i += 1 if w == "half" else 2
            continue
        if w in _TENS or (w in _ONES and w not in ("a", "an")) or (
            w in ("a", "an") and i + 1 < len(words) and words[i + 1] in ("hundred", "thousand")
        ):
            total = current = 0
            while i < len(words) and (words[i] in _ONES or words[i] in _TENS
                                      or words[i] in ("hundred", "thousand", "and")):
                x = words[i]
                if x == "and":
                    # "one hundred and twenty" continues a number; "and" alone ends it.
                    if not (current or total) or i + 1 >= len(words) or not (
                        words[i + 1] in _ONES or words[i + 1] in _TENS
                    ):
                        break
                elif x in _ONES:
                    current += _ONES[x]
                elif x in _TENS:
                    current += _TENS[x]
                elif x == "hundred":
                    current = max(current, 1) * 100
                elif x == "thousand":
                    total += max(current, 1) * 1000
                    current = 0
                i += 1
            out.append(str(total + current))
            continue
        out.append(w)
        i += 1
    return " ".join(out)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)  # 1,000
    text = re.sub(r"(\d)\s*°", r"\1 degrees", text)
    text = re.sub(r"(\d)(cm|mm|m|in|ft|s)\b", r"\1 \2", text)  # 20cm
    text = re.sub(r"[^a-z0-9.\s,;]", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " . ", text)  # sentence dots, not 1.5
    text = _words_to_numbers(" ".join(text.replace(",", " , ").replace(";", " ; ").split()))
    return text


def _split(text: str) -> list[str]:
    parts = re.split(r"\s*(?:,|;|\s\.\s|\bthen\b|\band then\b|\bafter that\b|\band\b)\s*", f" {text} ")
    return [p.strip(" .") for p in parts if p.strip(" .")]


def _number(tokens: list[str]) -> tuple[float | None, str | None]:
    """The first number in tokens and the unit word right after it."""
    for i, t in enumerate(tokens):
        if re.fullmatch(r"\d+(\.\d+)?", t):
            rest = [w for w in tokens[i + 1 :] if w not in ("a", "an", "of")]  # "half a meter"
            return float(t), rest[0] if rest else None
    return None, None


def parse_segment(segment: str) -> list[Command] | None:
    tokens = segment.split()
    words = set(tokens)
    value, unit = _number(tokens)

    if words & _STOP:
        return [Stop()]
    if words & {"faster", "quicker"} or "speed up" in segment:
        return [SetSpeed(1.5)]
    if words & {"slower"} or "slow down" in segment:
        return [SetSpeed(1 / 1.5)]

    left, right = "left" in words, "right" in words
    if "around" in words or "u turn" in segment or "about face" in segment:
        return [Turn(-180 if left else 180)]
    if left or right or words & {"turn", "rotate", "spin"}:
        sign = -1.0 if left else 1.0
        if value is not None and unit in _SECONDS:
            return [Drive(0, sign * TIMED_TURN_RATE, value)]
        if value is not None and (unit in _DEGREES or unit is None):
            return [Turn(sign * value)]
        if "full" in words or "circle" in words:
            return [Turn(sign * 360)]
        if "half" in words:
            return [Turn(sign * 180)]
        return [Turn(sign * DEFAULT_ANGLE)]

    backward = bool(words & _BACKWARD)
    if backward or words & _FORWARD:
        sign = -1.0 if backward else 1.0
        if value is not None and unit in _SECONDS:
            return [Drive(sign * TIMED_SPEED, 0, value)]
        if value is not None:
            scale = _UNITS_MM.get(unit or "", 10.0)  # no unit: centimetres
            return [Straight(sign * value * scale)]
        return [Straight(sign * DEFAULT_DISTANCE_MM)]
    return None


def parse(text: str) -> Parse:
    commands: list[Command] = []
    not_understood: list[str] = []
    for segment in _split(normalize(text)):
        parsed = parse_segment(segment)
        if parsed is None:
            not_understood.append(segment)
        else:
            commands.extend(parsed)
    return Parse(commands, not_understood)
