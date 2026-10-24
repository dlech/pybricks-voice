"""Turns what Whisper heard into robot commands.

A small grammar rather than a language model: plain driving instructions are
a handful of shapes, and anything that doesn't fit is reported back instead of
guessed at.

    "forward 50 centimeters, turn right 90 degrees, then back 20"
    -> [Straight(500), Turn(90), Straight(-200)]
    "drive in a big circle to the left, then spin around twice"
    -> [Curve(600, -360), Turn(360), Turn(360)]

Distances without a unit are centimetres; turns without an angle are 90°.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .robot import Command, Curve, Drive, Home, SetSpeed, Stop, Straight, Turn

DEFAULT_DISTANCE_MM = 200.0
MIN_RADIUS_MM = 60.0  # about half the axle track: the tightest circle a driving base can follow
SHAPE_SIDE_MM = {"small": 150.0, "little": 150.0, "tiny": 100.0, "big": 500.0, "large": 500.0, "huge": 800.0}
DEFAULT_SHAPE_SIDE_MM = 300.0
_SHAPES = {"triangle": 3, "square": 4, "box": 4, "pentagon": 5, "hexagon": 6, "octagon": 8}
# "back to the starting point", "return to start", "go home", "back where you started";
# not "start driving forward".
_HOME_RE = re.compile(
    r"\b(?:back|return|returning|head|drive|go)\s+(?:to\s+)?(?:the\s+)?(?:start|starting point|beginning|origin)\b"
    r"|\b(?:go|going|head|drive|come)\s+(?:back\s+)?home\b|\bwhere (?:you|we|it) started\b"
)
# Words that can follow a distance without being a unit we don't know.
_AFTER_NUMBER = {
    "and",
    "then",
    "more",
    "please",
    "to",
    "at",
    "for",
    "now",
    "again",
    "forward",
    "forwards",
    "back",
    "backward",
    "backwards",
    "ahead",
    "straight",
    "the",
    "left",
    "right",
}
DEFAULT_ANGLE = 90.0
CIRCLE_RADIUS_MM = {
    "small": 150.0,
    "tight": 150.0,
    "little": 150.0,
    "big": 600.0,
    "large": 600.0,
    "wide": 600.0,
    "huge": 900.0,
}
DEFAULT_CIRCLE_RADIUS_MM = 300.0
CURVE_RADIUS_MM = 500.0
VEER_ANGLE = 30.0
MAX_REPEAT = 10

_CIRCLE = {"circle", "circles", "loop", "loops", "donut", "donuts", "doughnut", "doughnuts"}
_CURVE = {"curve", "arc", "veer", "swing", "bend", "swivel", "swerve"}
_IN_PLACE = {"spin", "turn", "rotate", "pivot"}
_TOO_FAR = {"mile", "miles", "km", "kilometer", "kilometers", "kilometre", "kilometres", "yard", "yards"}
# What Whisper writes for words it half-hears.
_SOUNDS_LIKE = {"vier": "veer", "steer": "veer", "spun": "spin", "twice": "2 times", "thrice": "3 times"}

EXAMPLES = [
    "forward 30 centimeters",
    "back up 10",
    "turn left",
    "turn right 45 degrees",
    "turn around",
    "spin around twice",
    "drive in a circle",
    "make a big circle to the left",
    "half a circle to the right",
    "do a figure eight",
    "drive in a square",
    "drive a small triangle",
    "go back to the start",
    "curve right",
    "veer left",
    "drive and turn left at the same time",
    "spin right for 2 seconds",
    "faster",
    "stop",
]
TIMED_SPEED = 150.0  # mm/s for "drive forward for 2 seconds"
TIMED_TURN_RATE = 90.0  # deg/s for "spin for 3 seconds"

_UNITS_MM = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "millimetre": 1.0,
    "millimetres": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "centimetre": 10.0,
    "centimetres": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "metre": 1000.0,
    "metres": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    "ft": 304.8,
    "foot": 304.8,
    "feet": 304.8,
}
_SECONDS = {"s", "sec", "secs", "second", "seconds"}
_DEGREES = {"degree", "degrees", "deg", "°"}

_DIRECTIONS = {"forward", "forwards", "ahead", "straight", "onward", "on"}
_FORWARD = _DIRECTIONS | {"go", "drive", "move"}
_BACKWARD = {"back", "backward", "backwards", "reverse"}
_STOP = {"stop", "halt", "freeze", "brake", "wait"}

_ONES = {
    "zero": 0,
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


@dataclass
class Parse:
    commands: list[Command]
    not_understood: list[str]
    remarks: list[str] = field(default_factory=list)  # what the robot says back to chatter


def _words_to_numbers(text: str) -> str:
    """ "one hundred twenty five" -> "125", "a half" -> "0.5", "2 0" stays."""
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w == "half" or (w in ("a", "one") and i + 1 < len(words) and words[i + 1] == "half"):
            out.append("0.5")
            i += 1 if w == "half" else 2
            continue
        if w == "quarter" or (w in ("a", "one") and i + 1 < len(words) and words[i + 1] == "quarter"):
            out.append("0.25")
            i += 1 if w == "quarter" else 2
            continue
        if (
            w in _TENS
            or (w in _ONES and w not in ("a", "an"))
            or (w in ("a", "an") and i + 1 < len(words) and words[i + 1] in ("hundred", "thousand"))
        ):
            total = current = 0
            while i < len(words) and (
                words[i] in _ONES or words[i] in _TENS or words[i] in ("hundred", "thousand", "and")
            ):
                x = words[i]
                if x == "and":
                    # "one hundred and twenty" continues a number; "and" alone ends it.
                    if (
                        not (current or total)
                        or i + 1 >= len(words)
                        or not (words[i + 1] in _ONES or words[i + 1] in _TENS)
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
    text = " ".join(_SOUNDS_LIKE.get(w, w) for w in text.replace(",", " , ").replace(";", " ; ").split())
    text = _words_to_numbers(text)
    # Driving while turning is one arc, not "drive" and then "turn", so say so
    # before the text is split at "and".
    text = re.sub(
        r"\b(?:drive|go|move)\s+(?:forward\s+)?(?:and|while)\s+turn(?:ing)?\s+(left|right)"
        r"(?:\s+at the same time)?",
        r"curve \1",
        text,
    )
    text = re.sub(
        r"\bturn(?:ing)?\s+(left|right)\s+(?:while|as you)\s+(?:drive|driving|go|going|move|moving)\b",
        r"curve \1",
        text,
    )
    text = re.sub(r"\bat the same time\b", "", text)
    return " ".join(text.split())


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
    # "spin around 3 times": do the rest of it that many times.
    times = re.search(r"\b(\d+) times\b", segment)
    if times:
        rest = (segment[: times.start()] + segment[times.end() :]).strip()
        parsed = parse_segment(rest) if rest else None
        return None if parsed is None else parsed * max(1, min(MAX_REPEAT, int(times.group(1))))

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
    sign = -1.0 if left else 1.0
    backward = bool(words & _BACKWARD)

    if _HOME_RE.search(segment):
        return [Home()]

    shape = next((n for w, n in _SHAPES.items() if w in words or w + "s" in words), None)
    if shape:
        side = next((s for w, s in SHAPE_SIDE_MM.items() if w in words), DEFAULT_SHAPE_SIDE_MM)
        if value is not None and unit in _UNITS_MM:
            side = value * _UNITS_MM[unit]
        return [Straight(side), Turn(sign * 360 / shape)] * shape

    radius = next((r for w, r in CIRCLE_RADIUS_MM.items() if w in words), DEFAULT_CIRCLE_RADIUS_MM)
    if value is not None and "radius" in words:
        radius = value * _UNITS_MM.get(unit or "", 10.0)
    elif value is not None and unit in _UNITS_MM:
        radius = value * _UNITS_MM[unit] / 2  # "a two centimeter circle": its size, so the diameter
    radius = max(radius, MIN_RADIUS_MM) * (-1 if backward else 1)
    if "figure" in words and ("8" in words or "eight" in words):
        # A circle one way, then the other.
        return [Curve(radius, sign * 360), Curve(radius, -sign * 360)]
    if words & _CIRCLE:
        if (words & _IN_PLACE and not words & {"drive", "go", "make", "do"}) or "in place" in segment:
            return [Turn(sign * 360)]
        # "half a circle", "a quarter circle": part of the way around.
        fraction = (
            value if value is not None and value <= 1 and unit not in _UNITS_MM and "radius" not in words else 1.0
        )
        return [Curve(radius, sign * 360 * fraction)]
    if words & _CURVE:
        angle = (
            value
            if value is not None and (unit in _DEGREES or unit is None)
            else (VEER_ANGLE if words & {"veer", "swerve"} else DEFAULT_ANGLE)
        )
        radius = (
            800.0
            if words & {"wide", "big", "gentle", "gently"}
            else (250.0 if words & {"tight", "sharp", "sharply"} else CURVE_RADIUS_MM)
        )
        return [Curve(-radius if backward else radius, sign * angle)]

    if "u turn" in segment or "about face" in segment:
        return [Turn(sign * 180)]
    if "around" in words:
        # "Turn around" is about-face; "spin around", "go around" (and Whisper's
        # "been around") are a full turn.
        half = "turn" in words and "all the way" not in segment
        return [Turn(sign * (180 if half else 360))]
    if left or right or words & {"turn", "rotate", "spin"}:
        if value is not None and unit in _SECONDS:
            return [Drive(0, sign * TIMED_TURN_RATE, value)]
        if value is not None and (unit in _DEGREES or unit is None):
            return [Turn(sign * value)]
        if "full" in words or "circle" in words:
            return [Turn(sign * 360)]
        if "half" in words:
            return [Turn(sign * 180)]
        return [Turn(sign * DEFAULT_ANGLE)]

    if backward or words & _FORWARD:
        sign = -1.0 if backward else 1.0  # left/right don't apply to straight lines
        if value is not None and unit in _SECONDS:
            return [Drive(sign * TIMED_SPEED, 0, value)]
        if unit in _TOO_FAR:
            return None  # "three miles": not on this table
        if unit is not None and unit not in _UNITS_MM and unit not in _AFTER_NUMBER:
            return None  # "two units": a unit we don't know, so not 2 cm
        if value is not None:
            scale = _UNITS_MM.get(unit or "", 10.0)  # no unit: centimetres
            return [Straight(sign * value * scale)]
        if backward or words & _DIRECTIONS:
            return [Straight(sign * DEFAULT_DISTANCE_MM)]
        return None  # a bare "go" or "drive", as in "there we go", isn't a command
    return None


def describe(commands: list[Command]) -> str:
    """Commands in plain words, for the robot's side of the conversation."""

    def side(x: float) -> str:
        return "left" if x < 0 else "right"

    def one(c: Command) -> str:
        match c:
            case Straight(d):
                return f"{'forward' if d >= 0 else 'back'} {abs(d) / 10:g} cm"
            case Turn(a) if abs(a) == 360:
                return f"spin {side(a)} all the way around"
            case Turn(a):
                return f"turn {side(a)} {abs(a):g}°"
            case Curve(r, a) if abs(a) == 360:
                return f"drive {'backwards ' if r < 0 else ''}a circle to the {side(a)} ({abs(r) / 10:g} cm radius)"
            case Curve(r, a) if abs(a) == 180:
                return (
                    f"drive {'backwards ' if r < 0 else ''}half a circle to the {side(a)} ({abs(r) / 10:g} cm radius)"
                )
            case Curve(r, a):
                return f"curve {side(a)}{' backwards' if r < 0 else ''} {abs(a):g}° ({abs(r) / 10:g} cm radius)"
            case Home():
                return "go back to the start"
            case Drive(v, w, t) if v == 0:
                return f"spin {side(w)} for {t:g} s"
            case Drive(v, w, t):
                return f"drive {'forward' if v >= 0 else 'back'} for {t:g} s"
            case Stop():
                return "stop"
            case SetSpeed(f):
                return "go faster" if f > 1 else "slow down"
        return str(c)

    def say(commands: list[Command]) -> str:
        # A pattern repeated, as a square is: "4 × (forward 30 cm, then turn right 90°)".
        for period in range(2, len(commands) // 2 + 1):
            if len(commands) % period == 0 and commands == commands[:period] * (len(commands) // period):
                return f"{len(commands) // period} × ({say(commands[:period])})"
        # "spin left all the way around ×3" reads better than three of them.
        parts: list[tuple[str, int]] = []
        for c in commands:
            text = one(c)
            if parts and parts[-1][0] == text:
                parts[-1] = (text, parts[-1][1] + 1)
            else:
                parts.append((text, 1))
        return ", then ".join(t if n == 1 else f"{t} ×{n}" for t, n in parts)

    sentence = say(commands)
    return sentence[:1].upper() + sentence[1:]


def suggest(segment: str, count: int = 2) -> list[str]:
    """Example commands that share the most words with something not understood."""
    words = set(segment.split())
    scored = sorted(EXAMPLES, key=lambda e: -len(words & set(e.split())))
    if not words & set(" ".join(scored[:count]).split()):
        return ["forward 30 centimeters", "drive in a circle"][:count]
    return scored[:count]


_PRAISE = {
    "good",
    "great",
    "nice",
    "awesome",
    "cool",
    "amazing",
    "brilliant",
    "clever",
    "smart",
    "perfect",
    "excellent",
    "wonderful",
    "yay",
    "bravo",
}
_CHATTER = _PRAISE | {
    "no",
    "yes",
    "yeah",
    "ok",
    "okay",
    "please",
    "like",
    "um",
    "uh",
    "hey",
    "now",
    "so",
    "well",
    "robot",
    "oh",
    "and",
    "thanks",
    "thank",
    "you",
    "again",
    "my",
    "wow",
    "whoa",
    "oops",
    "huh",
    "hmm",
    "such",
    "a",
    "re",
    "are",
    "very",
    "job",
    "done",
    "that",
    "was",
    "s",
    "it",
    "is",
    "what",
    "haha",
    "ha",
    "lol",
    "gosh",
    "dear",
}


def parse(text: str) -> Parse:
    commands: list[Command] = []
    not_understood: list[str] = []
    remarks: list[str] = []
    for segment in _split(normalize(text)):
        if set(segment.split()) <= _CHATTER:
            # "No, like…", "Oh my" and "please" on their own aren't commands;
            # "you're such a good robot" deserves an answer.
            if set(segment.split()) & _PRAISE and not remarks:
                remarks.append("Thanks! 😊")
            continue
        parsed = parse_segment(segment)
        if parsed is None:
            not_understood.append(segment)
        else:
            commands.extend(parsed)
    return Parse(commands, not_understood, remarks)
