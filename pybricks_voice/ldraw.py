"""
LEGO models from the LDraw parts library, as triangles for the 3D view.

An LDraw file is lines of text. Type 1 places another file (a part, or a
primitive such as a stud or a cylinder) with a color and a 3x4 transform;
types 3 and 4 are triangles and quads; type 0 is a comment or a directive.
Edges (types 2 and 5) aren't drawn. Color 16 means "the color of whatever
placed me", and LDConfig.ldr names the rest.

Units are LDU, 0.4 mm, with y pointing down.

Triangles come out wound counterclockwise seen from outside, following the
BFC directives of the files that declare them (nearly all the official
parts): which way a file winds, INVERTNEXT, and placements that mirror. So
cross products give outward normals, which normals() smooths across the
shallow edges of round parts such as tires.

Each file is flattened once, into triangles per color in its own frame, and
then only transformed wherever it is placed; a driving base places the same
pins and studs hundreds of times.
"""

from __future__ import annotations

import functools
import math
import pathlib
import re
from dataclasses import dataclass

import numpy as np

MAIN_COLOR = 16
LDU = 0.0004  # meters

_COLOR = re.compile(r"^0\s+!COLOUR\s+(\S+)\s+CODE\s+(\d+)\s+VALUE\s+#([0-9A-Fa-f]{6})(.*)$")


@dataclass(frozen=True)
class Color:
    name: str
    rgb: str  # "#rrggbb"
    alpha: float = 1.0


class Library:
    """
    An unpacked LDraw library: the official one, and any others searched
    after it, such as the unofficial parts.
    """

    def __init__(self, *roots: pathlib.Path) -> None:
        self.folders = [root / sub for root in roots for sub in ("parts", "p", "models", "")]
        self.colors = _colors(roots[0] / "LDConfig.ldr")
        self._flat = functools.cache(self._flatten)

    def find(self, name: str) -> pathlib.Path:
        if (path := pathlib.Path(name)).is_absolute():  # a model of our own
            return path
        relative = name.replace("\\", "/").lower()
        for folder in self.folders:
            path = folder / relative
            if path.is_file():
                return path
        raise FileNotFoundError(f"{name} is not in the LDraw library")

    def color(self, code: int) -> Color:
        if code in self.colors:
            return self.colors[code]
        if 0x2000000 <= code <= 0x2FFFFFF:  # a direct color, 0x2RRGGBB
            return Color(f"#{code & 0xFFFFFF:06x}", f"#{code & 0xFFFFFF:06x}")
        raise KeyError(f"LDraw color {code} isn't in LDConfig.ldr")

    def triangles(self, name: str, color: int = MAIN_COLOR) -> dict[int, np.ndarray]:
        """
        The triangles of a part or model, n x 3 x 3 in LDU, by color.
        Whatever it leaves as color 16 gets `color`.
        """
        flat = self._flat(name if pathlib.Path(name).is_absolute() else name.lower())
        if color == MAIN_COLOR or MAIN_COLOR not in flat:
            return flat
        out = dict(flat)
        main = out.pop(MAIN_COLOR)
        out[color] = np.concatenate([out[color], main]) if color in out else main
        return out

    def _flatten(self, name: str) -> dict[int, np.ndarray]:
        pieces: dict[int, list[np.ndarray]] = {}
        clockwise = invert_next = False
        for line in self.find(name).read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if not fields:
                continue
            kind = fields[0]
            if kind == "0" and fields[1:2] == ["BFC"]:
                words = set(fields[2:])
                if "CW" in words or "CCW" in words:
                    clockwise = "CW" in words
                invert_next = invert_next or "INVERTNEXT" in words
            elif kind == "1" and len(fields) >= 15:
                color = int(fields[1], 0)
                x, y, z, a, b, c, d, e, f, g, h, i = map(float, fields[2:14])
                matrix = np.array([[a, b, c], [d, e, f], [g, h, i]])
                # A mirroring placement turns the winding over, as INVERTNEXT does.
                flip = invert_next != (np.linalg.det(matrix) < 0)
                invert_next = False
                for code, tris in self.triangles(" ".join(fields[14:]), color).items():
                    placed = tris @ matrix.T + (x, y, z)
                    pieces.setdefault(code, []).append(placed[:, ::-1] if flip else placed)
            elif kind in ("3", "4"):
                n = int(kind)
                if len(fields) < 2 + 3 * n:
                    continue
                points = np.array(fields[2 : 2 + 3 * n], float).reshape(n, 3)
                if clockwise:
                    points = points[::-1]
                tris = points[None] if n == 3 else points[[[0, 1, 2], [0, 2, 3]]]
                pieces.setdefault(int(fields[1], 0), []).append(tris)
        return {code: np.concatenate(parts) for code, parts in pieces.items()}


def normals(tris: np.ndarray, crease: float = math.radians(40)) -> np.ndarray:
    """
    Vertex normals for n x 3 x 3 counterclockwise triangles: averaged with the
    faces around each corner, unless that bends it more than `crease` from its
    own face's, so curves look round and edges stay sharp.
    """
    faces = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])  # length: twice the area
    own = faces / np.maximum(np.linalg.norm(faces, axis=1, keepdims=True), 1e-12)
    # Corners at the same place, to a hundredth of an LDU, share a normal.
    _, corner = np.unique(np.round(tris.reshape(-1, 3), 2), axis=0, return_inverse=True)
    shared = np.zeros((corner.max() + 1, 3))
    np.add.at(shared, corner, np.repeat(faces, 3, axis=0))
    smooth = shared[corner]
    smooth /= np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-12)
    flat = np.repeat(own, 3, axis=0)
    bent = (smooth * flat).sum(axis=1, keepdims=True) < math.cos(crease)
    return np.where(bent, flat, smooth).reshape(-1, 3, 3)


def _colors(path: pathlib.Path) -> dict[int, Color]:
    colors = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := _COLOR.match(line.strip()):
            name, code, rgb, rest = m.groups()
            alpha = re.search(r"ALPHA\s+(\d+)", rest)
            colors[int(code)] = Color(name, f"#{rgb.lower()}", int(alpha.group(1)) / 255 if alpha else 1.0)
    return colors
