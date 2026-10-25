"""Spike: SPIKE Prime parts from the LDraw library, drawn offscreen with pygfx.

uv run python spikes/ldraw_preview.py LDRAW_DIR [UNOFFICIAL_DIR] --out out/ldraw.png
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rendercanvas.offscreen import RenderCanvas

from pybricks_voice.ldraw import LDU, Library
from pybricks_voice.scene import gfx, use_d3d12  # pygfx, once scene.py has found its FreeType DLL

PARTS = [  # file, color, where it goes (cm, scene frame)
    ("45601c01.dat", 15, (0, 0, 0)),
    ("54696p01.dat", 15, (9, 0, 0)),
    ("39367p01.dat", 15, (17, 0, 0)),
    ("37308.dat", 15, (-8, 0, 0)),
    ("37316.dat", 15, (-14, 0, 0)),
    ("32524.dat", 15, (0, 0, 10)),
]


def mesh(library: Library, name: str, color: int, matrix=None, at=(0, 0, 0)) -> gfx.Group:
    """A part as one mesh per color, in centimeters, y up; matrix and at
    place it first, in LDU."""
    group = gfx.Group()
    for code, tris in library.triangles(name, color).items():
        if matrix is not None:
            tris = tris @ np.array(matrix, float).reshape(3, 3).T + at
        points = tris.reshape(-1, 3) * (LDU * 100) * (-1, -1, 1)  # a half turn about z: y up, LDraw -z forward
        edges = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
        normals = np.cross(*edges) * (-1, -1, 1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        geometry = gfx.Geometry(
            positions=points.astype(np.float32),
            normals=np.repeat(normals, 3, axis=0).astype(np.float32),
            indices=np.arange(len(points), dtype=np.int32).reshape(-1, 3),
        )
        c = library.color(code)
        name_ = c.name.lower()
        material = gfx.MeshStandardMaterial(
            color=c.rgb,
            roughness=0.85 if "rubber" in name_ else 0.35,
            metalness=0.8 if "chrome" in name_ or "metal" in name_ else 0.0,
            side="both",
        )
        group.add(gfx.Mesh(geometry, material))
    return group


def write_png(path: pathlib.Path, rgba: np.ndarray) -> None:
    height, width, _ = rgba.shape
    raw = b"".join(b"\0" + rgba[y].tobytes() for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


WHEELS = [  # LDraw lines' matrix and position for each wheel, in LDU (see driving_base.ldr)
    ((0, 0, -1, 0, 1, 0, 1, 0, 0), (-140, -70, 0)),
    ((0, 0, 1, 0, 1, 0, -1, 0, 0), (140, -70, 0)),
]
VIEWS = {  # camera position and target, cm, scene frame
    "front-left": ((-20, 14, -22), (0, 4, 2)),
    "side": ((-34, 5, 3), (0, 4, 3)),
    "back-right": ((20, 16, 30), (0, 4, 3)),
    "top": ((0, 40, 3.01), (0, 0, 3)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=pathlib.Path)
    parser.add_argument("--model", type=pathlib.Path, help="a model to draw with wheels, from four sides")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out/ldraw.png"))
    options = parser.parse_args()
    use_d3d12()
    library = Library(*options.roots)

    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#1b1f27"), gfx.AmbientLight("#dde6ff", 0.6))
    sun = gfx.DirectionalLight("#ffffff", 2.4)
    sun.local.position = (30, 60, 40)
    scene.add(sun)
    start = time.perf_counter()
    if options.model:
        scene.add(mesh(library, str(options.model.resolve()), 16))
        for matrix, at in WHEELS:
            scene.add(mesh(library, "39367p01.dat", 0, matrix, at))
        grid = gfx.GridHelper(40, 20, "#56617a", "#3a4250")
        scene.add(grid)
        views = VIEWS
    else:
        for name, color, at in PARTS:
            part = mesh(library, name, color)
            part.local.position = at
            scene.add(part)
        views = {"parts": ((2, 22, 34), (2, 0, 2))}
    print(f"loaded in {time.perf_counter() - start:.2f} s")

    images = []
    for position, target in views.values():
        camera = gfx.PerspectiveCamera(40)
        camera.local.position = position
        camera.look_at(target)
        canvas = RenderCanvas(size=(700, 500) if options.model else (1400, 800), pixel_ratio=1)
        renderer = gfx.WgpuRenderer(canvas)
        canvas.request_draw(lambda r=renderer, c=camera: r.render(scene, c))
        images.append(np.asarray(canvas.draw()))
    image = np.vstack([np.hstack(images[:2]), np.hstack(images[2:])]) if len(images) == 4 else images[0]
    options.out.parent.mkdir(parents=True, exist_ok=True)
    write_png(options.out, np.ascontiguousarray(image))
    print(f"wrote {options.out}")


if __name__ == "__main__":
    main()
