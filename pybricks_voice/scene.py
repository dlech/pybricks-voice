"""
The 3D view's scene, drawn with pygfx: the table, the things on it, and the
robot with its trail, seen by a camera that follows it or looks down from
above, and the thread that steps and draws it. Getting it onto the screen
is each platform's: see windows/scene_view.py and macos/scene_view.py.

One unit is 1 cm, y up, forward is -z, as in three.js.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import pathlib
import threading
import time
from collections.abc import Callable

import numpy as np

# freetype-py, which pygfx imports, has no FreeType for Windows ARM64 of its
# own; tools/get_freetype.py builds one into vendor/, where it looks on PATH.
_VENDOR = pathlib.Path(__file__).resolve().parent.parent / "vendor"
if (_VENDOR / "freetype.dll").exists():
    os.environ["PATH"] = f"{_VENDOR}{os.pathsep}{os.environ.get('PATH', '')}"
try:
    import pygfx as gfx
except RuntimeError as e:
    if "Freetype" not in str(e):
        raise
    raise RuntimeError("FreeType is missing: run `uv run tools/get_freetype.py`") from e


import pylinalg as la  # noqa: E402
from rendercanvas.base import BaseCanvasGroup, BaseRenderCanvas  # noqa: E402

from . import ldraw  # noqa: E402
from .physics import THINGS, World  # noqa: E402

log = logging.getLogger(__name__)

# ----- the scene -------------------------------------------------------------------


_MODELS = pathlib.Path(__file__).resolve().parent / "models"
# From LDraw (LDU, y down, and our model faces -z) to the scene (cm, y up, the
# robot facing -z): a half turn about z.
_LDRAW_TO_SCENE = np.array([-1, -1, 1]) * ldraw.LDU * 100
# The wheels' LDraw placements, matrix and position, left then right: the
# robot's left is LDraw's +x. See driving_base.ldr.
_WHEELS = [
    ((0, 0, 1, 0, 1, 0, -1, 0, 0), (140, -70, 0)),
    ((0, 0, -1, 0, 1, 0, 1, 0, 0), (-140, -70, 0)),
]


def _ldraw_group(library: ldraw.Library, name: str, color: int = ldraw.MAIN_COLOR, matrix=None) -> gfx.Group:
    """
    An LDraw model or part as one mesh per color, in the scene's frame,
    turned by matrix (an LDraw line's nine numbers) first.
    """
    group = gfx.Group()
    for code, tris in library.triangles(name, color).items():
        if matrix is not None:
            tris = tris @ np.reshape(matrix, (3, 3)).T
        points = tris * _LDRAW_TO_SCENE  # a rotation: the winding stays counterclockwise
        geometry = gfx.Geometry(
            positions=points.reshape(-1, 3).astype(np.float32),
            normals=ldraw.normals(points).reshape(-1, 3).astype(np.float32),
            indices=np.arange(3 * len(points), dtype=np.int32).reshape(-1, 3),
        )
        c = library.color(code)
        rubber, metal = "rubber" in c.name.lower(), "chrome" in c.name.lower() or "metal" in c.name.lower()
        material = gfx.MeshStandardMaterial(
            color=c.rgb, roughness=0.85 if rubber else 0.35, metalness=0.8 if metal else 0.0, side="both"
        )
        mesh = gfx.Mesh(geometry, material)
        mesh.cast_shadow = mesh.receive_shadow = True
        group.add(mesh)
    return group


def _material(color: str, roughness: float = 0.5) -> gfx.MeshStandardMaterial:
    return gfx.MeshStandardMaterial(color=color, roughness=roughness)


def _about(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    return la.quat_from_axis_angle(axis, angle)


# From the physics' frame (x right, y forward, z up, in meters) to the scene's
# (x right, y up, forward -z, in centimeters): a quarter turn about x.
_TO_SCENE = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def _place(obj: gfx.WorldObject, position: np.ndarray, rotation: np.ndarray) -> None:
    obj.local.position = _TO_SCENE @ position * 100
    obj.local.rotation = la.quat_from_mat(_TO_SCENE @ rotation @ _TO_SCENE.T)


def _sphere(radius: float) -> gfx.Geometry:
    return gfx.sphere_geometry(radius, 24, 16)  # pyright: ignore[reportArgumentType]  (typed int, takes float)


def _box(width: float, height: float, depth: float) -> gfx.Geometry:
    return gfx.box_geometry(width, height, depth)  # pyright: ignore[reportArgumentType]  (typed int, takes float)


class RobotScene:
    """
    The table, a SPIKE-ish driving base, its trail, and a camera that
    follows it from behind or looks down from above.
    """

    MAX_TRAIL = 50_000

    def __init__(self) -> None:
        self.scene = gfx.Scene()
        self.scene.add(gfx.Background.from_color("#1b1f27"))
        self.scene.add(gfx.AmbientLight("#dde6ff", 0.5))
        self.sun = gfx.DirectionalLight("#ffffff", 2.4, cast_shadow=True)  # pyright: ignore[reportArgumentType]
        assert self.sun.shadow is not None
        self.sun.shadow.camera.width = self.sun.shadow.camera.height = 240
        # LDraw's triangles face either way (ldraw.py ignores its winding
        # directives), so no culling and a little bias, or they shadow themselves.
        self.sun.shadow.cull_mode = "none"
        self.sun.shadow.bias = 0.002
        self.scene.add(self.sun)

        # The table: a big floor with a 10 cm grid and a 1 m grid on top.
        floor = gfx.Mesh(gfx.plane_geometry(2000, 2000), _material("#2a303b", 0.95))
        floor.local.rotation = _about((1, 0, 0), -math.pi / 2)
        floor.receive_shadow = True
        fine = gfx.GridHelper(1000, 100, "#3a4250", "#343b47")
        coarse = gfx.GridHelper(1000, 10, "#56617a", "#56617a")
        fine.local.y, coarse.local.y = 0.02, 0.03
        self.scene.add(floor, fine, coarse)

        # A SPIKE Prime driving base, from LDraw's parts (see models/). The
        # axle midpoint is its reference point, as it is for Pybricks' DriveBase.
        library = ldraw.Library(_MODELS / "ldraw")
        self.robot = _ldraw_group(library, str(_MODELS / "driving_base.ldr"))
        self.wheels = []
        for matrix, at in _WHEELS:  # left, right
            wheel = gfx.Group()
            wheel.add(_ldraw_group(library, "39367p01.dat", 0, matrix))
            wheel.local.position = np.array(at) * _LDRAW_TO_SCENE
            self.robot.add(wheel)
            self.wheels.append(wheel)
        self.scene.add(self.robot)

        # Things on the table to push around, as physics.py defines them.
        self.things = {}
        for thing in THINGS:
            if thing.shape == "box":
                hx, hy, hz = (2 * v * 100 for v in thing.size)
                geometry = _box(hx, hz, hy)  # the physics' y is the scene's depth
            else:
                geometry = _sphere(thing.size[0] * 100)
            mesh = self._part(geometry, _material(thing.color, 0.6), (0, 0, 0))
            self.scene.add(mesh)
            self.things[thing.name] = mesh

        # Where the robot has been; NaN points aren't drawn.
        self.trail_points = np.full((self.MAX_TRAIL, 3), np.nan, np.float32)
        self.trail_count = 0
        self.trail_buffer = gfx.Buffer(self.trail_points)
        self.trail = gfx.Line(gfx.Geometry(positions=self.trail_buffer), gfx.LineMaterial(thickness=2, color="#4f8cff"))
        self.scene.add(self.trail)

        self.camera = gfx.PerspectiveCamera(50, depth_range=(1, 2000))
        self.camera.local.position = (0, 45, 60)
        self.chase = True
        self.zoom = 1.0

    @staticmethod
    def _part(geometry, material, at) -> gfx.Mesh:
        mesh = gfx.Mesh(geometry, material)
        mesh.local.position = at
        mesh.cast_shadow = mesh.receive_shadow = True
        return mesh

    def reset(self) -> None:
        self.trail_points[: self.trail_count] = np.nan
        self.trail_buffer.update_range(0, self.trail_count)
        self.trail_count = 0

    def looks(self) -> np.ndarray:
        """
        Everything in view that moves, in one array: while it doesn't change,
        nor does the picture.
        """
        parts = [self.robot.local.position, self.robot.local.rotation]
        parts += [wheel.local.rotation for wheel in self.wheels]
        for mesh in self.things.values():
            parts += [mesh.local.position, mesh.local.rotation]
        parts += [self.camera.local.position, (self.trail_count, self.zoom, self.chase)]
        return np.concatenate([np.ravel(part) for part in parts]).astype(float)

    def pose(self, world: World) -> None:
        robot = world.robot
        _place(self.robot, robot.position, robot.rotation)
        x, _, z = self.robot.local.position
        turn = -math.radians(world.heading)
        self.wheels[0].local.rotation = _about((1, 0, 0), -math.radians(world.wheels[0]))
        self.wheels[1].local.rotation = _about((1, 0, 0), -math.radians(world.wheels[1]))
        for name, pose in world.things.items():
            _place(self.things[name], pose.position, pose.rotation)
        last = self.trail_points[self.trail_count - 1] if self.trail_count else None
        if self.trail_count < self.MAX_TRAIL and (last is None or math.dist(last[::2], (x, z)) >= 0.5):
            self.trail_points[self.trail_count] = (x, 0.1, z)
            self.trail_buffer.update_range(self.trail_count, 1)
            self.trail_count += 1

        if self.chase:  # behind and above, turning with the robot
            back = la.vec_transform_quat((0, 50 * self.zoom, 70 * self.zoom), _about((0, 1, 0), turn))
            wanted = np.array((x, 0, z)) + back
            look = (x, 5, z)
        else:  # straight down
            wanted = np.array((x, 260 * self.zoom, z + 1))
            look = (x, 0, z)
        position = self.camera.local.position
        self.camera.local.position = position + (wanted - position) * 0.08
        self.camera.look_at(look)  # pyright: ignore[reportArgumentType]  (a position will do)
        self.sun.local.position = (x + 80, 160, z + 60)
        self.sun.target.local.position = (x, 0, z)


def set_quality(renderer: gfx.WgpuRenderer, scale: float) -> None:
    # The canvas has physical pixels already: no extra supersampling. And from
    # 200% up pixels are small enough that antialiasing isn't worth its pass.
    renderer.pixel_ratio = scale
    renderer.ppaa = "none" if scale >= 2 else "ddaa"


# ----- drawing it ------------------------------------------------------------------


class _Group(BaseCanvasGroup):
    pass


class ExternalCanvas(BaseRenderCanvas):
    """
    A rendercanvas for a surface the app's own window owns. It has no loop of
    its own: the app draws with force_draw() when it wants a frame.
    """

    _rc_canvas_group = _Group(None)  # pyright: ignore[reportArgumentType]  (no loop: the app draws)

    def __init__(self) -> None:
        self._closed = False
        super().__init__(size=(1, 1))
        self._final_canvas_init()

    def resize(self, width: float, height: float, scale: float) -> None:
        """The size of the view in points (DIPs), and how many pixels there are to one."""
        self._size_info.set_physical_size(max(1, round(width * scale)), max(1, round(height * scale)), scale)

    def present_info(self) -> dict:
        """What wgpu's get_surface_id_from_info() gets: the platform's surface."""
        raise NotImplementedError

    def _rc_get_present_info(self, present_methods):
        return self.present_info() if "screen" in present_methods else None

    def _rc_force_paint(self):
        self._time_to_paint()

    def _rc_close(self):
        self._closed = True

    def _rc_get_closed(self):
        return self._closed

    # Nothing to do for the rest: the app owns the window, the loop and the size.
    def _rc_gui_poll(self):
        pass

    def _rc_request_draw(self):
        pass

    def _rc_request_paint(self):
        pass

    def _rc_set_logical_size(self, width, height):
        pass

    def _rc_set_title(self, title):
        pass

    def _rc_set_cursor(self, cursor):
        pass


class RenderThread:
    """
    A RobotScene drawn by a thread of its own. 60 times a second it calls
    step(dt), on that thread, for the world, and draws it if anything in view
    has moved: a robot at rest costs the GPU nothing. The platform's view
    makes the canvas, sets size as the view is laid out, and calls
    first_frame() on whichever thread its surface needs.
    """

    FPS = 60

    def __init__(self, step: Callable[[float], World]) -> None:
        self.step = step
        self.scene = RobotScene()
        self.size: tuple[float, float, float] | None = None  # the latest layout: width, height, scale
        self.canvas: ExternalCanvas | None = None
        self.renderer: gfx.WgpuRenderer | None = None
        self._drawn = threading.Event()  # the first frame
        self.on_first_frame: Callable[[], object] = lambda: None  # called on the render thread
        self.frame_gaps: list[float] = []
        self.frames_drawn = 0
        self._reset = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="render", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def reset(self) -> None:
        """Clears the trail, on the render thread's next frame."""
        self._reset = True

    def first_frame(self, canvas: ExternalCanvas) -> None:
        """Draws the first frame into canvas, at self.size, and lets the thread go on."""
        assert self.size is not None, "lay the view out first"
        renderer = gfx.WgpuRenderer(canvas)
        canvas.request_draw(functools.partial(renderer.render, self.scene.scene, self.scene.camera))
        set_quality(renderer, self.size[2])
        canvas.resize(*self.size)
        self.scene.pose(self.step(0.0))
        canvas.force_draw()
        self.canvas, self.renderer = canvas, renderer
        self._drawn.set()

    def _run(self) -> None:
        try:
            self._frames()
        except Exception:
            log.exception("the 3D view stopped")

    def _frames(self) -> None:
        """
        Steps and draws on the clock: a late frame makes the next one sooner
        rather than pushing them all back.
        """
        while not self._drawn.wait(0.1):
            if self._stop.is_set():
                return
        canvas, renderer = self.canvas, self.renderer
        assert canvas is not None and renderer is not None
        self.on_first_frame()
        size = self.size
        drawn = None  # what the last frame drawn showed: RobotScene.looks()
        period = 1 / self.FPS
        last = next_frame = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            self.frame_gaps.append(now - last)
            dt, last = min(now - last, 0.1), now
            world = self.step(dt)
            if self._reset:
                self._reset = False
                self.scene.reset()
            self.scene.pose(world)
            looks = self.scene.looks()
            if self.size != size and (size := self.size) is not None:
                set_quality(renderer, size[2])
                canvas.resize(*size)
                drawn = None
            # A hundredth of a millimeter, or of a degree or so: settled.
            if drawn is None or np.abs(looks - drawn).max() > 1e-3:
                canvas.force_draw()
                self.frames_drawn += 1
                drawn = looks
            next_frame = max(next_frame + period, time.perf_counter() - period)
            self._stop.wait(max(0.0, next_frame - time.perf_counter()))
