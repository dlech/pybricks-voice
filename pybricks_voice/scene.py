"""
The 3D view: pygfx, presented into a WinUI SwapChainPanel.

wgpu-native can make a surface from a SwapChainPanel's ISwapChainPanelNative
(WGPUSurfaceSourceSwapChainPanel, D3D12 only), which PyWinRT's
winrt.runtime.interop.as_interface() gives. wgpu-py's Python layer only makes
surfaces from window handles, so its get_surface_id_from_info is wrapped to
handle the panel too; that is wgpu-py internals, hence its pinned version.

The panel is an ordinary XAML element, so XAML (the HUD) can be drawn on top.
It shows its swap chain one pixel per DIP, though, and wgpu doesn't give the
swap chain the inverse of the composition scale, so the panel is laid out at
physical size in a Canvas and scaled back down with a RenderTransform.

The scene is drawn by a thread of its own. The UI thread stops running the
app's work whenever Windows runs a modal loop, for a few hundred milliseconds
each time the window starts to move or resize, and the view would freeze with
it; a SwapChainPanel's swap chain can be presented from any thread.

One unit is 1 cm, y up, forward is -z, as in three.js.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import math
import os
import pathlib
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

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
import wgpu  # noqa: E402
from rendercanvas.base import BaseCanvasGroup, BaseRenderCanvas  # noqa: E402
from wgpu.backends.wgpu_native import _api, _helpers  # noqa: E402
from wgpu.backends.wgpu_native._ffi import ffi, lib  # noqa: E402
from winrt.microsoft.ui.xaml import FrameworkElement  # noqa: E402
from winrt.microsoft.ui.xaml.controls import SwapChainPanel  # noqa: E402
from winrt.microsoft.ui.xaml.media import ScaleTransform  # noqa: E402
from winrt.runtime.interop import as_interface  # noqa: E402

from . import ldraw  # noqa: E402
from .physics import THINGS, World  # noqa: E402

log = logging.getLogger(__name__)

ISWAPCHAINPANELNATIVE = uuid.UUID("63aad0b8-7c24-40ff-85a8-640d944cc325")  # microsoft.ui.xaml.media.dxinterop.h


def use_d3d12() -> str:
    """
    Has pygfx render with D3D12, the only backend that can present into a
    SwapChainPanel; returns the adapter's name. Call before the first render.
    """
    adapters = [a for a in wgpu.gpu.enumerate_adapters_sync() if a.info["backend_type"] == "D3D12"]
    adapters.sort(key=lambda a: a.info["adapter_type"] not in ("DiscreteGPU", "IntegratedGPU"))
    if not adapters:
        raise RuntimeError("no D3D12 adapter")
    gfx.renderers.wgpu.select_adapter(adapters[0])
    return adapters[0].info["device"]


# ----- a wgpu surface from a SwapChainPanel ----------------------------------------

_surface_from_info = _api.get_surface_id_from_info


def _get_surface_id(present_info: dict):
    if present_info.get("platform") != "winui-swapchainpanel":
        return _surface_from_info(present_info)
    # cffi's structs and functions come from the C header at run time.
    source: Any = ffi.new("WGPUSurfaceSourceSwapChainPanel *")
    native: Any = lib
    source.chain.sType = native.WGPUSType_SurfaceSourceSwapChainPanel
    source.panelNative = ffi.cast("void *", present_info["panel_native"])
    descriptor: Any = ffi.new("WGPUSurfaceDescriptor *")
    descriptor.label.data = ffi.NULL
    descriptor.nextInChain = ffi.cast("WGPUChainedStruct *", source)
    return native.wgpuInstanceCreateSurface(_helpers.get_wgpu_instance(), descriptor)


_api.get_surface_id_from_info = _get_surface_id

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_GetPointer.restype = ctypes.c_void_p


class _PanelGroup(BaseCanvasGroup):
    pass


class SwapChainPanelCanvas(BaseRenderCanvas):
    """
    A rendercanvas that presents into a SwapChainPanel. It has no loop of its
    own: the app draws with force_draw() when it wants a frame.
    """

    _rc_canvas_group = _PanelGroup(None)  # pyright: ignore[reportArgumentType]  (no loop: the app draws)

    def __init__(self, native: object) -> None:
        # The panel's ISwapChainPanelNative, as a capsule that holds a reference
        # for as long as the canvas lives.
        self._native = native
        self._closed = False
        super().__init__(size=(1, 1))
        self._final_canvas_init()

    def resize(self, width: float, height: float, scale: float) -> None:
        """The DIP size of the view, and how many pixels there are to a DIP."""
        self._size_info.set_physical_size(max(1, round(width * scale)), max(1, round(height * scale)), scale)

    def _rc_get_present_info(self, present_methods):
        if "screen" not in present_methods:
            return None
        pointer = _PyCapsule_GetPointer(self._native, b"winrt.interface")
        return {"method": "screen", "platform": "winui-swapchainpanel", "panel_native": pointer, "window": 0}

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


class SceneView:
    """
    A RobotScene drawn into a SwapChainPanel that sits in a Canvas, 60 times a
    second by a thread of its own. Each frame it calls step(dt), on that thread,
    for the world to draw.

    The first frame is drawn on the UI thread, though: that is when wgpu gives
    the panel its swap chain, with ISwapChainPanelNative::SetSwapChain, which
    fails on any other thread. Resizing the swap chain later doesn't need the
    panel.
    """

    FPS = 60

    def __init__(self, host: FrameworkElement, panel: SwapChainPanel, step: Callable[[float], World]) -> None:
        self.host = host
        self.panel = panel
        self.step = step
        self.scene = RobotScene()
        # Asked for on the UI thread: the panel is a XAML object.
        self.native = as_interface(panel, ISWAPCHAINPANELNATIVE)
        self.scale = 0.0
        self.watching_root = False
        self.size: tuple[float, float, float] | None = None  # the latest layout, for the render thread
        self.canvas: SwapChainPanelCanvas | None = None
        self.renderer: gfx.WgpuRenderer | None = None
        self._drawn = threading.Event()  # the first frame, on the UI thread
        self.on_first_frame: Callable[[], object] = lambda: None  # called on the render thread
        self.frame_gaps: list[float] = []
        self._reset = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="render", daemon=True)
        host.add_size_changed(lambda s, e: self.layout())

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def reset(self) -> None:
        """Clears the trail, on the render thread's next frame."""
        self._reset = True

    def layout(self) -> None:
        width, height = self.host.actual_width, self.host.actual_height
        root = self.panel.xaml_root
        if width < 1 or height < 1 or root is None:
            return
        if not self.watching_root:
            # Moving to a display with another scale.
            root.add_changed(lambda s, e: self.layout())
            self.watching_root = True
        # The display's scale. Not the panel's CompositionScaleX: that includes
        # the RenderTransform below, so following it would undo the transform,
        # then redo it, frame after frame.
        scale = root.rasterization_scale or 1.0
        if scale != self.scale:
            self.scale = scale
            shrink = ScaleTransform()
            shrink.scale_x = shrink.scale_y = 1 / scale
            self.panel.render_transform = shrink
        self.panel.width, self.panel.height = width * scale, height * scale
        self.size = (width, height, scale)
        if self.canvas is None:
            canvas = SwapChainPanelCanvas(self.native)
            renderer = gfx.WgpuRenderer(canvas)
            canvas.request_draw(functools.partial(renderer.render, self.scene.scene, self.scene.camera))
            # The canvas has physical pixels already: no extra supersampling.
            renderer.pixel_ratio = scale
            canvas.resize(width, height, scale)
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
            if self.size != size and (size := self.size) is not None:
                renderer.pixel_ratio = size[2]
                canvas.resize(*size)
            canvas.force_draw()
            next_frame = max(next_frame + period, time.perf_counter() - period)
            self._stop.wait(max(0.0, next_frame - time.perf_counter()))
