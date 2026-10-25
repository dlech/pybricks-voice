"""The 3D view: pygfx, presented into a WinUI SwapChainPanel.

wgpu-native can make a surface from a SwapChainPanel's ISwapChainPanelNative
(WGPUSurfaceSourceSwapChainPanel, D3D12 only), which PyWinRT's
winrt.runtime.interop.as_interface() gives. wgpu-py's Python layer only makes
surfaces from window handles, so its get_surface_id_from_info is wrapped to
handle the panel too; that is wgpu-py internals, hence its pinned version.

The panel is an ordinary XAML element, so XAML (the HUD) can be drawn on top.
It shows its swap chain one pixel per DIP, though, and wgpu doesn't give the
swap chain the inverse of the composition scale, so the panel is laid out at
physical size in a Canvas and scaled back down with a RenderTransform.

One unit is 1 cm, y up, forward is -z, as in three.js.
"""

from __future__ import annotations

import ctypes
import math
import os
import pathlib
import uuid
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

from .robot import State  # noqa: E402

ISWAPCHAINPANELNATIVE = uuid.UUID("63aad0b8-7c24-40ff-85a8-640d944cc325")  # microsoft.ui.xaml.media.dxinterop.h


def use_d3d12() -> str:
    """Has pygfx render with D3D12, the only backend that can present into a
    SwapChainPanel; returns the adapter's name. Call before the first render."""
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
    """A rendercanvas that presents into a SwapChainPanel. It has no loop of its
    own: the app draws with force_draw() when it wants a frame."""

    _rc_canvas_group = _PanelGroup(None)  # pyright: ignore[reportArgumentType]  (no loop: the app draws)

    def __init__(self, panel: SwapChainPanel) -> None:
        # The capsule holds a reference to ISwapChainPanelNative for as long as
        # the canvas lives.
        self._native = as_interface(panel, ISWAPCHAINPANELNATIVE)
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


def _material(color: str, roughness: float = 0.5) -> gfx.MeshStandardMaterial:
    return gfx.MeshStandardMaterial(color=color, roughness=roughness)


def _about(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    return la.quat_from_axis_angle(axis, angle)


def _box(width: float, height: float, depth: float) -> gfx.Geometry:
    return gfx.box_geometry(width, height, depth)  # pyright: ignore[reportArgumentType]  (typed int, takes float)


class RobotScene:
    """The table, a SPIKE-ish driving base, its trail, and a camera that
    follows it from behind or looks down from above."""

    MAX_TRAIL = 50_000

    def __init__(self) -> None:
        self.scene = gfx.Scene()
        self.scene.add(gfx.Background.from_color("#1b1f27"))
        self.scene.add(gfx.AmbientLight("#dde6ff", 0.5))
        self.sun = gfx.DirectionalLight("#ffffff", 2.4, cast_shadow=True)  # pyright: ignore[reportArgumentType]
        assert self.sun.shadow is not None
        self.sun.shadow.camera.width = self.sun.shadow.camera.height = 240
        self.scene.add(self.sun)

        # The table: a big floor with a 10 cm grid and a 1 m grid on top.
        floor = gfx.Mesh(gfx.plane_geometry(2000, 2000), _material("#2a303b", 0.95))
        floor.local.rotation = _about((1, 0, 0), -math.pi / 2)
        floor.receive_shadow = True
        fine = gfx.GridHelper(1000, 100, "#3a4250", "#343b47")
        coarse = gfx.GridHelper(1000, 10, "#56617a", "#56617a")
        fine.local.y, coarse.local.y = 0.02, 0.03
        self.scene.add(floor, fine, coarse)

        # The axle midpoint is the robot's reference point, as it is for
        # Pybricks' DriveBase.
        self.robot = gfx.Group()
        white, yellow = _material("#f2f2f2"), _material("#ffcd03", 0.45)
        dark, teal = _material("#23262b", 0.8), _material("#00a3a6")
        for size, material, at in (
            ((9.6, 3.2, 12), white, (0, 4.2, -2)),  # frame
            ((8.8, 3.2, 5.6), yellow, (0, 7.4, -2.5)),  # hub
            ((5, 0.6, 3), dark, (0, 9.1, -2.5)),  # hub screen
            ((2.4, 2.4, 3.2), teal, (0, 4, -9.2)),  # sensor at the front
        ):
            self.robot.add(self._part(_box(*size), material, at))
        self.wheels = []
        for side in (-1, 1):
            wheel = gfx.Group()
            tyre = self._part(gfx.cylinder_geometry(2.8, 2.8, 1.4, 32), dark, (0, 0, 0))
            tyre.local.rotation = _about((0, 1, 0), math.pi / 2)  # pygfx's cylinders run along z
            spoke = self._part(_box(1.5, 3.6, 0.8), white, (side * 0.4, 0, 0))  # shows the spin
            wheel.add(tyre, spoke)
            wheel.local.position = (side * 5.6, 2.8, 0)
            self.robot.add(wheel)
            self.wheels.append(wheel)
        self.robot.add(self._part(gfx.sphere_geometry(1, 16, 16), white, (0, 1, 6.5)))  # caster
        arrow = gfx.Mesh(gfx.cone_geometry(1.2, 3, 3), gfx.MeshBasicMaterial(color="#4f8cff"))
        arrow.local.position = (0, 11, -6)
        arrow.local.rotation = _about((0, 1, 0), math.pi)  # pygfx's cones point along +z
        self.robot.add(arrow)
        self.scene.add(self.robot)

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

    def pose(self, s: State) -> None:
        x, z = s.x / 10, -s.y / 10  # mm -> cm, forward is -z
        turn = -math.radians(s.heading)
        self.robot.local.position = (x, 0, z)
        self.robot.local.rotation = _about((0, 1, 0), turn)
        self.wheels[0].local.rotation = _about((1, 0, 0), -math.radians(s.left_angle))
        self.wheels[1].local.rotation = _about((1, 0, 0), -math.radians(s.right_angle))
        last = self.trail_points[self.trail_count - 1] if self.trail_count else None
        if self.trail_count < self.MAX_TRAIL and (last is None or math.dist(last[::2], (x, z)) >= 0.5):
            self.trail_points[self.trail_count] = (x, 0.1, z)
            self.trail_buffer.update_range(self.trail_count, 1)
            self.trail_count += 1

        if self.chase:  # behind and above, turning with the robot
            back = la.vec_transform_quat((0, 45 * self.zoom, 60 * self.zoom), _about((0, 1, 0), turn))
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
    """A RobotScene drawn into a SwapChainPanel that sits in a Canvas."""

    def __init__(self, host: FrameworkElement, panel: SwapChainPanel) -> None:
        self.host = host
        self.panel = panel
        self.scene = RobotScene()
        self.canvas: SwapChainPanelCanvas | None = None
        self.scale = 0.0
        self.watching_root = False
        host.add_size_changed(lambda s, e: self.layout())

    @property
    def ready(self) -> bool:
        return self.canvas is not None

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
        if self.canvas is None:
            self.canvas = SwapChainPanelCanvas(self.panel)
            self.renderer = gfx.WgpuRenderer(self.canvas)
            self.canvas.request_draw(lambda: self.renderer.render(self.scene.scene, self.scene.camera))
        # The canvas has physical pixels already: no extra supersampling.
        self.renderer.pixel_ratio = scale
        self.canvas.resize(width, height, scale)

    def draw(self, s: State) -> None:
        self.scene.pose(s)
        if self.canvas is not None:
            self.canvas.force_draw()
