"""
The 3D view on Windows: scene.py's RobotScene, presented into a WinUI
SwapChainPanel.

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
"""

from __future__ import annotations

import ctypes
import functools
import logging
import sys
import threading
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

assert not TYPE_CHECKING or sys.platform == "win32"

import numpy as np
import wgpu
from rendercanvas.base import BaseCanvasGroup, BaseRenderCanvas
from wgpu.backends.wgpu_native import _api, _helpers
from wgpu.backends.wgpu_native._ffi import ffi, lib
from winrt.microsoft.ui.xaml import FrameworkElement
from winrt.microsoft.ui.xaml.controls import SwapChainPanel
from winrt.microsoft.ui.xaml.media import ScaleTransform
from winrt.runtime.interop import as_interface

from ..physics import World
from ..scene import RobotScene, gfx, set_quality

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


class SceneView:
    """
    A RobotScene drawn into a SwapChainPanel that sits in a Canvas, by a
    thread of its own. 60 times a second it calls step(dt), on that thread,
    for the world, and draws it if anything in view has moved: a robot at
    rest costs the GPU nothing.

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
        self.frames_drawn = 0
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
            set_quality(renderer, scale)
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
