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
import logging
import sys
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

assert not TYPE_CHECKING or sys.platform == "win32"

import wgpu
from wgpu.backends.wgpu_native import _api, _helpers
from wgpu.backends.wgpu_native._ffi import ffi, lib
from winrt.microsoft.ui.xaml import FrameworkElement
from winrt.microsoft.ui.xaml.controls import SwapChainPanel
from winrt.microsoft.ui.xaml.media import ScaleTransform
from winrt.runtime.interop import as_interface

from ..physics import World
from ..scene import ExternalCanvas, RenderThread, gfx

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


class SwapChainPanelCanvas(ExternalCanvas):
    """A canvas that presents into a SwapChainPanel."""

    def __init__(self, native: object) -> None:
        # The panel's ISwapChainPanelNative, as a capsule that holds a reference
        # for as long as the canvas lives.
        self._native = native
        super().__init__()

    def present_info(self) -> dict:
        pointer = _PyCapsule_GetPointer(self._native, b"winrt.interface")
        return {"method": "screen", "platform": "winui-swapchainpanel", "panel_native": pointer, "window": 0}


class SceneView(RenderThread):
    """
    scene.py's RenderThread, drawing into a SwapChainPanel that sits in a
    Canvas.

    The first frame is drawn on the UI thread, though: that is when wgpu gives
    the panel its swap chain, with ISwapChainPanelNative::SetSwapChain, which
    fails on any other thread. Resizing the swap chain later doesn't need the
    panel.
    """

    def __init__(self, host: FrameworkElement, panel: SwapChainPanel, step: Callable[[float], World]) -> None:
        super().__init__(step)
        self.host = host
        self.panel = panel
        # Asked for on the UI thread: the panel is a XAML object.
        self.native = as_interface(panel, ISWAPCHAINPANELNATIVE)
        self.scale = 0.0
        self.watching_root = False
        host.add_size_changed(lambda s, e: self.layout())

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
            self.first_frame(SwapChainPanelCanvas(self.native))
