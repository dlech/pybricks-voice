"""
The 3D view on macOS: scene.py's RobotScene, drawn into a CAMetalLayer.

wgpu-native makes a surface from a CAMetalLayer (WGPUSurfaceSourceMetalLayer),
but wgpu-py's Python layer only takes a window, whose content view it gives a
layer of its own; so its get_surface_id_from_info is wrapped to take the layer
of a view of ours, as windows/scene_view.py does for a SwapChainPanel.

The view is an NSView whose backing layer is a CAMetalLayer, so other views
(the HUD) can sit on top of it. Its first frame is drawn on the main thread,
and then RenderThread draws from a thread of its own, which Metal allows: it
keeps drawing while AppKit tracks a live resize.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

assert not TYPE_CHECKING or sys.platform == "darwin"

from rubicon.objc import NSMakeRect, NSSize, ObjCClass, objc_method, objc_property, send_super
from rubicon.objc.runtime import load_library
from wgpu.backends.wgpu_native import _api, _helpers
from wgpu.backends.wgpu_native._ffi import ffi, lib

from ..physics import World
from ..scene import ExternalCanvas, RenderThread

load_library("AppKit")
load_library("QuartzCore")
NSView = ObjCClass("NSView")
CAMetalLayer = ObjCClass("CAMetalLayer")

# ----- a wgpu surface from a CAMetalLayer --------------------------------------------

_surface_from_info = _api.get_surface_id_from_info


def _get_surface_id(present_info: dict):
    if present_info.get("platform") != "metal-layer":
        return _surface_from_info(present_info)
    # cffi's structs and functions come from the C header at run time.
    source: Any = ffi.new("WGPUSurfaceSourceMetalLayer *")
    native: Any = lib
    source.chain.sType = native.WGPUSType_SurfaceSourceMetalLayer
    source.layer = ffi.cast("void *", present_info["layer"])
    descriptor: Any = ffi.new("WGPUSurfaceDescriptor *")
    descriptor.label.data = ffi.NULL
    descriptor.nextInChain = ffi.cast("WGPUChainedStruct *", source)
    return native.wgpuInstanceCreateSurface(_helpers.get_wgpu_instance(), descriptor)


_api.get_surface_id_from_info = _get_surface_id


class MetalLayerCanvas(ExternalCanvas):
    """A canvas that presents into a CAMetalLayer."""

    def __init__(self, layer: int) -> None:
        self._layer = layer  # the view holds the layer for as long as the canvas lives
        super().__init__()

    def present_info(self) -> dict:
        return {"method": "screen", "platform": "metal-layer", "layer": self._layer, "window": 0}


# ----- the view ---------------------------------------------------------------------


class PybricksVoiceMetalView(NSView):  # an Objective-C class: its name is global
    """An NSView backed by a CAMetalLayer, which tells its SceneView about input and size."""

    scene_view = objc_property(object, weak=True)

    @objc_method
    def makeBackingLayer(self):
        return CAMetalLayer.layer()

    @objc_method
    def acceptsFirstResponder(self) -> bool:
        return True

    @objc_method
    def mouseDown_(self, event) -> None:
        self.window.makeFirstResponder(self)

    @objc_method
    def keyDown_(self, event) -> None:
        if not self.scene_view.on_key(str(event.charactersIgnoringModifiers)):
            send_super(__class__, self, "keyDown:", event)  # the beep, and the responder chain

    @objc_method
    def scrollWheel_(self, event) -> None:
        self.scene_view.on_wheel(float(event.scrollingDeltaY))

    @objc_method
    def setFrameSize_(self, size: NSSize) -> None:
        send_super(__class__, self, "setFrameSize:", size, argtypes=[NSSize])
        self.scene_view.layout()

    @objc_method
    def viewDidMoveToWindow(self) -> None:
        self.scene_view.layout()

    @objc_method
    def viewDidChangeBackingProperties(self) -> None:
        # Moving to a display with another scale.
        self.scene_view.layout()


class SceneView(RenderThread):
    """scene.py's RenderThread, drawing into a PybricksVoiceMetalView."""

    def __init__(self, step: Callable[[float], World]) -> None:
        super().__init__(step)
        self.on_key: Callable[[str], bool] = lambda characters: False  # whether it was handled
        self.on_wheel: Callable[[float], object] = lambda delta: None  # points, up is positive
        self.view = PybricksVoiceMetalView.alloc().initWithFrame(NSMakeRect(0, 0, 640, 480))
        self.view.scene_view = self
        self.view.wantsLayer = True

    def layout(self) -> None:
        """On the main thread, whenever the view's size or scale may have changed."""
        window = self.view.window
        size = self.view.bounds.size
        if window is None or size.width < 1 or size.height < 1:
            return
        scale = float(window.backingScaleFactor)
        layer = self.view.layer
        layer.contentsScale = scale
        self.size = (float(size.width), float(size.height), scale)
        if self.canvas is None:
            self.first_frame(MetalLayerCanvas(layer.ptr.value))
