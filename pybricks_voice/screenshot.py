"""A window to PNG with Windows.Graphics.Capture, the API the Snipping Tool uses.

It captures what is on screen for the window, the SwapChainPanel's 3D view
included (XAML's RenderTargetBitmap leaves that out), whatever is in front of
it. Graphics.Capture needs a Direct3D device, which no WinRT API creates, so
that one step is D3D11CreateDevice through ctypes, handed to PyWinRT's
Direct3D 11 interop.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import logging
import pathlib
import uuid

from winrt.microsoft.ui.interop import get_window_from_window_id
from winrt.microsoft.ui.xaml import Window
from winrt.windows.graphics.capture import (
    Direct3D11CaptureFramePool,
    GraphicsCaptureAccess,
    GraphicsCaptureAccessKind,
)
from winrt.windows.graphics.capture.interop import create_for_window
from winrt.windows.graphics.directx import DirectXPixelFormat
from winrt.windows.graphics.directx.direct3d11 import IDirect3DDevice
from winrt.windows.graphics.directx.direct3d11.interop import create_direct3d11_device_from_dxgi_device
from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapEncoder, BitmapPixelFormat, SoftwareBitmap
from winrt.windows.security.authorization.appcapabilityaccess import AppCapabilityAccessStatus
from winrt.windows.storage.streams import DataReader, InMemoryRandomAccessStream

log = logging.getLogger(__name__)

_IID_IDXGIDEVICE = uuid.UUID("54ec77fa-1377-44e6-8c32-88fd5f44c84c")


def _direct3d_device() -> IDirect3DDevice:
    """A hardware D3D11 device, as the IDirect3DDevice Graphics.Capture takes."""
    device = ctypes.c_void_p()
    hardware, bgra_support, sdk_version = 1, 0x20, 7  # D3D_DRIVER_TYPE_HARDWARE, ..._BGRA_SUPPORT, D3D11_SDK_VERSION
    hr = ctypes.windll.d3d11.D3D11CreateDevice(
        None, hardware, None, bgra_support, None, 0, sdk_version, ctypes.byref(device), None, None
    )
    if hr != 0:
        raise OSError(hr, "D3D11CreateDevice failed")

    def method(obj: ctypes.c_void_p, index: int, *argtypes):
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[index])

    # IUnknown::QueryInterface (0) for IDXGIDevice, then Release (2) both.
    iid = (ctypes.c_byte * 16).from_buffer_copy(_IID_IDXGIDEVICE.bytes_le)
    dxgi = ctypes.c_void_p()
    hr = method(device, 0, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(device, iid, ctypes.byref(dxgi))
    method(device, 2)(device)
    if hr != 0 or dxgi.value is None:
        raise OSError(hr, "IDXGIDevice not supported")
    try:
        return create_direct3d11_device_from_dxgi_device(dxgi.value)
    finally:
        method(dxgi, 2)(dxgi)


async def _capture(window: Window, timeout: float) -> SoftwareBitmap:
    """One frame of the window, as it is on screen."""
    item = create_for_window(get_window_from_window_id(window.app_window.id))
    loop = asyncio.get_running_loop()
    arrived: asyncio.Future[None] = loop.create_future()

    def on_frame_arrived(sender, args) -> None:
        # A free-threaded pool raises this on a capture thread of its own.
        loop.call_soon_threadsafe(lambda: arrived.done() or arrived.set_result(None))

    # Undone in reverse, whatever happens: the handler, the session, the pool,
    # the device.
    with contextlib.ExitStack() as stack:
        device = stack.enter_context(_direct3d_device())
        pool = stack.enter_context(
            Direct3D11CaptureFramePool.create_free_threaded(
                device, DirectXPixelFormat.B8_G8_R8_A8_UINT_NORMALIZED, 1, item.size
            )
        )
        session = stack.enter_context(pool.create_capture_session(item))
        stack.callback(pool.remove_frame_arrived, pool.add_frame_arrived(on_frame_arrived))

        # Without this Windows draws a yellow border around the window while
        # it is captured; Windows 11 lets an app ask to leave it out.
        access = await GraphicsCaptureAccess.request_access_async(GraphicsCaptureAccessKind.BORDERLESS)
        log.debug("borderless capture: %s", access.name)
        if access == AppCapabilityAccessStatus.ALLOWED:
            session.is_border_required = False
        session.is_cursor_capture_enabled = False
        session.start_capture()
        async with asyncio.timeout(timeout):
            await arrived
        frame = pool.try_get_next_frame()
        if frame is None:
            raise RuntimeError("FrameArrived without a frame")
        with frame:
            return await SoftwareBitmap.create_copy_from_surface_async(frame.surface)


async def save_window(window: Window, path: pathlib.Path, timeout: float = 2.0) -> None:
    """Saves one frame of the window, as it is on screen, to a PNG."""
    with (
        await _capture(window, timeout) as bitmap,
        # The frame is premultiplied BGRA; a PNG of the window wants it opaque.
        SoftwareBitmap.convert(bitmap, BitmapPixelFormat.BGRA8, BitmapAlphaMode.IGNORE) as opaque,
        InMemoryRandomAccessStream() as stream,
    ):
        encoder = await BitmapEncoder.create_async(BitmapEncoder.png_encoder_id, stream)
        encoder.set_software_bitmap(opaque)
        await encoder.flush_async()
        with DataReader(stream.get_input_stream_at(0)) as reader:
            data = bytes(reader.read_buffer(await reader.load_async(stream.size)))
        width, height = opaque.pixel_width, opaque.pixel_height
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    log.info("saved %s (%d x %d)", path, width, height)
