"""
The few CoreFoundation calls the guest loop needs, declared by hand with
ctypes until generated bindings replace them. Signatures are from the SDK's
CFRunLoop.h and CFFileDescriptor.h.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import c_double, c_int32, c_long, c_ubyte, c_uint32, c_ulong, c_void_p
from typing import TYPE_CHECKING, Any

assert not TYPE_CHECKING or sys.platform == "darwin"

_cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

# Every CF object is a pointer; Python sees it as an int, or None for NULL.
CFTypeRef = c_void_p
CFAbsoluteTime = CFTimeInterval = c_double
CFOptionFlags = c_ulong
CFIndex = c_long
Boolean = c_ubyte

kCFStringEncodingUTF8 = 0x08000100
kCFFileDescriptorReadCallBack = 1 << 0
kCFFileDescriptorWriteCallBack = 1 << 1
kCFRunLoopRunFinished, kCFRunLoopRunStopped, kCFRunLoopRunTimedOut, kCFRunLoopRunHandledSource = 1, 2, 3, 4

kCFRunLoopDefaultMode: int = c_void_p.in_dll(_cf, "kCFRunLoopDefaultMode").value or 0
kCFRunLoopCommonModes: int = c_void_p.in_dll(_cf, "kCFRunLoopCommonModes").value or 0

CFRunLoopTimerCallBack = ctypes.CFUNCTYPE(None, CFTypeRef, c_void_p)
CFFileDescriptorCallBack = ctypes.CFUNCTYPE(None, CFTypeRef, CFOptionFlags, c_void_p)
CFRunLoopSourcePerform = ctypes.CFUNCTYPE(None, c_void_p)


class CFRunLoopSourceContext(ctypes.Structure):
    """A version 0 source: signaled, then performed on its run loop's thread."""

    _fields_ = [
        ("version", CFIndex),
        ("info", c_void_p),
        ("retain", c_void_p),
        ("release", c_void_p),
        ("copyDescription", c_void_p),
        ("equal", c_void_p),
        ("hash", c_void_p),
        ("schedule", c_void_p),
        ("cancel", c_void_p),
        ("perform", CFRunLoopSourcePerform),
    ]


def _declare(name: str, restype: type | None, *argtypes: type) -> Any:
    function = getattr(_cf, name)
    function.restype = restype
    function.argtypes = argtypes
    return function


CFRelease = _declare("CFRelease", None, CFTypeRef)
CFAbsoluteTimeGetCurrent = _declare("CFAbsoluteTimeGetCurrent", CFAbsoluteTime)
CFStringCreateWithCString = _declare("CFStringCreateWithCString", CFTypeRef, CFTypeRef, ctypes.c_char_p, c_uint32)

CFRunLoopGetCurrent = _declare("CFRunLoopGetCurrent", CFTypeRef)
CFRunLoopGetMain = _declare("CFRunLoopGetMain", CFTypeRef)
CFRunLoopRunInMode = _declare("CFRunLoopRunInMode", c_int32, CFTypeRef, CFTimeInterval, Boolean)
CFRunLoopStop = _declare("CFRunLoopStop", None, CFTypeRef)
CFRunLoopWakeUp = _declare("CFRunLoopWakeUp", None, CFTypeRef)
CFRunLoopAddCommonMode = _declare("CFRunLoopAddCommonMode", None, CFTypeRef, CFTypeRef)
CFRunLoopAddSource = _declare("CFRunLoopAddSource", None, CFTypeRef, CFTypeRef, CFTypeRef)
CFRunLoopRemoveSource = _declare("CFRunLoopRemoveSource", None, CFTypeRef, CFTypeRef, CFTypeRef)
CFRunLoopAddTimer = _declare("CFRunLoopAddTimer", None, CFTypeRef, CFTypeRef, CFTypeRef)

CFRunLoopSourceCreate = _declare(
    "CFRunLoopSourceCreate", CFTypeRef, CFTypeRef, CFIndex, ctypes.POINTER(CFRunLoopSourceContext)
)
CFRunLoopSourceSignal = _declare("CFRunLoopSourceSignal", None, CFTypeRef)
CFRunLoopSourceInvalidate = _declare("CFRunLoopSourceInvalidate", None, CFTypeRef)

# The context is optional (NULL), so it is declared as a plain pointer.
CFRunLoopTimerCreate = _declare(
    "CFRunLoopTimerCreate",
    CFTypeRef,
    CFTypeRef,
    CFAbsoluteTime,
    CFTimeInterval,
    CFOptionFlags,
    CFIndex,
    CFRunLoopTimerCallBack,
    c_void_p,
)
CFRunLoopTimerSetNextFireDate = _declare("CFRunLoopTimerSetNextFireDate", None, CFTypeRef, CFAbsoluteTime)
CFRunLoopTimerInvalidate = _declare("CFRunLoopTimerInvalidate", None, CFTypeRef)

CFFileDescriptorCreate = _declare(
    "CFFileDescriptorCreate", CFTypeRef, CFTypeRef, ctypes.c_int, Boolean, CFFileDescriptorCallBack, c_void_p
)
CFFileDescriptorCreateRunLoopSource = _declare(
    "CFFileDescriptorCreateRunLoopSource", CFTypeRef, CFTypeRef, CFTypeRef, CFIndex
)
CFFileDescriptorEnableCallBacks = _declare("CFFileDescriptorEnableCallBacks", None, CFTypeRef, CFOptionFlags)
CFFileDescriptorDisableCallBacks = _declare("CFFileDescriptorDisableCallBacks", None, CFTypeRef, CFOptionFlags)
CFFileDescriptorGetNativeDescriptor = _declare("CFFileDescriptorGetNativeDescriptor", ctypes.c_int, CFTypeRef)
CFFileDescriptorInvalidate = _declare("CFFileDescriptorInvalidate", None, CFTypeRef)


def cfstring(text: str) -> int:
    """A new CFString, which the caller releases."""
    string = CFStringCreateWithCString(None, text.encode(), kCFStringEncodingUTF8)
    if not string:
        raise ValueError(f"not a CFString: {text!r}")
    return string
