"""
Where the window was, kept between runs.

WinUI 3 doesn't remember a window's size and position as UWP did, and
AppWindow has nothing to save them with, so this keeps Win32's
WINDOWPLACEMENT in a file: the window's restored rectangle and whether it
was maximized. SetWindowPlacement also brings a window whose monitor is
gone back onto one that's there, which saving AppWindow's position and size
would not.
"""

import ctypes
import json
import logging
import os
import pathlib
from ctypes import wintypes

log = logging.getLogger(__name__)

SW_HIDE = 0
SW_SHOWMINIMIZED = 2
SW_SHOWMAXIMIZED = 3
WPF_RESTORETOMAXIMIZED = 2


class _WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.UINT),
        ("flags", wintypes.UINT),
        ("showCmd", wintypes.UINT),
        ("ptMinPosition", wintypes.POINT),
        ("ptMaxPosition", wintypes.POINT),
        ("rcNormalPosition", wintypes.RECT),
    ]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.GetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(_WINDOWPLACEMENT)]
_user32.GetWindowPlacement.restype = wintypes.BOOL
_user32.SetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(_WINDOWPLACEMENT)]
_user32.SetWindowPlacement.restype = wintypes.BOOL

PATH = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home())) / "pybricks-voice" / "window.json"


def save(hwnd: int, path: pathlib.Path = PATH) -> None:
    """Saves where the window is, as the app closes."""
    placement = _WINDOWPLACEMENT(length=ctypes.sizeof(_WINDOWPLACEMENT))
    if not _user32.GetWindowPlacement(hwnd, ctypes.byref(placement)):
        raise ctypes.WinError(ctypes.get_last_error())
    rect = placement.rcNormalPosition
    saved = {
        "normal": [rect.left, rect.top, rect.right, rect.bottom],
        # Minimized from maximized, it is restored maximized.
        "maximized": placement.showCmd == SW_SHOWMAXIMIZED
        or (placement.showCmd == SW_SHOWMINIMIZED and bool(placement.flags & WPF_RESTORETOMAXIMIZED)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved), encoding="utf-8")


def restore(hwnd: int, path: pathlib.Path = PATH) -> bool:
    """
    Puts a window not yet shown where it was, still hidden, and returns
    whether it was maximized, for the app to maximize it once shown. A
    window never saved, or saved in a file that can't be read, stays as
    it is.
    """
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        left, top, right, bottom = (int(n) for n in saved["normal"])
        maximized = bool(saved["maximized"])
    except FileNotFoundError:
        return False
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.warning("ignoring %s: %s", path, e)
        return False
    placement = _WINDOWPLACEMENT(
        length=ctypes.sizeof(_WINDOWPLACEMENT),
        showCmd=SW_HIDE,
        rcNormalPosition=wintypes.RECT(left, top, right, bottom),
    )
    if not _user32.SetWindowPlacement(hwnd, ctypes.byref(placement)):
        raise ctypes.WinError(ctypes.get_last_error())
    return maximized
