import ctypes
import pathlib
import tempfile
import unittest
from ctypes import wintypes

from pybricks_voice import placement

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.CreateWindowExW.restype = wintypes.HWND
_user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
_user32.CreateWindowExW.argtypes += [ctypes.c_int] * 4 + [wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE]
_user32.CreateWindowExW.argtypes += [wintypes.LPVOID]
_user32.DestroyWindow.argtypes = [wintypes.HWND]
_user32.MoveWindow.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.BOOL]
_user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]

WS_OVERLAPPEDWINDOW = 0x00CF0000


class TestPlacement(unittest.TestCase):
    def setUp(self) -> None:
        # A hidden top-level window of the built-in STATIC class.
        self.hwnd = _user32.CreateWindowExW(0, "STATIC", "test", WS_OVERLAPPEDWINDOW, 0, 0, 100, 100, 0, 0, 0, 0)
        self.assertTrue(self.hwnd, ctypes.WinError(ctypes.get_last_error()))
        self.addCleanup(_user32.DestroyWindow, self.hwnd)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = pathlib.Path(directory.name) / "nested" / "window.json"

    def rect(self) -> tuple[int, int, int, int]:
        rect = wintypes.RECT()
        _user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
        return rect.left, rect.top, rect.right, rect.bottom

    def test_round_trip(self) -> None:
        _user32.MoveWindow(self.hwnd, 120, 80, 640, 480, False)
        placement.save(self.hwnd, self.path)
        _user32.MoveWindow(self.hwnd, 10, 10, 200, 200, False)
        self.assertFalse(placement.restore(self.hwnd, self.path))
        self.assertEqual(self.rect(), (120, 80, 760, 560))
        self.assertFalse(_user32.IsWindowVisible(self.hwnd))  # restored while still hidden

    def test_maximized(self) -> None:
        _user32.MoveWindow(self.hwnd, 120, 80, 640, 480, False)
        _user32.ShowWindow(self.hwnd, placement.SW_SHOWMAXIMIZED)
        placement.save(self.hwnd, self.path)
        _user32.ShowWindow(self.hwnd, placement.SW_HIDE)
        self.assertTrue(placement.restore(self.hwnd, self.path))

    def test_never_saved(self) -> None:
        _user32.MoveWindow(self.hwnd, 10, 10, 200, 200, False)
        self.assertFalse(placement.restore(self.hwnd, self.path))
        self.assertEqual(self.rect(), (10, 10, 210, 210))

    def test_unreadable(self) -> None:
        self.path.parent.mkdir()
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertLogs(placement.log, "WARNING"):
            self.assertFalse(placement.restore(self.hwnd, self.path))


if __name__ == "__main__":
    unittest.main()
