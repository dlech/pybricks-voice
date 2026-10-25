"""
Spike: why the guest asyncio loop stalls when a window drag starts.

A WinUI window with the guest loop ticking every 16 ms. It logs, with times:
  - each tick, and each gap between ticks over 40 ms
  - the loop asking for an iteration (a DispatcherQueue work item, or its
    DispatcherQueueTimer) and the iteration running
  - every message the UI thread takes from its queue (WH_GETMESSAGE) and every
    message sent to its windows (WH_CALLWNDPROC)
and, when the UI thread stalls for more than 150 ms, a native stack of it from
cdb (the Windows SDK's console debugger), attached non-invasively.

    uv run python spikes/drag_stall.py [--synthetic] [--seconds 60] [--no-cdb]

--synthetic posts WM_NCLBUTTONDOWN on the caption and WM_SYSCOMMAND SC_DRAGMOVE
instead of waiting for a person to drag the window; the mouse isn't touched.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import ctypes.wintypes as w
import itertools
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import override

from winrt.microsoft.ui.interop import get_window_from_window_id
from winrt.microsoft.ui.xaml import Application, LaunchActivatedEventArgs, Window
from winrt.microsoft.ui.xaml.controls import TextBlock
from winrt.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import InitializeOptions, initialize
from winrt.runtime import ApartmentType, init_apartment

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pybricks_voice import loop as guest

CDB = pathlib.Path(r"C:\Program Files (x86)\Windows Kits\10\Debuggers\arm64\cdb.exe")
SYMBOLS = r"srv*C:\symbols*https://msdl.microsoft.com/download/symbols"
OUT = pathlib.Path(__file__).resolve().parent.parent / "out" / "drag_stall"

user32 = ctypes.WinDLL("user32", use_last_error=True)
HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, w.WPARAM, w.LPARAM)
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, w.HINSTANCE, w.DWORD]
user32.SetWindowsHookExW.restype = w.HHOOK
user32.CallNextHookEx.argtypes = [w.HHOOK, ctypes.c_int, w.WPARAM, w.LPARAM]
user32.CallNextHookEx.restype = ctypes.c_ssize_t
user32.PostMessageW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
user32.GetClassNameW.argtypes = [w.HWND, w.LPWSTR, ctypes.c_int]
user32.GetWindowRect.argtypes = [w.HWND, ctypes.POINTER(w.RECT)]
WH_GETMESSAGE, WH_CALLWNDPROC = 3, 4


class CWPSTRUCT(ctypes.Structure):
    _fields_ = [("lParam", w.LPARAM), ("wParam", w.WPARAM), ("message", w.UINT), ("hwnd", w.HWND)]


NAMES = {
    0x0006: "WM_ACTIVATE",
    0x000F: "WM_PAINT",
    0x0014: "WM_ERASEBKGND",
    0x001C: "WM_ACTIVATEAPP",
    0x0020: "WM_SETCURSOR",
    0x0021: "WM_MOUSEACTIVATE",
    0x0046: "WM_WINDOWPOSCHANGING",
    0x0047: "WM_WINDOWPOSCHANGED",
    0x0084: "WM_NCHITTEST",
    0x0086: "WM_NCACTIVATE",
    0x00A0: "WM_NCMOUSEMOVE",
    0x00A1: "WM_NCLBUTTONDOWN",
    0x00A2: "WM_NCLBUTTONUP",
    0x00A3: "WM_NCLBUTTONDBLCLK",
    0x0112: "WM_SYSCOMMAND",
    0x0113: "WM_TIMER",
    0x0118: "WM_SYSTIMER",
    0x0200: "WM_MOUSEMOVE",
    0x0201: "WM_LBUTTONDOWN",
    0x0202: "WM_LBUTTONUP",
    0x0215: "WM_CAPTURECHANGED",
    0x0216: "WM_MOVING",
    0x0214: "WM_SIZING",
    0x0231: "WM_ENTERSIZEMOVE",
    0x0232: "WM_EXITSIZEMOVE",
    0x0241: "WM_NCPOINTERUPDATE",
    0x0242: "WM_NCPOINTERDOWN",
    0x0243: "WM_NCPOINTERUP",
    0x0245: "WM_POINTERUPDATE",
    0x0246: "WM_POINTERDOWN",
    0x0247: "WM_POINTERUP",
    0x024A: "WM_POINTERCAPTURECHANGED",
    0x02A1: "WM_MOUSEHOVER",
    0x02A3: "WM_MOUSELEAVE",
    0x0003: "WM_MOVE",
    0x0005: "WM_SIZE",
    0x0024: "WM_GETMINMAXINFO",
    0x0083: "WM_NCCALCSIZE",
    0x0085: "WM_NCPAINT",
    0x031F: "WM_DWMNCRENDERINGCHANGED",
    0x0060: "WM_???60",
}

SYSCOMMANDS = {0xF000: "SC_SIZE", 0xF010: "SC_MOVE", 0xF030: "SC_MAXIMIZE", 0xF120: "SC_RESTORE", 0xF020: "SC_MINIMIZE"}
CLICKS = ("WM_NCLBUTTON", "WM_LBUTTON", "WM_SYSCOMMAND", "SIZEMOVE", "nudge")

events: list[tuple[float, str, str]] = []  # (time, kind, detail)
start = time.perf_counter()


def note(kind: str, detail: str = "") -> None:
    events.append((time.perf_counter(), kind, detail))


_classes: dict[int, str] = {}


def window_class(hwnd: int) -> str:
    if hwnd not in _classes:
        buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buffer, 256)
        _classes[hwnd] = buffer.value
    return _classes[hwnd]


def message(msg: int, hwnd: int) -> str:
    name = NAMES.get(msg, f"{msg:#06x}")
    return f"{name} -> {window_class(hwnd) if hwnd else 'thread'}"


hooks = []


@HOOKPROC
def on_get_message(code: int, wparam: int, lparam: int) -> int:
    if code >= 0:
        m = ctypes.cast(lparam, ctypes.POINTER(w.MSG)).contents
        note("get" if wparam else "peek", message(m.message, m.hWnd or 0))
    return user32.CallNextHookEx(None, code, wparam, lparam)


@HOOKPROC
def on_call_wndproc(code: int, wparam: int, lparam: int) -> int:
    if code >= 0:
        c = ctypes.cast(lparam, ctypes.POINTER(CWPSTRUCT)).contents
        detail = message(c.message, c.hwnd or 0)
        if c.message == 0x0112:  # WM_SYSCOMMAND: which command
            detail += f" {SYSCOMMANDS.get(c.wParam & 0xFFF0, hex(c.wParam & 0xFFF0))}"
        note("sent", detail)
    return user32.CallNextHookEx(None, code, wparam, lparam)


# --nudge: DefWindowProc's SC_MOVE and SC_SIZE wait in the kernel for the mouse
# to move before they start their modal loop, and dispatch nothing meanwhile.
# A WM_MOUSEMOVE posted as the button goes down ends that wait at once.
comctl32 = ctypes.WinDLL("comctl32")
SUBCLASSPROC = ctypes.WINFUNCTYPE(
    ctypes.c_ssize_t, w.HWND, w.UINT, w.WPARAM, w.LPARAM, ctypes.c_size_t, ctypes.c_size_t
)
comctl32.SetWindowSubclass.argtypes = [w.HWND, SUBCLASSPROC, ctypes.c_size_t, ctypes.c_size_t]
comctl32.DefSubclassProc.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
comctl32.DefSubclassProc.restype = ctypes.c_ssize_t
user32.ScreenToClient.argtypes = [w.HWND, ctypes.POINTER(w.POINT)]
WM_NCLBUTTONDOWN, WM_MOUSEMOVE, MK_LBUTTON = 0x00A1, 0x0200, 0x0001
DRAG_HITS = {2} | set(range(10, 18))  # HTCAPTION, and HTLEFT to HTBOTTOMRIGHT


@SUBCLASSPROC
def nudge(hwnd: int, msg: int, wparam: int, lparam: int, id: int, data: int) -> int:
    if msg == WM_NCLBUTTONDOWN and wparam in DRAG_HITS:
        point = w.POINT(ctypes.c_short(lparam & 0xFFFF).value, ctypes.c_short(lparam >> 16).value)
        user32.ScreenToClient(hwnd, ctypes.byref(point))
        note("spike", "nudge: post WM_MOUSEMOVE")
        user32.PostMessageW(hwnd, WM_MOUSEMOVE, MK_LBUTTON, ((point.y & 0xFFFF) << 16) | (point.x & 0xFFFF))
    return comctl32.DefSubclassProc(hwnd, msg, wparam, lparam)


class TracedLoop(guest.DispatcherQueueEventLoop):
    def _schedule_tick(self) -> None:
        if not self._tick_pending:
            note("loop", "enqueue work item")
        super()._schedule_tick()

    def _tick(self) -> None:
        note("loop", "iteration")
        super()._tick()
        if self._timer.is_running:
            note("loop", f"timer set for {self._timer.interval.total_seconds() * 1000:.1f} ms")


class App(Application):
    def __init__(self, options: argparse.Namespace) -> None:
        super().__init__()
        self.options = options
        self.last_tick = time.perf_counter()
        self.ui_tid = threading.get_native_id()

    @override
    def _on_launched(self, args: LaunchActivatedEventArgs) -> None:
        self.window = Window()
        self.window.title = "drag stall spike"
        self.text = TextBlock()
        self.text.font_size = 32
        self.window.content = self.text
        self.window.app_window.resize((900, 600))
        self.hwnd = get_window_from_window_id(self.window.app_window.id)
        for kind, proc in ((WH_GETMESSAGE, on_get_message), (WH_CALLWNDPROC, on_call_wndproc)):
            hooks.append(user32.SetWindowsHookExW(kind, proc, None, self.ui_tid))
        if self.options.nudge:
            comctl32.SetWindowSubclass(self.hwnd, nudge, 1, 0)
        self.loop = TracedLoop(self.window.dispatcher_queue)
        self.loop.start()
        self.task = self.loop.create_task(self.main())
        self.task.add_done_callback(lambda t: (self.loop.stop(), self.exit()))
        self.window.add_closed(lambda s, e: self.task.cancel())
        self.window.activate()
        if not self.options.no_cdb:
            threading.Thread(target=self.watch, daemon=True).start()

    async def main(self) -> None:
        if self.options.synthetic:
            asyncio.get_running_loop().call_later(3, self.post, 0x00A1, 2)  # WM_NCLBUTTONDOWN, HTCAPTION
            asyncio.get_running_loop().call_later(6, self.post, 0x0112, 0xF012)  # WM_SYSCOMMAND, SC_DRAGMOVE
        end = time.perf_counter() + self.options.seconds
        n = 0
        while time.perf_counter() < end:
            await asyncio.sleep(0.016)
            now = time.perf_counter()
            gap = now - self.last_tick
            self.last_tick = now
            note("tick", f"gap {gap * 1000:.0f} ms")
            n += 1
            if n % 6 == 0:
                self.text.text = f"{now - start:.1f} s   last gap {gap * 1000:.0f} ms"

    def post(self, msg: int, wparam: int) -> None:
        rect = w.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
        x, y = rect.left + 200, rect.top + 15
        note("spike", f"post {NAMES[msg]}")
        user32.PostMessageW(self.hwnd, msg, wparam, (y << 16) | (x & 0xFFFF))

    def watch(self) -> None:
        """Takes a stack of the UI thread with cdb each time it stalls."""
        OUT.mkdir(parents=True, exist_ok=True)
        dumped_for = 0.0
        while True:
            time.sleep(0.02)
            stalled_since = self.last_tick
            if time.perf_counter() - stalled_since > 0.15 and stalled_since != dumped_for:
                dumped_for = stalled_since
                note("spike", "cdb attaching")
                result = subprocess.run(
                    [CDB, "-pv", "-p", str(os.getpid()), "-y", SYMBOLS, "-c", f"~~[{self.ui_tid:#x}]kc 80; qd"],
                    capture_output=True,
                    text=True,
                    errors="replace",
                )
                path = OUT / f"stack-{stalled_since - start:.2f}.txt"
                path.write_text(result.stdout, encoding="utf-8")
                note("spike", f"cdb wrote {path.name}")


def report() -> None:
    """Every stall, with what the UI thread did from just before it to its end."""
    lines = []
    ticks = [i for i, (_, kind, _) in enumerate(events) if kind == "tick"]
    for before, after in itertools.pairwise(ticks):
        t0, t1 = events[before][0], events[after][0]
        if t1 - t0 < 0.04:
            continue
        lines.append(f"\n=== stall of {(t1 - t0) * 1000:.0f} ms at {t0 - start:.3f} s")
        previous, repeats = None, 0
        for t, kind, detail in events[before : after + 1]:
            entry = f"{kind:5} {detail}"
            if entry == previous:
                repeats += 1
                continue
            if repeats:
                lines.append(f"{'':12}  (x{repeats + 1})")
            previous, repeats = entry, 0
            lines.append(f"{(t - t0) * 1000:9.1f} ms  {entry}")
        if repeats:
            lines.append(f"{'':12}  (x{repeats + 1})")
    lines.append("\n=== every click, the whole run")
    for t, kind, detail in events:
        if any(c in detail for c in CLICKS):
            lines.append(f"{t - start:9.3f} s  {kind:5} {detail}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines) or "no stalls")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--no-cdb", action="store_true")
    parser.add_argument("--nudge", action="store_true", help="post WM_MOUSEMOVE as a drag starts")
    options = parser.parse_args()
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    init_apartment(ApartmentType.SINGLE_THREADED)
    apps: list[App] = []
    with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
        Application.start(lambda _: apps.append(App(options)))
        for app in apps:
            app.loop.close()
    for hook in hooks:
        user32.UnhookWindowsHookEx(hook)
    report()


if __name__ == "__main__":
    main()
