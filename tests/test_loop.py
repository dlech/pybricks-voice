import asyncio
import contextlib
import ctypes
import socket
import threading
import time
import unittest
from collections.abc import Callable, Coroutine
from ctypes import wintypes
from typing import Any

from winrt.microsoft.ui.dispatching import DispatcherQueue, DispatcherQueueController
from winrt.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import initialize
from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

from pybricks_voice.loop import DispatcherQueueEventLoop

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.MessageBoxW.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]
_user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
_user32.FindWindowW.restype = wintypes.HWND
_user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.DefWindowProcW.restype = ctypes.c_ssize_t
_user32.CreateWindowExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
]
_user32.CreateWindowExW.restype = wintypes.HWND
_user32.DestroyWindow.argtypes = [wintypes.HWND]
WM_CLOSE, WM_APP = 0x0010, 0x8000
HWND_MESSAGE = wintypes.HWND(-3)

_WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class _WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", _WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


_on_app_message: list[Callable[[], None]] = []


@_WNDPROC
def _wndproc(hwnd: int, msg: int, wparam: int, lparam: int) -> int:
    if msg == WM_APP:
        _on_app_message.pop()()
        return 0
    return _user32.DefWindowProcW(hwnd, msg, wparam, lparam)


_window_class = _WNDCLASSW(lpfnWndProc=_wndproc, lpszClassName="PybricksVoiceLoopTest")
_user32.RegisterClassW(ctypes.byref(_window_class))

_runtime = contextlib.ExitStack()


def setUpModule() -> None:
    _runtime.enter_context(initialize())


def tearDownModule() -> None:
    _runtime.close()


def run(make: Callable[[DispatcherQueue], Coroutine[Any, Any, Any]], timeout: float = 10) -> Any:
    """Runs a coroutine on a guest loop, on a thread of its own whose
    DispatcherQueue is what pumps it, the way Application.start() would."""
    outcome: dict[str, Any] = {}

    def thread() -> None:
        controller = DispatcherQueueController.create_on_current_thread()
        queue = controller.dispatcher_queue
        loop = DispatcherQueueEventLoop(queue)
        try:
            loop.start()
            task = loop.create_task(make(queue))

            def done(_: asyncio.Task[Any]) -> None:
                loop.stop()
                queue.enqueue_event_loop_exit()

            task.add_done_callback(done)
            queue.run_event_loop()
            outcome["result"] = task.result()
        except BaseException as e:
            outcome["error"] = e
        finally:
            loop.close()
            controller.shutdown_queue()

    t = threading.Thread(target=thread)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError("the loop didn't finish")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def message_box_from_window_message(seconds: float, done: Callable[[], object]) -> None:
    """A modal loop entered from a window message, as a window drag is, and
    closed from another thread after a while. On the calling thread."""
    title = "pybricks-voice modal test"
    hwnd = _user32.CreateWindowExW(0, "PybricksVoiceLoopTest", None, 0, 0, 0, 0, 0, HWND_MESSAGE, None, None, None)

    def close_it() -> None:
        time.sleep(seconds)
        _user32.PostMessageW(_user32.FindWindowW(None, title), WM_CLOSE, 0, 0)

    def show() -> None:
        threading.Thread(target=close_it, daemon=True).start()
        _user32.MessageBoxW(None, "modal loop", title, 0)
        _user32.DestroyWindow(hwnd)
        done()

    _on_app_message.append(show)
    _user32.PostMessageW(hwnd, WM_APP, 0, 0)


class TestDispatcherQueueEventLoop(unittest.TestCase):
    def test_dispatcher_work_and_coroutines_share_the_thread(self) -> None:
        async def main(queue: DispatcherQueue) -> list[str]:
            loop = asyncio.get_running_loop()
            done = loop.create_future()
            seen: list[str] = []
            thread = threading.get_ident()
            queue.try_enqueue(lambda: seen.append("dispatcher" if threading.get_ident() == thread else "elsewhere"))
            # call_soon() from outside an iteration has to ask for one.
            queue.try_enqueue(lambda: loop.call_soon(done.set_result, None))
            await done
            return seen

        self.assertEqual(run(main), ["dispatcher"])

    def test_sleep_and_timers(self) -> None:
        async def main(queue: DispatcherQueue) -> tuple[float, list[str]]:
            loop = asyncio.get_running_loop()
            order: list[str] = []
            loop.call_later(0.1, order.append, "late")
            loop.call_later(0.02, order.append, "early")
            start = time.monotonic()
            await asyncio.sleep(0.2)
            return time.monotonic() - start, order

        took, order = run(main)
        self.assertGreaterEqual(took, 0.19)
        self.assertLess(took, 0.4)
        self.assertEqual(order, ["early", "late"])

    def test_awaiting_a_winrt_operation(self) -> None:
        async def main(queue: DispatcherQueue) -> int:
            writer = DataWriter(InMemoryRandomAccessStream())
            writer.write_bytes(b"hello")
            return await writer.store_async()

        self.assertEqual(run(main), 5)

    def test_another_thread_wakes_it_promptly(self) -> None:
        async def main(queue: DispatcherQueue) -> float:
            loop = asyncio.get_running_loop()
            woke = loop.create_future()
            sent = 0.0

            def other() -> None:
                nonlocal sent
                time.sleep(0.05)
                sent = time.monotonic()
                loop.call_soon_threadsafe(woke.set_result, None)

            threading.Thread(target=other).start()
            await woke
            return time.monotonic() - sent

        self.assertLess(run(main), 0.05)

    def test_executor_timeout_and_task_group(self) -> None:
        async def main(queue: DispatcherQueue) -> tuple[int, bool]:
            loop = asyncio.get_running_loop()
            async with asyncio.TaskGroup() as group:
                squares = group.create_task(asyncio.to_thread(sum, (x * x for x in range(4))))
            timed_out = False
            try:
                async with asyncio.timeout(0.05):
                    await loop.run_in_executor(None, time.sleep, 0.2)
            except TimeoutError:
                timed_out = True
            return squares.result(), timed_out

        self.assertEqual(run(main), (14, True))

    def test_sockets_and_subprocesses_through_the_completion_port(self) -> None:
        async def main(queue: DispatcherQueue) -> tuple[bytes, str]:
            loop = asyncio.get_running_loop()
            a, b = socket.socketpair()
            with a, b:
                a.setblocking(False)
                threading.Timer(0.05, b.send, (b"ping",)).start()
                data = await loop.sock_recv(a, 4)
            process = await asyncio.create_subprocess_exec("cmd", "/c", "echo", "hello", stdout=asyncio.subprocess.PIPE)
            out, _ = await process.communicate()
            return data, out.decode().strip()

        self.assertEqual(run(main), (b"ping", "hello"))

    def test_keeps_running_during_a_modal_loop(self) -> None:
        async def main(queue: DispatcherQueue) -> int:
            loop = asyncio.get_running_loop()
            ticks = 0

            async def count() -> None:
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.1)
                    ticks += 1

            counter = asyncio.create_task(count())
            closed = loop.create_future()
            message_box_from_window_message(1.0, lambda: loop.call_soon(closed.set_result, ticks))
            during = await closed
            counter.cancel()
            return during

        # About 10 ticks of 100 ms while the MessageBox was up for 1 s.
        self.assertGreaterEqual(run(main), 7)


if __name__ == "__main__":
    unittest.main()
