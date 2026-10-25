import asyncio
import os
import signal
import socket
import sys
import threading
import time
import unittest
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

assert not TYPE_CHECKING or sys.platform == "darwin"

from pybricks_voice.macos.corefoundation import (
    CFAbsoluteTimeGetCurrent,
    CFRelease,
    CFRunLoopAddCommonMode,
    CFRunLoopAddTimer,
    CFRunLoopGetMain,
    CFRunLoopRunInMode,
    CFRunLoopStop,
    CFRunLoopTimerCallBack,
    CFRunLoopTimerCreate,
    cfstring,
    kCFRunLoopCommonModes,
    kCFRunLoopDefaultMode,
)
from pybricks_voice.macos.loop import CFRunLoopEventLoop

# A mode of our own, standing in for AppKit's NSEventTrackingRunLoopMode: a
# common mode, which a live resize or a menu runs the main run loop in.
TRACKING_MODE = cfstring("PybricksVoiceTrackingMode")
CFRunLoopAddCommonMode(CFRunLoopGetMain(), TRACKING_MODE)

_timers: list[Any] = []  # their callbacks, which CF holds only pointers to


def after(seconds: float, callback: Callable[[], object]) -> None:
    """Calls back from the run loop, outside asyncio, as AppKit would."""
    fire = CFRunLoopTimerCallBack(lambda timer, info: callback())
    _timers.append(fire)
    timer = CFRunLoopTimerCreate(None, CFAbsoluteTimeGetCurrent() + seconds, 0, 0, 0, fire, None)
    CFRunLoopAddTimer(CFRunLoopGetMain(), timer, kCFRunLoopCommonModes)
    CFRelease(timer)


def run(make: Callable[[], Coroutine[Any, Any, Any]], timeout: float = 10) -> Any:
    """
    Runs a coroutine on a guest loop on this thread, whose run loop is pumped
    by CFRunLoopRunInMode as NSApplication's run() would.
    """
    loop = CFRunLoopEventLoop()
    try:
        loop.start()
        task = loop.create_task(make())
        task.add_done_callback(lambda _: CFRunLoopStop(CFRunLoopGetMain()))
        deadline = time.monotonic() + timeout
        while not task.done():
            if time.monotonic() > deadline:
                raise TimeoutError("the loop didn't finish")
            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, False)
        loop.stop()
        while loop.is_running():
            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, True)
        return task.result()
    finally:
        loop.close()


class TestCFRunLoopEventLoop(unittest.TestCase):
    def test_run_loop_work_and_coroutines_share_the_thread(self) -> None:
        async def main() -> list[str]:
            loop = asyncio.get_running_loop()
            done = loop.create_future()
            seen: list[str] = []
            thread = threading.get_ident()
            after(0, lambda: seen.append("run loop" if threading.get_ident() == thread else "elsewhere"))
            # call_soon() from outside an iteration has to ask for one.
            after(0.01, lambda: loop.call_soon(done.set_result, None))
            await done
            return seen

        self.assertEqual(run(main), ["run loop"])

    def test_sleep_and_timers(self) -> None:
        async def main() -> tuple[float, list[str]]:
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

    def test_another_thread_wakes_it_promptly(self) -> None:
        async def main() -> float:
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
        async def main() -> tuple[int, bool]:
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

    def test_sockets_and_subprocesses_through_the_run_loop(self) -> None:
        async def main() -> tuple[bytes, str]:
            loop = asyncio.get_running_loop()
            a, b = socket.socketpair()
            with a, b:
                a.setblocking(False)
                threading.Timer(0.05, b.send, (b"ping",)).start()
                data = await loop.sock_recv(a, 4)
            process = await asyncio.create_subprocess_exec("echo", "hello", stdout=asyncio.subprocess.PIPE)
            out, _ = await process.communicate()
            return data, out.decode().strip()

        self.assertEqual(run(main), (b"ping", "hello"))

    def test_a_signal_wakes_it(self) -> None:
        async def main() -> str:
            loop = asyncio.get_running_loop()
            got = loop.create_future()
            loop.add_signal_handler(signal.SIGUSR1, got.set_result, "SIGUSR1")
            threading.Timer(0.05, os.kill, (os.getpid(), signal.SIGUSR1)).start()
            async with asyncio.timeout(1):
                return await got

        self.assertEqual(run(main), "SIGUSR1")

    def test_keeps_running_during_a_tracking_loop(self) -> None:
        async def main() -> int:
            loop = asyncio.get_running_loop()
            ticks = 0

            async def count() -> None:
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.1)
                    ticks += 1

            counter = asyncio.create_task(count())
            closed = loop.create_future()

            def track() -> None:
                # As a live resize does: the run loop, in another mode, for a while.
                CFRunLoopRunInMode(TRACKING_MODE, 1.0, False)
                loop.call_soon(closed.set_result, ticks)

            after(0, track)
            during = await closed
            counter.cancel()
            return during

        # About 10 ticks of 100 ms while it was tracking for 1 s.
        self.assertGreaterEqual(run(main), 7)


if __name__ == "__main__":
    unittest.main()
