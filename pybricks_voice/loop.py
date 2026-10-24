"""asyncio for a WinUI 3 app, as a guest of the UI thread's DispatcherQueue.

WinUI's Application.start() pumps the UI thread, and so does every modal loop
Windows runs inside it: the one that moves or resizes the window while it is
dragged, a MessageBox, the system menu. An asyncio loop that owned the thread
would stop for each of those. This one never waits: each iteration is a
DispatcherQueue work item, so whatever pumps the thread runs asyncio too, and
coroutines, XAML and WinRT awaits all share the UI thread:

    def _on_launched(self, args):
        ...
        self.loop = DispatcherQueueEventLoop(self.window.dispatcher_queue)
        self.loop.start()
        self.main_task = self.loop.create_task(self.main())

I/O completions (sockets, pipes, subprocesses) are taken off the completion
port by a helper thread and handed over the same way. The proactor is
IocpProactor with the body of its _poll() split in two, as qasync does for Qt.
"""

# This is built from asyncio's private parts, which typeshed leaves out.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import _overlapped
import _winapi
import asyncio
import collections
import threading
from asyncio import events, windows_events
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from winrt.microsoft.ui.dispatching import DispatcherQueue

_Status = tuple[int, int, int, int]


class _HandoffProactor(windows_events.IocpProactor):
    """An IOCP proactor whose port is waited on by a thread of its own, which
    hands each completion to the loop's thread; the loop's thread never blocks."""

    def __init__(self, wake: Callable[[], None]) -> None:
        super().__init__()
        self._wake = wake
        self._handed: collections.deque[_Status] = collections.deque()
        self._thread: threading.Thread | None = None
        self._stop_requested = False

    def start_handoff(self) -> None:
        self._stop_requested = False
        self._thread = threading.Thread(target=self._wait_for_completions, name="iocp-handoff", daemon=True)
        self._thread.start()

    def stop_handoff(self) -> None:
        if self._thread is None:
            return
        self._stop_requested = True
        _overlapped.PostQueuedCompletionStatus(self._iocp, 0, 0, 0)
        self._thread.join()
        self._thread = None
        self._process_handed()  # what the helper took but the loop didn't get to

    def _wait_for_completions(self) -> None:
        while True:
            try:
                status = _overlapped.GetQueuedCompletionStatus(self._iocp, windows_events.INFINITE)
            except OSError:
                # _overlapped can't return a packet without an OVERLAPPED and
                # raises instead, with a stale error code. The only such packet
                # is the one stop_handoff() posts.
                if self._stop_requested:
                    return
                raise
            if status is not None:
                self._handed.append(status)
                self._wake()

    def _poll(self, timeout: float | None = None) -> None:
        if self._thread is None:
            # Before start() and while closing, the port is waited on here.
            super()._poll(timeout)
            return
        self._process_handed()
        for ov in self._unregistered:
            self._cache.pop(ov.address, None)
        self._unregistered.clear()

    def _process_handed(self) -> None:
        while self._handed:
            self._process(self._handed.popleft())

    def _process(self, status: _Status) -> None:
        # The body of IocpProactor._poll()'s loop, for one packet.
        err, transferred, key, address = status
        try:
            f, ov, obj, callback = self._cache.pop(address)
        except KeyError:
            if self._loop.get_debug():
                self._loop.call_exception_handler(
                    {
                        "message": "GetQueuedCompletionStatus() returned an unexpected event",
                        "status": f"err={err} transferred={transferred} key={key:#x} address={address:#x}",
                    }
                )
            # A pipe handle comes back in key, and has to be closed.
            if key not in (0, _overlapped.INVALID_HANDLE_VALUE):
                _winapi.CloseHandle(key)
            return
        if obj in self._stopped_serving:
            f.cancel()
        elif not f.done():  # not already read by _register(), or cancelled
            try:
                value = callback(transferred, key, ov)
            except OSError as e:
                f.set_exception(e)
            else:
                f.set_result(value)
            self._results.append(f)

    def close(self) -> None:
        self.stop_handoff()
        super().close()


class DispatcherQueueEventLoop(asyncio.ProactorEventLoop):
    """A ProactorEventLoop that runs as a guest of a DispatcherQueue.

    Create it and call start() on the queue's thread, and let the queue's owner
    pump the thread. start() returns at once; the loop runs until stop(), and
    is closed with close() after that. Since it never blocks the thread,
    run_forever(), run_until_complete() and asyncio.run() aren't available.
    A coroutine that blocks, including by entering a modal loop itself, stops
    asyncio and the UI together, as it would anywhere.
    """

    def __init__(self, dispatcher_queue: DispatcherQueue) -> None:
        self._dispatcher_queue = dispatcher_queue
        self._tick_pending = False
        self._in_tick = False
        self._guest_running = False
        self._timer = dispatcher_queue.create_timer()
        self._timer.is_repeating = False
        self._timer.add_tick(lambda sender, args: self._tick())
        self.iterations = 0
        super().__init__(_HandoffProactor(self._schedule_tick))

    def start(self) -> None:
        """Starts running on the DispatcherQueue, from its thread, and returns."""
        self._run_forever_setup()
        self._guest_running = True
        self._proactor.start_handoff()
        self._schedule_tick()

    def stop(self) -> None:
        super().stop()
        self._schedule_tick()

    def run_forever(self) -> None:
        raise RuntimeError(f"{type(self).__name__} runs on its DispatcherQueue: start() it instead")

    def run_until_complete(self, future: Any) -> Any:
        raise RuntimeError(f"{type(self).__name__} runs on its DispatcherQueue: start() it and create a task")

    def _schedule_tick(self) -> None:
        # From any thread. A spare tick is harmless and a lost one is not, so
        # the flag only saves work, and needs no lock: whoever adds work does so
        # before looking at it.
        if self._tick_pending:
            return
        self._tick_pending = True
        if not self._dispatcher_queue.try_enqueue(self._tick):
            self._tick_pending = False  # the queue is shutting down

    def _tick(self) -> None:
        self._tick_pending = False
        if not self._guest_running:
            return
        if self._in_tick:
            # A modal loop entered from inside a callback pumps this work item
            # too, and _run_once() isn't reentrant. The outer iteration
            # schedules the next one when it is done.
            return
        self._in_tick = True
        try:
            self._timer.stop()
            # Each work item may enter Python with a thread state of its own,
            # which wouldn't know which loop is running.
            events._set_running_loop(self)
            self._run_once()
            self.iterations += 1
        finally:
            self._in_tick = False

        if self._stopping:
            self._guest_running = False
            self._proactor.stop_handoff()
            self._run_forever_cleanup()
        elif self._ready:
            self._schedule_tick()
        elif self._scheduled:
            delay = max(0.0, self._scheduled[0].when() - self.time())
            self._timer.interval = timedelta(seconds=delay)
            self._timer.start()

    # Work that arrives between iterations, from a XAML event handler or from
    # another thread, has to ask for one: nothing else would.

    def _call_soon(self, callback: Any, args: Any, context: Any) -> Any:
        handle = super()._call_soon(callback, args, context)
        if self._guest_running and not self._in_tick:
            self._schedule_tick()
        return handle

    def call_at(self, when: float, callback: Any, *args: Any, context: Any = None) -> Any:
        handle = super().call_at(when, callback, *args, context=context)
        if self._guest_running and not self._in_tick:
            self._schedule_tick()
        return handle

    def _write_to_self(self) -> None:
        # call_soon_threadsafe(): wake through the queue, rather than through
        # the self-pipe, the port and the helper thread.
        self._schedule_tick()
