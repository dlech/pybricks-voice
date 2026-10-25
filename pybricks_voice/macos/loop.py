"""
asyncio for an AppKit app, as a guest of the main thread's CFRunLoop.

NSApplication's run() pumps the main thread, and so do the loops AppKit runs
inside it: the one that tracks a window's live resize, a menu, a modal panel.
An asyncio loop that owned the thread would stop for each of those, and
couldn't dispatch AppKit's events at all. This one never waits: each
iteration is the perform callback of a run loop source in the common modes,
which AppKit's tracking and modal modes are among, so whatever pumps the
thread runs asyncio too:

    def applicationDidFinishLaunching_(self, notification):
        ...
        self.loop = CFRunLoopEventLoop()
        self.loop.start()
        self.main_task = self.loop.create_task(self.main())

Each file descriptor the selector watches (sockets, pipes, the self-pipe that
signals write to) is a CFFileDescriptor in the same modes, as in
objc_tools.selectors' CFRunLoopSelector; a delay is a CFRunLoopTimer.
"""

# This is built from asyncio's private parts, which typeshed leaves out.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import asyncio
import selectors
import sys
from asyncio import events
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

assert not TYPE_CHECKING or sys.platform == "darwin"

from .corefoundation import (
    CFAbsoluteTimeGetCurrent,
    CFFileDescriptorCallBack,
    CFFileDescriptorCreate,
    CFFileDescriptorCreateRunLoopSource,
    CFFileDescriptorEnableCallBacks,
    CFFileDescriptorGetNativeDescriptor,
    CFFileDescriptorInvalidate,
    CFRelease,
    CFRunLoopAddSource,
    CFRunLoopAddTimer,
    CFRunLoopGetMain,
    CFRunLoopSourceContext,
    CFRunLoopSourceCreate,
    CFRunLoopSourceInvalidate,
    CFRunLoopSourcePerform,
    CFRunLoopSourceSignal,
    CFRunLoopTimerCallBack,
    CFRunLoopTimerCreate,
    CFRunLoopTimerInvalidate,
    CFRunLoopTimerSetNextFireDate,
    CFRunLoopWakeUp,
    kCFFileDescriptorReadCallBack,
    kCFFileDescriptorWriteCallBack,
    kCFRunLoopCommonModes,
)

# A timer that never fires on its own: CFRunLoopTimerSetNextFireDate brings it
# forward. A one-shot timer would be invalidated after firing.
_NEVER = 1e12  # seconds, about 32,000 years


class _RunLoopSelector(selectors._BaseSelectorImpl):
    """
    A selector whose file descriptors are watched by a CFRunLoop rather than
    waited on: each is a CFFileDescriptor whose callback notes it as ready and
    wakes the loop. select() never waits, and returns what has been noted.

    CFFileDescriptor callbacks fire once, then need enabling again; rearm()
    does that after each iteration, once the handlers have read or written,
    so a descriptor that stays ready fires again, as select() would report it.
    """

    def __init__(self, run_loop: int, wake: Callable[[], None]) -> None:
        super().__init__()
        self._run_loop = run_loop
        self._wake = wake
        self._callback = CFFileDescriptorCallBack(self._fired)  # kept: CF holds only the pointer
        self._descriptors: dict[int, tuple[int, int, int]] = {}  # fd -> CFFileDescriptor, its source, flags
        self._pending: dict[int, int] = {}  # fd -> selectors.EVENT_*

    def _fired(self, descriptor: int, callback_types: int, info: int | None) -> None:
        fd = CFFileDescriptorGetNativeDescriptor(descriptor)
        mask = 0
        if callback_types & kCFFileDescriptorReadCallBack:
            mask |= selectors.EVENT_READ
        if callback_types & kCFFileDescriptorWriteCallBack:
            mask |= selectors.EVENT_WRITE
        self._pending[fd] = self._pending.get(fd, 0) | mask
        self._wake()

    def register(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        key = super().register(fileobj, events, data)
        descriptor = CFFileDescriptorCreate(None, key.fd, False, self._callback, None)
        if not descriptor:
            super().unregister(fileobj)
            raise OSError(f"CFFileDescriptorCreate failed for fd {key.fd}")
        source = CFFileDescriptorCreateRunLoopSource(None, descriptor, 0)
        CFRunLoopAddSource(self._run_loop, source, kCFRunLoopCommonModes)
        flags = (kCFFileDescriptorReadCallBack if events & selectors.EVENT_READ else 0) | (
            kCFFileDescriptorWriteCallBack if events & selectors.EVENT_WRITE else 0
        )
        self._descriptors[key.fd] = (descriptor, source, flags)
        CFFileDescriptorEnableCallBacks(descriptor, flags)
        return key

    def unregister(self, fileobj: Any) -> selectors.SelectorKey:
        key = super().unregister(fileobj)
        descriptor, source, _ = self._descriptors.pop(key.fd)
        # Invalidating removes the source from the run loop too.
        CFFileDescriptorInvalidate(descriptor)
        CFRelease(source)
        CFRelease(descriptor)
        self._pending.pop(key.fd, None)
        return key

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        ready = []
        pending, self._pending = self._pending, {}
        for fd, mask in pending.items():
            key = self._fd_to_key.get(fd)
            if key is not None and mask & key.events:
                ready.append((key, mask & key.events))
        return ready

    def rearm(self) -> None:
        for descriptor, _, flags in self._descriptors.values():
            CFFileDescriptorEnableCallBacks(descriptor, flags)

    def close(self) -> None:
        for descriptor, source, _ in self._descriptors.values():
            CFFileDescriptorInvalidate(descriptor)
            CFRelease(source)
            CFRelease(descriptor)
        self._descriptors.clear()
        super().close()


class CFRunLoopEventLoop(asyncio.SelectorEventLoop):
    """
    A SelectorEventLoop that runs as a guest of a CFRunLoop, the main
    thread's unless another is given.

    Create it and call start() on the run loop's thread, and let the run
    loop's owner pump the thread. start() returns at once; the loop runs until
    stop(), and is closed with close() after that. Since it never blocks the
    thread, run_forever(), run_until_complete() and asyncio.run() aren't
    available. A coroutine that blocks, including by entering a modal loop
    itself, stops asyncio and the UI together, as it would anywhere.
    """

    def __init__(self, run_loop: int | None = None) -> None:
        self._run_loop: int = run_loop or CFRunLoopGetMain()
        self._tick_pending = False
        self._in_tick = False
        self._guest_running = False
        self.iterations = 0

        # Each iteration: a version 0 source, signaled from any thread.
        self._perform = CFRunLoopSourcePerform(lambda info: self._tick())
        self._context = CFRunLoopSourceContext(version=0, perform=self._perform)
        self._source: int | None = CFRunLoopSourceCreate(None, 0, self._context)
        CFRunLoopAddSource(self._run_loop, self._source, kCFRunLoopCommonModes)
        # The next delayed callback.
        self._fire = CFRunLoopTimerCallBack(lambda timer, info: self._tick())
        self._timer: int | None = CFRunLoopTimerCreate(
            None, CFAbsoluteTimeGetCurrent() + _NEVER, _NEVER, 0, 0, self._fire, None
        )
        CFRunLoopAddTimer(self._run_loop, self._timer, kCFRunLoopCommonModes)

        super().__init__(_RunLoopSelector(self._run_loop, self._schedule_tick))

    def start(self) -> None:
        """Starts running on the CFRunLoop, from its thread, and returns."""
        self._run_forever_setup()
        self._guest_running = True
        self._schedule_tick()

    def stop(self) -> None:
        super().stop()
        self._schedule_tick()

    def run_forever(self) -> None:
        raise RuntimeError(f"{type(self).__name__} runs on its CFRunLoop: start() it instead")

    def run_until_complete(self, future: Any) -> Any:
        raise RuntimeError(f"{type(self).__name__} runs on its CFRunLoop: start() it and create a task")

    def close(self) -> None:
        super().close()
        if self._source:
            CFRunLoopSourceInvalidate(self._source)
            CFRelease(self._source)
            self._source = None
        if self._timer:
            CFRunLoopTimerInvalidate(self._timer)
            CFRelease(self._timer)
            self._timer = None

    def _schedule_tick(self) -> None:
        # From any thread. A spare tick is harmless and a lost one is not, so
        # the flag only saves work, and needs no lock: whoever adds work does so
        # before looking at it.
        if self._tick_pending or not self._source:
            return
        self._tick_pending = True
        CFRunLoopSourceSignal(self._source)
        CFRunLoopWakeUp(self._run_loop)

    def _tick(self) -> None:
        self._tick_pending = False
        if not self._guest_running:
            return
        if self._in_tick:
            # A modal loop entered from inside a callback runs this source
            # too, and _run_once() isn't reentrant. The outer iteration
            # schedules the next one when it is done.
            return
        self._in_tick = True
        try:
            CFRunLoopTimerSetNextFireDate(self._timer, CFAbsoluteTimeGetCurrent() + _NEVER)
            events._set_running_loop(self)
            self._run_once()
            self.iterations += 1
        finally:
            self._in_tick = False
            self._selector.rearm()

        if self._stopping:
            self._guest_running = False
            self._run_forever_cleanup()
        elif self._ready:
            self._schedule_tick()
        elif self._scheduled:
            delay = max(0.0, self._scheduled[0].when() - self.time())
            CFRunLoopTimerSetNextFireDate(self._timer, CFAbsoluteTimeGetCurrent() + delay)

    # Work that arrives between iterations, from an AppKit action or from
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
        # call_soon_threadsafe(): wake through the source rather than the
        # self-pipe. Signals still write to the pipe, which wakes it too.
        self._schedule_tick()
