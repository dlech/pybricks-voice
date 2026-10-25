import contextlib
import functools
import queue
import sys
import threading
import unittest
from collections.abc import Callable
from typing import TYPE_CHECKING

assert not TYPE_CHECKING or sys.platform == "win32"

from winrt.microsoft.ui.dispatching import DispatcherQueueController
from winrt.microsoft.ui.xaml.hosting import WindowsXamlManager
from winrt.microsoft.ui.xaml.interop import (
    NotifyCollectionChangedAction,
    NotifyCollectionChangedEventArgs,
    NotifyCollectionChangedEventHandler,
)
from winrt.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import initialize
from winrt.runtime import ApartmentType, init_apartment
from winrt.system import Object
from winrt.windows.foundation import IStringable

from pybricks_voice.windows.observable_list import ObservableList
from pybricks_voice.windows.winrt_event import Event

_runtime = contextlib.ExitStack()
_xaml_work: queue.Queue[Callable[[], None] | None] = queue.Queue()


def _xaml_thread() -> None:
    """
    Runs _xaml_work on a thread that XAML is set up on: WinUI's
    NotifyCollectionChangedEventArgs can only be made on one, and XAML can't
    be set up again on another thread once it's closed, so it's one thread
    for all the tests.
    """
    init_apartment(ApartmentType.SINGLE_THREADED)
    controller = DispatcherQueueController.create_on_current_thread()
    xaml = WindowsXamlManager.initialize_for_current_thread()
    while (work := _xaml_work.get()) is not None:
        work()
    xaml.close()
    controller.shutdown_queue()


_worker = threading.Thread(target=_xaml_thread)


def setUpModule() -> None:
    _runtime.enter_context(initialize())
    _worker.start()


def tearDownModule() -> None:
    _xaml_work.put(None)
    _worker.join()
    _runtime.close()


def on_xaml_thread[T](test: Callable[[T], None]) -> Callable[[T], None]:
    """Runs a test on the XAML thread."""

    @functools.wraps(test)
    def run(self: T) -> None:
        done = threading.Event()
        failure: list[BaseException] = []

        def work() -> None:
            try:
                test(self)
            except BaseException as e:
                failure.append(e)
            finally:
                done.set()

        _xaml_work.put(work)
        done.wait()
        if failure:
            raise failure[0]

    return run


class Item(IStringable):
    def __init__(self, name: str) -> None:
        self.name = name

    def to_string(self) -> str:
        return self.name


class TestEvent(unittest.TestCase):
    def test_add_call_remove(self) -> None:
        event = Event[NotifyCollectionChangedEventHandler]()
        self.assertFalse(event)
        calls: list[str] = []
        first = event.add(lambda sender, args: calls.append("first"))
        second = event.add(lambda sender, args: calls.append("second"))
        self.assertNotEqual(first, second)
        self.assertTrue(event)
        event(Item("sender"), None)  # pyright: ignore[reportArgumentType]
        self.assertEqual(calls, ["first", "second"])
        event.remove(first)
        event.remove(first)  # unknown tokens are ignored
        event(Item("sender"), None)  # pyright: ignore[reportArgumentType]
        self.assertEqual(calls, ["first", "second", "second"])

    def test_remove_while_calling(self) -> None:
        event = Event[NotifyCollectionChangedEventHandler]()
        calls: list[str] = []
        token = event.add(lambda sender, args: (calls.append("first"), event.remove(token)))
        event.add(lambda sender, args: calls.append("second"))
        event(Item("sender"), None)  # pyright: ignore[reportArgumentType]
        event(Item("sender"), None)  # pyright: ignore[reportArgumentType]
        self.assertEqual(calls, ["first", "second", "second"])


class TestObservableList(unittest.TestCase):
    def setUp(self) -> None:
        # Made on the main thread; only raising the event needs XAML.
        self.items = ObservableList[Item]()
        self.changes: list[tuple[NotifyCollectionChangedAction, list[str], list[str], int, int]] = []
        self.token = self.items.add_collection_changed(self.changed)

    def changed(self, sender: Object, args: NotifyCollectionChangedEventArgs) -> None:
        self.assertIs(sender, self.items)

        def names(items) -> list[str]:
            return [items.get_at(i).as_(IStringable).to_string() for i in range(items.size)]

        new, old = names(args.new_items), names(args.old_items)
        self.changes.append((args.action, new, old, args.new_starting_index, args.old_starting_index))

    @on_xaml_thread
    def test_list(self) -> None:
        a, b, c = Item("a"), Item("b"), Item("c")
        self.items.append(a)
        self.items.append(c)
        self.items.insert(1, b)
        self.assertEqual(list(self.items), [a, b, c])
        self.assertIs(self.items.pop(0), a)
        del self.items[-1]
        self.assertEqual(len(self.items), 1)
        self.items.clear()
        A, R = NotifyCollectionChangedAction.ADD, NotifyCollectionChangedAction.REMOVE
        self.assertEqual(
            self.changes,
            [
                (A, ["a"], [], 0, -1),
                (A, ["c"], [], 1, -1),
                (A, ["b"], [], 1, -1),
                (R, [], ["a"], -1, 0),
                (R, [], ["c"], -1, 1),
                (NotifyCollectionChangedAction.RESET, [], [], -1, -1),
            ],
        )
        with self.assertRaises(IndexError):
            self.items.pop()

    @on_xaml_thread
    def test_bindable_vector(self) -> None:
        a, b = Item("a"), Item("b")
        self.items.insert_at(0, a)
        self.items.set_at(0, b)
        self.assertEqual(self.items.size, 1)
        self.assertIs(self.items.get_at(0), b)
        self.assertEqual(self.items.index_of(b), (True, 0))
        self.assertEqual(self.items.index_of(a), (False, 0))
        view, iterator = self.items.get_view(), self.items.first()
        self.assertIs(view.get_at(0), b)
        self.assertTrue(iterator.has_current)
        self.assertIs(iterator.current, b)
        self.assertFalse(iterator.move_next())
        self.items.remove_at_end()
        self.assertEqual(self.changes[1], (NotifyCollectionChangedAction.REPLACE, ["b"], ["a"], 0, 0))

    @on_xaml_thread
    def test_removed_handler(self) -> None:
        self.items.remove_collection_changed(self.token)
        self.items.append(Item("a"))
        self.assertEqual(self.changes, [])


if __name__ == "__main__":
    unittest.main()
