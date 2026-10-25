"""
A Python list that XAML lists can show, and that tells them when it changes.

A C# app gives an ItemsRepeater or a ListView an ObservableCollection. What
XAML reads from one is two of WinUI's non-generic interfaces: IBindableVector,
to read the items, and INotifyCollectionChanged, to hear about changes. They
take no type parameters, so Python can implement them, which PyWinRT can't do
for IObservableVector<T>. Its items must be WinRT objects: a Python object is
one if its class implements any WinRT interface, IStringable say, and XAML
hands back the same Python object.

It is the kind of thing PyWinRT could ship.
"""

from __future__ import annotations

import operator
import sys
import typing
from collections.abc import Iterator
from typing import TYPE_CHECKING, override

assert not TYPE_CHECKING or sys.platform == "win32"

from winrt.microsoft.ui.xaml.interop import (
    IBindableIterator,
    IBindableVector,
    IBindableVectorView,
    INotifyCollectionChanged,
    NotifyCollectionChangedAction,
    NotifyCollectionChangedEventArgs,
    NotifyCollectionChangedEventHandler,
)
from winrt.system import Object
from winrt.windows.foundation import EventRegistrationToken

from .winrt_event import Event


class ObservableList[T: Object](IBindableVector, INotifyCollectionChanged):
    """
    A list for an ItemsSource. Python uses it as a list: append, insert, del,
    pop, clear, indexing and iterating; each change tells XAML. Call it on the
    UI thread, as XAML's own collections must be.
    """

    def __init__(self, items: list[T] | None = None) -> None:
        self._items: list[T] = list(items or [])
        self.collection_changed = Event[NotifyCollectionChangedEventHandler]()

    # ----- Python's side -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __delitem__(self, index: int) -> None:
        self.pop(index)

    # Also IBindableVector.Append, which takes any Object; XAML only reads a
    # list it shows, and Python's callers are typed to its items.
    @override
    def append(self, value: T) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.insert(len(self._items), value)

    def insert(self, index: int, value: T) -> None:
        index = min(max(index if index >= 0 else len(self._items) + index, 0), len(self._items))
        self._items.insert(index, value)
        self._changed(NotifyCollectionChangedAction.ADD, new=[value], new_index=index)

    def pop(self, index: int = -1) -> T:
        index = range(len(self._items))[index]  # IndexError as a list's
        value = self._items.pop(index)
        self._changed(NotifyCollectionChangedAction.REMOVE, old=[value], old_index=index)
        return value

    @override
    def clear(self) -> None:  # also IBindableVector.Clear
        self._items.clear()
        self._changed(NotifyCollectionChangedAction.RESET)

    def _changed(
        self,
        action: NotifyCollectionChangedAction,
        new: list[T] | None = None,
        old: list[T] | None = None,
        new_index: int = -1,
        old_index: int = -1,
    ) -> None:
        if self.collection_changed:
            self.collection_changed(
                self,
                NotifyCollectionChangedEventArgs(
                    action, ObservableList(new or []), ObservableList(old or []), new_index, old_index
                ),
            )

    # ----- IBindableVector, for XAML -------------------------------------------

    @property
    @override
    def size(self) -> int:
        return len(self._items)

    @override
    def get_at(self, index) -> T:
        return self._items[operator.index(index)]

    @override
    def get_view(self) -> IBindableVectorView:
        return _View(list(self._items))

    @override
    def index_of(self, value: Object) -> tuple[bool, int]:
        for i, item in enumerate(self._items):
            if item is value:
                return True, i
        return False, 0

    @override
    def set_at(self, index, value: Object) -> None:
        index = operator.index(index)
        old, self._items[index] = self._items[index], typing.cast(T, value)
        self._changed(NotifyCollectionChangedAction.REPLACE, [self._items[index]], [old], index, index)

    @override
    def insert_at(self, index, value: Object) -> None:
        self.insert(operator.index(index), typing.cast(T, value))

    @override
    def remove_at(self, index) -> None:
        self.pop(operator.index(index))

    @override
    def remove_at_end(self) -> None:
        self.pop()

    @override
    def first(self) -> IBindableIterator:
        return _Iterator(list(self._items))

    # ----- INotifyCollectionChanged, for XAML -----------------------------------

    @override
    def add_collection_changed(self, handler: NotifyCollectionChangedEventHandler) -> EventRegistrationToken:
        return self.collection_changed.add(handler)

    @override
    def remove_collection_changed(self, token: EventRegistrationToken) -> None:
        self.collection_changed.remove(token)


class _View(IBindableVectorView):
    def __init__(self, items: list) -> None:
        self._items = items

    @property
    @override
    def size(self) -> int:
        return len(self._items)

    @override
    def get_at(self, index) -> Object:
        return self._items[operator.index(index)]

    @override
    def index_of(self, value: Object) -> tuple[bool, int]:
        return next(((True, i) for i, item in enumerate(self._items) if item is value), (False, 0))

    @override
    def first(self) -> IBindableIterator:
        return _Iterator(self._items)


class _Iterator(IBindableIterator):
    def __init__(self, items: list) -> None:
        self._items = items
        self._index = 0

    @property
    @override
    def current(self) -> Object:
        return self._items[self._index]

    @property
    @override
    def has_current(self) -> bool:
        return self._index < len(self._items)

    @override
    def move_next(self) -> bool:
        self._index += 1
        return self.has_current
