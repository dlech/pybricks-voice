"""
An event that Python implements, for a WinRT interface that has one.

WinRT projects an event as add_<event>() and remove_<event>(): add_ takes a
handler and returns a token, remove_ takes the token back. A Python class
that implements such an interface, INotifyCollectionChanged say, has to keep
its handlers and make up tokens itself. C++/WinRT has winrt::event for that,
and C# has its event keyword; this is the same for Python:

    class Thing(INotifyCollectionChanged):
        def __init__(self) -> None:
            self.collection_changed = Event[NotifyCollectionChangedEventHandler]()

        @override
        def add_collection_changed(self, handler: NotifyCollectionChangedEventHandler) -> EventRegistrationToken:
            return self.collection_changed.add(handler)

        @override
        def remove_collection_changed(self, token: EventRegistrationToken) -> None:
            self.collection_changed.remove(token)

and self.collection_changed(self, args) calls the handlers.

It is the kind of thing PyWinRT could ship, next to the delegate types.
"""

import itertools
import typing
from collections.abc import Callable

from winrt.windows.foundation import EventRegistrationToken


class Event[H: Callable[..., object]]:
    """The handlers of an event, H being its delegate type."""

    def __init__(self) -> None:
        self._handlers: dict[int, H] = {}
        self._tokens = itertools.count(1)

    def add(self, handler: H) -> EventRegistrationToken:
        """Adds a handler, for add_<event>(), and returns its token."""
        token = next(self._tokens)
        self._handlers[token] = handler
        # PyWinRT's EventRegistrationToken can't be made in Python, but WinRT
        # takes any int as the token an implementation returns.
        return token  # pyright: ignore[reportReturnType]

    def remove(self, token: EventRegistrationToken) -> None:
        """Removes the handler of a token, for remove_<event>(). An unknown token is ignored, as in WinRT."""
        self._handlers.pop(int(token), None)

    def __bool__(self) -> bool:
        """Whether there are handlers: when there are none, the event's args needn't be made."""
        return bool(self._handlers)

    if typing.TYPE_CHECKING:
        __call__: H
    else:

        def __call__(self, *args):
            """Calls the handlers, those added as it started, in the order added."""
            for handler in list(self._handlers.values()):
                handler(*args)
