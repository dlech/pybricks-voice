"""
Spike: a XAML list fed from Python, as the chat could be.

An ItemsRepeater with a DataTemplate for its bubbles, whose ItemsSource is a
Python list: BindableList implements WinUI's non-generic collection
interfaces, IBindableVector to be read and INotifyCollectionChanged to say
what changed, which is what a C# ObservableCollection shows XAML. Its items
are Python objects too, that implement IStringable to be WinRT objects. Each
bubble is filled in from its item as the repeater prepares it
(ElementPrepared). Every call WinRT makes into the Python objects is logged,
to see what XAML asks of them.

    uv run python spikes/chat_list.py [--seconds 8] [--screenshot out/chat_list.png]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import sys
from typing import override

from winrt.microsoft.ui.xaml import Application, FrameworkElement, HorizontalAlignment, LaunchActivatedEventArgs, Window
from winrt.microsoft.ui.xaml.controls import (
    ItemsRepeater,
    ItemsRepeaterElementPreparedEventArgs,
    TextBlock,
    XamlControlsResources,
)
from winrt.microsoft.ui.xaml.interop import (
    IBindableIterator,
    IBindableVector,
    IBindableVectorView,
    INotifyCollectionChanged,
    NotifyCollectionChangedAction,
    NotifyCollectionChangedEventArgs,
    NotifyCollectionChangedEventHandler,
)
from winrt.microsoft.ui.xaml.markup import IXamlMetadataProvider, IXamlType, XamlReader, XmlnsDefinition
from winrt.microsoft.ui.xaml.xamltypeinfo import XamlControlsXamlMetaDataProvider
from winrt.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import InitializeOptions, initialize
from winrt.runtime import ApartmentType, init_apartment
from winrt.system import Array, Object
from winrt.windows.foundation import EventRegistrationToken, IStringable
from winrt.windows.ui.xaml.interop import TypeKind, TypeName

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pybricks_voice.loop import DispatcherQueueEventLoop
from pybricks_voice.screenshot import save_window

log = logging.getLogger("chat_list")

XAML = """
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml">
    <ScrollViewer Padding="16">
        <ItemsRepeater Name="list">
            <ItemsRepeater.Layout><StackLayout Spacing="8"/></ItemsRepeater.Layout>
            <ItemsRepeater.ItemTemplate>
                <DataTemplate>
                    <Border Name="bubble" MaxWidth="290" Padding="12,8" CornerRadius="16"
                            Background="{ThemeResource AccentFillColorDefaultBrush}">
                        <TextBlock Name="text" TextWrapping="Wrap"/>
                    </Border>
                </DataTemplate>
            </ItemsRepeater.ItemTemplate>
        </ItemsRepeater>
    </ScrollViewer>
</Window>
"""


class Message(IStringable):
    """A chat message: a Python object, which IStringable makes a WinRT one."""

    def __init__(self, text: str, mine: bool) -> None:
        self.text = text
        self.mine = mine

    @override
    def to_string(self) -> str:
        return self.text


class _Iterator(IBindableIterator):
    def __init__(self, items: list[object]) -> None:
        self._items = items
        self._index = 0

    @property
    @override
    def current(self) -> Object:
        log.debug("iterator.current %d", self._index)
        return self._items[self._index]  # pyright: ignore[reportReturnType]

    @property
    @override
    def has_current(self) -> bool:
        return self._index < len(self._items)

    @override
    def move_next(self) -> bool:
        self._index += 1
        return self.has_current


class BindableList(IBindableVector, INotifyCollectionChanged):
    """A Python list XAML can read, and that tells it when it changes."""

    def __init__(self) -> None:
        self._items: list[object] = []
        self._handlers: dict[int, NotifyCollectionChangedEventHandler] = {}
        self._next_token = 1

    # Python's side
    def add(self, item: Object) -> None:
        self._items.append(item)
        self._changed(NotifyCollectionChangedAction.ADD, [item], len(self._items) - 1)

    def _changed(self, action: NotifyCollectionChangedAction, items: list[object], index: int) -> None:
        new = BindableList()
        new._items = items
        args = NotifyCollectionChangedEventArgs(action, new, new, index, -1)
        for handler in list(self._handlers.values()):
            handler(self, args)

    # IBindableIterable
    @override
    def first(self) -> IBindableIterator:
        log.debug("first")
        return _Iterator(self._items)

    # IBindableVector
    @property
    @override
    def size(self) -> int:
        log.debug("size -> %d", len(self._items))
        return len(self._items)

    @override
    def get_at(self, index) -> Object:
        log.debug("get_at %d", index)
        return self._items[index]  # pyright: ignore[reportReturnType]

    @override
    def get_view(self) -> IBindableVectorView:
        log.debug("get_view")
        raise NotImplementedError

    @override
    def index_of(self, value: Object) -> tuple[bool, int]:
        log.debug("index_of")
        for i, item in enumerate(self._items):
            if item is value:
                return True, i
        return False, 0

    @override
    def set_at(self, index, value: Object) -> None:
        raise NotImplementedError

    @override
    def insert_at(self, index, value: Object) -> None:
        raise NotImplementedError

    @override
    def remove_at(self, index) -> None:
        raise NotImplementedError

    @override
    def append(self, value: Object) -> None:
        self.add(value)

    @override
    def remove_at_end(self) -> None:
        raise NotImplementedError

    @override
    def clear(self) -> None:
        raise NotImplementedError

    # INotifyCollectionChanged
    @override
    def add_collection_changed(self, handler: NotifyCollectionChangedEventHandler) -> EventRegistrationToken:
        token = self._next_token
        self._next_token += 1
        self._handlers[token] = handler
        log.debug("add_collection_changed -> %d", token)
        return token  # pyright: ignore[reportReturnType]  (only WinRT makes tokens, says PyWinRT)

    @override
    def remove_collection_changed(self, token: EventRegistrationToken) -> None:
        log.debug("remove_collection_changed %d", token)
        self._handlers.pop(int(token), None)


class App(Application, IXamlMetadataProvider):
    def __init__(self, options: argparse.Namespace) -> None:
        super().__init__()
        self._provider = XamlControlsXamlMetaDataProvider()
        self.options = options

    @override
    def _on_launched(self, args: LaunchActivatedEventArgs) -> None:
        self.resources.merged_dictionaries.append(XamlControlsResources())
        self.window = XamlReader.load(XAML).as_(Window)
        self.window.app_window.resize((700, 800))
        content = self.window.content.as_(FrameworkElement)
        self.list = content.find_name("list").as_(ItemsRepeater)
        self.messages = BindableList()
        self.list.add_element_prepared(self.on_element_prepared)
        self.list.items_source = self.messages
        self.loop = DispatcherQueueEventLoop(self.window.dispatcher_queue)
        self.loop.start()
        self.task = self.loop.create_task(self.main())
        self.task.add_done_callback(lambda t: (t.cancelled() or t.result(), self.loop.stop(), self.exit()))
        self.window.activate()

    def on_element_prepared(self, sender: Object, args: ItemsRepeaterElementPreparedEventArgs) -> None:
        item = self.messages.get_at(args.index)
        via_view = self.list.items_source_view.get_at(args.index)
        log.debug(
            "element_prepared %d: %r; via items_source_view: %r, same: %s", args.index, item, via_view, via_view is item
        )
        log.debug("find_name on the template root: %r", args.element.as_(FrameworkElement).find_name("text"))
        element = args.element.as_(FrameworkElement)
        assert isinstance(item, Message), type(item)
        element.find_name("text").as_(TextBlock).text = item.text
        element.horizontal_alignment = HorizontalAlignment.RIGHT if item.mine else HorizontalAlignment.LEFT

    async def main(self) -> None:
        lines = ["forward 30 cm", "Forward 30 cm.", "nice job", "Thanks! 😊", "drive in a square"]
        for i, line in enumerate(lines):
            self.messages.add(Message(line, mine=i % 2 == 0))
            await asyncio.sleep(0.5)
        await asyncio.sleep(1)
        log.info("items_source_view count: %d", self.list.items_source_view.count)
        if self.options.screenshot:
            async with asyncio.timeout(2):
                await save_window(self.window, self.options.screenshot)

    @override
    def get_xaml_type(self, type: TypeName | tuple[str, TypeKind]) -> IXamlType:
        return self._provider.get_xaml_type(type)

    @override
    def get_xaml_type_by_full_name(self, full_name: str) -> IXamlType:
        return self._provider.get_xaml_type_by_full_name(full_name)

    @override
    def get_xmlns_definitions(self) -> Array[XmlnsDefinition]:
        return self._provider.get_xmlns_definitions()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", type=pathlib.Path, default=pathlib.Path("out/chat_list.png"))
    options = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    init_apartment(ApartmentType.SINGLE_THREADED)
    apps: list[App] = []
    with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
        Application.start(lambda _: apps.append(App(options)))
        for app in apps:
            app.loop.close()


if __name__ == "__main__":
    main()
