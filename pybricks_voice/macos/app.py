"""
The app on macOS: core.py's Core in an AppKit window, built in code with
rubicon-objc.

NSApplication's run() owns the main thread, and asyncio runs as its guest
(see loop.py); the 3D view draws into a CAMetalLayer (see scene_view.py).
Whisper and the microphone come later: typed commands work meanwhile.

The AppKit calls aren't type checked yet: rubicon-objc has no types for the
frameworks, which generated bindings will bring.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import signal
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, override

assert not TYPE_CHECKING or sys.platform == "darwin"

import numpy as np
from rubicon.objc import (
    SEL,
    NSMakePoint,
    NSMakeRect,
    NSObject,
    NSRect,
    ObjCClass,
    objc_const,
    objc_method,
    objc_property,
)
from rubicon.objc.runtime import load_library

from ..core import Core, parse_args
from ..speech import Microphone
from ..whisper import Transcriber
from .corefoundation import CFRunLoopGetMain, CFRunLoopRunInMode, kCFRunLoopDefaultMode
from .loop import CFRunLoopEventLoop
from .scene_view import SceneView

log = logging.getLogger(__name__)

appkit = load_library("AppKit")
NSApplication = ObjCClass("NSApplication")
NSBezierPath = ObjCClass("NSBezierPath")
NSButton = ObjCClass("NSButton")
NSColor = ObjCClass("NSColor")
NSEvent = ObjCClass("NSEvent")
NSFont = ObjCClass("NSFont")
NSMenu = ObjCClass("NSMenu")
NSMenuItem = ObjCClass("NSMenuItem")
NSNotificationCenter = ObjCClass("NSNotificationCenter")
NSProgressIndicator = ObjCClass("NSProgressIndicator")
NSScrollView = ObjCClass("NSScrollView")
NSStackView = ObjCClass("NSStackView")
NSSwitch = ObjCClass("NSSwitch")
NSTextField = ObjCClass("NSTextField")
NSView = ObjCClass("NSView")
NSWindow = ObjCClass("NSWindow")

# From the SDK's headers, by hand until generated bindings have them.
NSWindowStyleMaskTitled, NSWindowStyleMaskClosable = 1 << 0, 1 << 1
NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskResizable = 1 << 2, 1 << 3
NSBackingStoreBuffered = 2
NSApplicationActivationPolicyRegular = 0
NSTerminateCancel = 0
NSViewMinXMargin, NSViewWidthSizable, NSViewHeightSizable = 1, 2, 16
NSUserInterfaceLayoutOrientationHorizontal, NSUserInterfaceLayoutOrientationVertical = 0, 1
NSStackViewDistributionFill = 0
NSLayoutAttributeLeading, NSLayoutAttributeCenterY = 5, 10
NSLayoutPriorityDefaultLow = 250
NSControlStateValueOn = 1
NSEventTypeApplicationDefined = 15
NSLineBreakByTruncatingTail = 4
NSFontWeightRegular, NSFontWeightSemibold = 0.0, 0.3
NSViewBoundsDidChangeNotification = objc_const(appkit, "NSViewBoundsDidChangeNotification")
NSViewFrameDidChangeNotification = objc_const(appkit, "NSViewFrameDidChangeNotification")

SIDE_WIDTH = 380  # the controls and the chat, to the right of the 3D view
MAX_BUBBLE_WIDTH = 290
LEVEL_MIN_DB, LEVEL_MAX_DB = -70.0, -10.0


class NoMicrophone:
    """Until the macOS microphone arrives: never open."""

    rate = 16000
    is_open = False

    async def start(self) -> None:
        raise RuntimeError("no microphone on macOS yet")

    def stop(self) -> None:
        pass


class PybricksVoiceFlippedView(NSView):  # an Objective-C class: its name is global
    """A view whose y grows downward, so a scroll view's content starts at the top."""

    @objc_method
    def isFlipped(self) -> bool:
        return True


class PybricksVoiceBubbleView(NSView):
    """
    A message in the chat: a rounded rectangle whose corner nearest its
    sender is sharper, as on Windows. Drawn rather than a layer's
    cornerRadius, which rounds every corner alike; and drawing resolves the
    system colors each time, so they follow light and dark mode.
    """

    mine = objc_property(object)

    @objc_method
    def drawRect_(self, dirty: NSRect) -> None:
        bounds = self.bounds
        # Half a point in, so a 1 point border is crisp. y grows upward here.
        left, bottom = bounds.origin.x + 0.5, bounds.origin.y + 0.5
        right, top = left + bounds.size.width - 1, bottom + bounds.size.height - 1
        # As XAML's CornerRadius: top left, top right, bottom right, bottom left.
        top_left, top_right, bottom_right, bottom_left = (16, 16, 4, 16) if self.mine else (16, 16, 16, 4)
        path = NSBezierPath.bezierPath()
        path.moveToPoint(NSMakePoint(left + bottom_left, bottom))
        for corner, toward, radius in (
            ((right, bottom), (right, top), bottom_right),
            ((right, top), (left, top), top_right),
            ((left, top), (left, bottom), top_left),
            ((left, bottom), (right, bottom), bottom_left),
        ):
            path.appendBezierPathWithArcFromPoint(NSMakePoint(*corner), toPoint=NSMakePoint(*toward), radius=radius)
        path.closePath()
        if self.mine:
            NSColor.controlAccentColor.setFill()
            path.fill()
        else:
            NSColor.controlBackgroundColor.setFill()
            path.fill()
            NSColor.separatorColor.setStroke()
            path.lineWidth = 1
            path.stroke()


class PybricksVoiceController(NSObject):
    """The app delegate, the window's delegate, and the target of every control."""

    app = objc_property(object)

    @objc_method
    def applicationDidFinishLaunching_(self, notification) -> None:
        self.app.launched()

    @objc_method
    def applicationShouldTerminate_(self, sender) -> int:
        # Quit closes the window, which shuts down; NSApplication's run()
        # returns when that's done, and main() then returns.
        self.app.window.performClose(None)
        return NSTerminateCancel

    @objc_method
    def windowWillClose_(self, notification) -> None:
        self.app.shutdown()

    @objc_method
    def chatChanged_(self, notification) -> None:
        self.app.on_chat_changed()

    @objc_method
    def onGo_(self, sender) -> None:
        self.app.submit_typed()

    @objc_method
    def onStop_(self, sender) -> None:
        self.app.stop()

    @objc_method
    def onReset_(self, sender) -> None:
        self.app.reset()

    @objc_method
    def onCamera_(self, sender) -> None:
        self.app.toggle_camera()

    @objc_method
    def onListen_(self, sender) -> None:
        self.app.set_listening(sender.state == NSControlStateValueOn)


# ----- building views ----------------------------------------------------------------


def _label(text: str = "", size: float = 13, mono: bool = False, wraps: bool = False, secondary: bool = False) -> Any:
    label = NSTextField.wrappingLabelWithString(text) if wraps else NSTextField.labelWithString(text)
    if mono:
        label.font = NSFont.monospacedSystemFontOfSize(size, weight=NSFontWeightRegular)
    else:
        label.font = NSFont.systemFontOfSize(size)
    if secondary:
        label.textColor = NSColor.secondaryLabelColor
    return label


def _stack(views: list[Any], vertical: bool, spacing: float) -> Any:
    stack = NSStackView.stackViewWithViews(views)
    stack.orientation = (
        NSUserInterfaceLayoutOrientationVertical if vertical else NSUserInterfaceLayoutOrientationHorizontal
    )
    stack.spacing = spacing
    stack.alignment = NSLayoutAttributeLeading if vertical else NSLayoutAttributeCenterY
    return stack


def _pin(
    view: Any,
    to: Any,
    leading: float | None = None,
    trailing: float | None = None,
    top: float | None = None,
    bottom: float | None = None,
) -> None:
    """Auto Layout: view's edges at these distances inside to's."""
    view.translatesAutoresizingMaskIntoConstraints = False
    if leading is not None:
        view.leadingAnchor.constraintEqualToAnchor(to.leadingAnchor, constant=leading).active = True
    if trailing is not None:
        view.trailingAnchor.constraintEqualToAnchor(to.trailingAnchor, constant=-trailing).active = True
    if top is not None:
        view.topAnchor.constraintEqualToAnchor(to.topAnchor, constant=top).active = True
    if bottom is not None:
        view.bottomAnchor.constraintEqualToAnchor(to.bottomAnchor, constant=-bottom).active = True


def _rounded(view: Any, radius: float, color: Any) -> Any:
    view.wantsLayer = True
    view.layer.cornerRadius = radius
    view.layer.backgroundColor = color.CGColor
    return view


class MacApp(Core):
    """The window, its controls, and the chat, around Core."""

    WHISPER_ON = "on the Neural Engine"

    def __init__(self, options: argparse.Namespace) -> None:
        super().__init__(options)
        self.controller = PybricksVoiceController.alloc().init()
        self.controller.app = self
        self.ns_app = NSApplication.sharedApplication
        self.ns_app.setActivationPolicy(NSApplicationActivationPolicyRegular)
        self.ns_app.delegate = self.controller
        self.ns_app.mainMenu = self.menu()
        self.shown: dict[str, object] = {}  # what update_view() last set
        self.chat_at_end = True  # where the chat was when it last scrolled
        self.chat_sizes: tuple[float, float, float] | None = None  # on_chat_changed()

    def run(self) -> None:
        """Runs the app until its window closes."""
        self.ns_app.run()
        # run() returns before the loop's last iteration: finish it.
        while self.loop.is_running():
            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, True)
        self.loop.close()

    def launched(self) -> None:
        self.window = self.make_window()
        self.scene_view.on_key = self.on_key
        self.scene_view.on_wheel = lambda delta: self.zoom(closer=delta > 0)

        loop = CFRunLoopEventLoop(CFRunLoopGetMain())
        loop.start()
        # Ctrl+C in the terminal: Python's own handler would raise in a run
        # loop callback, where it can't reach anything.
        loop.add_signal_handler(signal.SIGINT, self.shutdown)
        self.scene_view.on_first_frame = lambda: loop.call_soon_threadsafe(self.scene_ready.set)
        self.scene_view.start()
        self.start(loop)
        self.window.makeKeyAndOrderFront(None)
        self.ns_app.activateIgnoringOtherApps(True)

    # ----- the window ---------------------------------------------------------

    def menu(self) -> Any:
        """The app menu with Quit, and Edit, which text fields need for copy and paste."""
        bar = NSMenu.alloc().init()
        app_item = NSMenuItem.alloc().init()
        bar.addItem(app_item)
        app_menu = NSMenu.alloc().init()
        app_menu.addItemWithTitle("Quit Pybricks Voice", action=SEL("terminate:"), keyEquivalent="q")
        app_item.submenu = app_menu
        edit_item = NSMenuItem.alloc().init()
        bar.addItem(edit_item)
        edit_menu = NSMenu.alloc().initWithTitle("Edit")
        for title, action, key in (
            ("Undo", "undo:", "z"),
            ("Redo", "redo:", "Z"),
            ("Cut", "cut:", "x"),
            ("Copy", "copy:", "c"),
            ("Paste", "paste:", "v"),
            ("Select All", "selectAll:", "a"),
        ):
            edit_menu.addItemWithTitle(title, action=SEL(action), keyEquivalent=key)
        edit_item.submenu = edit_menu
        return bar

    def make_window(self) -> Any:
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
        )
        window = NSWindow.alloc().initWithContentRect(
            NSMakeRect(0, 0, 1280, 800), styleMask=style, backing=NSBackingStoreBuffered, defer=False
        )
        window.title = "Pybricks voice driving"
        window.releasedWhenClosed = False
        window.delegate = self.controller
        window.contentMinSize = (SIDE_WIDTH + 320, 480)
        window.center()
        # Where it was last time, but not when taking a screenshot, which
        # should be the same size every run.
        if self.options.screenshot is None:
            window.setFrameAutosaveName("PybricksVoiceMainWindow")

        content = window.contentView
        width, height = content.bounds.size.width, content.bounds.size.height
        scene_area = NSView.alloc().initWithFrame(NSMakeRect(0, 0, width - SIDE_WIDTH, height))
        scene_area.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable
        content.addSubview(scene_area)
        side = NSView.alloc().initWithFrame(NSMakeRect(width - SIDE_WIDTH, 0, SIDE_WIDTH, height))
        side.autoresizingMask = NSViewMinXMargin | NSViewHeightSizable
        content.addSubview(side)

        self.view = self.scene_view = SceneView(self.step)
        self.scene_view.view.frame = scene_area.bounds
        self.scene_view.view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable
        scene_area.addSubview(self.scene_view.view)
        self.make_hud(scene_area)
        self.make_side(side)
        return window

    def make_hud(self, area: Any) -> None:
        self.hud = _label(mono=True, size=12)
        self.hud.textColor = NSColor.colorWithSRGBRed(0xC9 / 255, green=0xD1 / 255, blue=0xDC / 255, alpha=1)
        box = _rounded(NSView.alloc().init(), 4, NSColor.colorWithWhite(0, alpha=0x88 / 255))
        box.addSubview(self.hud)
        _pin(self.hud, box, leading=8, trailing=8, top=4, bottom=4)
        area.addSubview(box)
        _pin(box, area, leading=12, bottom=12)
        hint = _label("C: camera · scroll: zoom", size=12)
        hint.textColor = NSColor.colorWithSRGBRed(0x8A / 255, green=0x94 / 255, blue=0xA3 / 255, alpha=1)
        area.addSubview(hint)
        _pin(hint, area, trailing=12, top=10)

    def make_side(self, side: Any) -> None:
        title = _label("Talk to the robot", size=17)
        title.font = NSFont.systemFontOfSize(17, weight=NSFontWeightSemibold)
        self.status = _label(wraps=True, secondary=True)

        self.listen = NSSwitch.alloc().init()
        self.listen.state = NSControlStateValueOn
        self.listen.target, self.listen.action = self.controller, SEL("onListen:")
        self.listen_label = _label(self.listen_line, size=12, secondary=True)
        self.level = NSProgressIndicator.alloc().init()
        self.level.indeterminate = False
        self.level.minValue, self.level.maxValue = LEVEL_MIN_DB, LEVEL_MAX_DB
        self.level.doubleValue = LEVEL_MIN_DB
        self.level_text = _label(mono=True, size=12, secondary=True)
        listen_row = _stack([self.listen, self.listen_label, self.level, self.level_text], vertical=False, spacing=8)

        self.typed = NSTextField.textFieldWithString("")
        self.typed.placeholderString = "…or type: forward 30 cm then turn left"
        # Return in the field presses Go, the window's default button.
        go = NSButton.buttonWithTitle("Go", target=self.controller, action=SEL("onGo:"))
        go.keyEquivalent = "\r"
        typed_row = _stack([self.typed, go], vertical=False, spacing=8)

        buttons = _stack(
            [
                NSButton.buttonWithTitle(name, target=self.controller, action=SEL(f"on{name}:"))
                for name in ("Stop", "Reset", "Camera")
            ],
            vertical=False,
            spacing=8,
        )

        self.chat = _stack([], vertical=True, spacing=8)
        document = PybricksVoiceFlippedView.alloc().init()
        document.addSubview(self.chat)
        _pin(self.chat, document, leading=0, trailing=12, top=0, bottom=0)
        self.scroller = NSScrollView.alloc().init()
        self.scroller.hasVerticalScroller = True
        self.scroller.drawsBackground = False
        self.scroller.documentView = document
        _pin(document, self.scroller.contentView, leading=0, trailing=0, top=0)
        clip = self.scroller.contentView
        clip.postsBoundsChangedNotifications = True
        center = NSNotificationCenter.defaultCenter
        for name, view in (
            (NSViewBoundsDidChangeNotification, clip),  # scrolled, or resized
            (NSViewFrameDidChangeNotification, clip),  # resized
            (NSViewFrameDidChangeNotification, document),  # a bubble came or went, or rewrapped
        ):
            center.addObserver(self.controller, selector=SEL("chatChanged:"), name=name, object=view)

        # Three lines high however long the queue, so the chat above doesn't
        # jump as it changes.
        self.queue = _label(wraps=True, secondary=True)
        self.queue.maximumNumberOfLines = 3
        self.queue.lineBreakMode = NSLineBreakByTruncatingTail
        self.queue.heightAnchor.constraintEqualToConstant(54).active = True

        column = _stack(
            [
                _stack([title, self.status], vertical=True, spacing=4),
                listen_row,
                typed_row,
                buttons,
                self.scroller,
                self.queue,
            ],
            vertical=True,
            spacing=12,
        )
        column.distribution = NSStackViewDistributionFill
        side.addSubview(column)
        _pin(column, side, leading=16, trailing=16, top=16, bottom=16)
        # Every row as wide as the column; the chat takes the height that's left.
        for row in (self.status, listen_row, typed_row, self.scroller, self.queue):
            row.widthAnchor.constraintEqualToAnchor(column.widthAnchor).active = True
        self.level.setContentHuggingPriority(1, forOrientation=NSUserInterfaceLayoutOrientationHorizontal)
        self.typed.setContentHuggingPriority(1, forOrientation=NSUserInterfaceLayoutOrientationHorizontal)
        self.scroller.setContentHuggingPriority(1, forOrientation=NSUserInterfaceLayoutOrientationVertical)
        self.status.preferredMaxLayoutWidth = SIDE_WIDTH - 32
        self.queue.preferredMaxLayoutWidth = SIDE_WIDTH - 32

    # ----- Core's -------------------------------------------------------------

    @override
    def make_microphone(self, on_samples: Callable[[np.ndarray], None]) -> Microphone:
        return NoMicrophone()

    @override
    def load_whisper(self) -> Transcriber:
        raise RuntimeError("not on macOS yet")

    @override
    def update_view(self) -> None:
        for key, value, show in (
            ("hud", self.hud_line, self.hud.setStringValue),
            ("status", self.status_line, self.status.setStringValue),
            ("listen", self.listen_line, self.listen_label.setStringValue),
            ("level", self.level_value, self.level.setDoubleValue),
            ("level_text", self.level_line, self.level_text.setStringValue),
            ("queue", self.queue_line, self.queue.setStringValue),
        ):
            if self.shown.get(key) != value:
                self.shown[key] = value
                show(value)

    @override
    def add_bubble(self, text: str, mine: bool, detail: str | None = None) -> None:
        labels = [_label(text, wraps=True)]
        if detail:
            labels.append(_label(detail, size=11, wraps=True))
            labels[-1].alphaValue = 0.75
        for label in labels:
            label.preferredMaxLayoutWidth = MAX_BUBBLE_WIDTH - 24
            if mine:
                label.textColor = NSColor.whiteColor
        bubble = PybricksVoiceBubbleView.alloc().init()
        bubble.mine = mine
        texts = _stack(labels, vertical=True, spacing=2)
        bubble.addSubview(texts)
        _pin(texts, bubble, leading=12, trailing=12, top=8, bottom=8)
        bubble.widthAnchor.constraintLessThanOrEqualToConstant(MAX_BUBBLE_WIDTH).active = True

        # Yours on the right, the robot's on the left, each in a row as wide as the chat.
        row = NSView.alloc().init()
        row.addSubview(bubble)
        if mine:
            _pin(bubble, row, trailing=0, top=0, bottom=0)
        else:
            _pin(bubble, row, leading=0, top=0, bottom=0)
        self.chat.addArrangedSubview(row)
        row.widthAnchor.constraintEqualToAnchor(self.chat.widthAnchor).active = True

        while len(self.chat.arrangedSubviews) > 100:
            old = self.chat.arrangedSubviews.objectAtIndex(0)
            self.chat.removeArrangedSubview(old)
            old.removeFromSuperview()
        # Laying the new bubble out changes the chat's height: on_chat_changed().

    def on_chat_changed(self) -> None:
        """
        Newest message at the bottom, as in a messaging app: when the chat is
        resized or a bubble comes, it stays at the end if it was there; when
        it's scrolled, that is where it was.
        """
        clip, document = self.scroller.contentView, self.scroller.documentView
        sizes = (clip.bounds.size.width, clip.bounds.size.height, document.frame.size.height)
        if sizes != self.chat_sizes:
            self.chat_sizes = sizes
            if self.chat_at_end:
                y = max(0.0, document.frame.size.height - clip.bounds.size.height)
                clip.scrollToPoint(NSMakePoint(0, y))
                self.scroller.reflectScrolledClipView(clip)
        else:
            self.chat_at_end = clip.bounds.origin.y + clip.bounds.size.height >= document.frame.size.height - 1

    @override
    async def save_screenshot(self, path: pathlib.Path) -> None:
        # The window as it is on screen, the 3D view included; needs the
        # Screen Recording permission.
        process = await asyncio.create_subprocess_exec(
            "screencapture", "-x", "-o", f"-l{self.window.windowNumber}", str(path)
        )
        if await process.wait():
            log.warning("couldn't save a screenshot to %s", path)

    @override
    def exit(self) -> None:
        self.loop.stop()
        self.ns_app.stop(None)
        # stop: only takes effect after the next event, which may be a while:
        # post one. rubicon-objc's CocoaLifecycle does the same.
        event = NSEvent.otherEventWithType(
            NSEventTypeApplicationDefined,
            location=NSMakePoint(0, 0),
            modifierFlags=0,
            timestamp=0,
            windowNumber=0,
            context=None,
            subtype=0,
            data1=0,
            data2=0,
        )
        self.ns_app.postEvent(event, atStart=True)

    # ----- input --------------------------------------------------------------

    def submit_typed(self) -> None:
        if self.submit(str(self.typed.stringValue)):
            self.typed.stringValue = ""

    def on_key(self, characters: str) -> bool:
        if characters.lower() == "c":
            self.toggle_camera()
            return True
        return False


def main() -> None:
    options = parse_args()
    MacApp(options).run()
