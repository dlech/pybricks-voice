"""
The app on macOS: AppKit through rubicon-objc.

Each module here starts with `assert not TYPE_CHECKING or sys.platform ==
"darwin"`, so type checking for another platform skips the rest of it. It does
nothing at run time.
"""
