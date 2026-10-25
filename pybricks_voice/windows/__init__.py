"""
The app on Windows: WinUI 3 through PyWinRT, Whisper on the Snapdragon's NPU.

Each module here starts with `assert not TYPE_CHECKING or sys.platform ==
"win32"`, so type checking for another platform skips the rest of it. It does
nothing at run time.
"""
