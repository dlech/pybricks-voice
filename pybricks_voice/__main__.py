import sys

if sys.platform == "win32":
    from .windows.app import main
else:
    sys.exit(f"no app for {sys.platform} yet")

main()
