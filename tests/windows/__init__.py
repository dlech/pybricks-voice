"""WinUI and WinRT tests, which run only on Windows."""

import sys
import unittest


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    if sys.platform == "win32":
        tests.addTests(loader.discover(__path__[0], pattern or "test*.py"))
    return tests
