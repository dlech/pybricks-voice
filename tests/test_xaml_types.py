import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("xaml_types", ROOT / "tools" / "xaml_types.py")
assert _spec is not None and _spec.loader is not None
xaml_types = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xaml_types)


class TestGeneratedXamlTypes(unittest.TestCase):
    def test_up_to_date(self) -> None:
        # What WinUI's compiler would catch for C#: an element renamed, or its
        # type changed, in the XAML but not in the class the app uses.
        for xaml in sorted((ROOT / "pybricks_voice").glob("*.xaml")):
            generated = xaml.with_suffix(".py").read_text(encoding="utf-8")
            self.assertEqual(generated, xaml_types.generate(xaml), f"run tools/xaml_types.py for {xaml.name}")


if __name__ == "__main__":
    unittest.main()
