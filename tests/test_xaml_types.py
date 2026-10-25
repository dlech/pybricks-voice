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


class TestProblems(unittest.TestCase):
    """What WinUI's XAML compiler would reject, found before the app runs."""

    def problems(self, body: str) -> list[str]:
        xaml = (
            '<Grid xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"'
            ' xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml">' + body + "</Grid>"
        )
        return xaml_types.problems(xaml_types.ElementTree.fromstring(xaml))

    def test_valid(self) -> None:
        body = """
            <Grid.RowDefinitions><RowDefinition Height="Auto"/></Grid.RowDefinitions>
            <ProgressBar x:Name="bar" Grid.Row="1" MinHeight="10" Value="3">
                <ProgressBar.Resources><x:Double x:Key="ProgressBarTrackHeight">10</x:Double></ProgressBar.Resources>
            </ProgressBar>
            <Button Content="Go" ToolTipService.ToolTip="Go"/>
        """
        self.assertEqual(self.problems(body), [])

    def test_misspelled_property(self) -> None:
        self.assertEqual(self.problems('<TextBlock Txt="hi"/>'), ["<TextBlock>: TextBlock has no Txt property"])

    def test_event_attribute(self) -> None:
        self.assertIn("use py:Click", self.problems('<Button Name="go" Click="on_go"/>')[0])

    def test_py_attributes(self) -> None:
        body = '<TextBlock xmlns:py="https://pybricks.com/pybricks-voice/xaml" Name="a" py:Txt="line" py:Text="line"/>'
        self.assertEqual(
            self.problems(body), ['<TextBlock Name="a">: py:Txt is neither an event nor a settable property']
        )

    def test_read_only(self) -> None:
        self.assertEqual(self.problems('<TextBox ActualWidth="3"/>'), ["<TextBox>: ActualWidth is read-only"])

    def test_attached(self) -> None:
        self.assertEqual(self.problems('<Button Grid.Rows="1"/>'), ["<Button>: Grid.Rows isn't an attached property"])

    def test_elements(self) -> None:
        self.assertEqual(
            self.problems("<Buton/><Grid.Columns/>"),
            ["<Buton> isn't a WinUI type", "<Grid.Columns>: Grid has no Columns property"],
        )


if __name__ == "__main__":
    unittest.main()
