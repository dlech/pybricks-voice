import importlib.util
import pathlib
import tempfile
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


class TestTemplates(unittest.TestCase):
    """py: bindings in a DataTemplate, which read from each item."""

    def generate(self, repeater: str) -> str:
        xaml = (
            '<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"'
            ' xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
            ' xmlns:py="https://pybricks.com/pybricks-voice/xaml" mc:Ignorable="py">' + repeater + "</Window>"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "test_window.xaml"
            path.write_text(xaml, encoding="utf-8")
            return xaml_types.generate(path)

    def template(self, content: str, item: str = ' py:Item="Row"') -> str:
        return self.generate(
            '<ItemsRepeater Name="rows" py:ItemsSource="items"><ItemsRepeater.ItemTemplate>'
            f"<DataTemplate{item}>{content}</DataTemplate>"
            "</ItemsRepeater.ItemTemplate></ItemsRepeater>"
        )

    def test_bindings(self) -> None:
        code = self.template('<Border py:Style="look"><TextBlock Name="label" py:Text="name"/></Border>')
        self.assertIn("class Row(typing.Protocol):", code)
        self.assertIn("    def look(self) -> Style: ...", code)
        self.assertIn("    def name(self) -> str: ...", code)
        self.assertIn("        self.rows.add_element_prepared(self._prepare_rows)", code)
        self.assertIn("        item = typing.cast(Row, self.rows.items_source_view.get_at(args.index))", code)
        self.assertIn('        label = find("label").as_(TextBlock)', code)
        self.assertIn("        root.style = item.look", code)
        self.assertIn("        label.text = item.name", code)
        self.assertNotIn("self.label", code)  # not an element of the window

    def test_needs_a_repeater(self) -> None:
        with self.assertRaisesRegex(SystemExit, "only in an ItemsRepeater's ItemTemplate"):
            self.generate(
                '<Grid><Grid.Resources><DataTemplate py:Item="Row"><Border/></DataTemplate></Grid.Resources></Grid>'
            )

    def test_needs_names(self) -> None:
        with self.assertRaisesRegex(SystemExit, "needs a Name"):
            self.template('<Border><TextBlock py:Text="name"/></Border>')

    def test_no_events(self) -> None:
        with self.assertRaisesRegex(SystemExit, "events in a DataTemplate"):
            self.template('<Button py:Click="on_click"/>')

    def test_event_handler(self) -> None:
        # EventHandler[Args], whose sender is an Object, rather than TypedEventHandler.
        code = self.generate('<ScrollViewer Name="scroller" py:ViewChanged="on_view_changed"/>')
        self.assertIn("self, sender: Object, args: ScrollViewerViewChangedEventArgs, /", code)

    def test_without_item(self) -> None:
        self.assertNotIn("_prepare_rows", self.template("<Border/>", item=""))


if __name__ == "__main__":
    unittest.main()
