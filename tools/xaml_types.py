"""
Generates typed Python classes for XAML files, as WinUI's compiler does for C#.

For each pybricks_voice/*.xaml, writes a .py next to it with a class that
loads the XAML and has an attribute, of the element's WinRT type, for each
element with a Name (or x:Name). So pyright checks every use, and a name
that isn't in the XAML fails here rather than when the app runs.

XamlReader can't take event handlers, which need compiled code-behind, but
it skips attributes in a namespace marked mc:Ignorable. So an element can say
py:Click="submit_typed", with the py prefix bound to PY below, and the
generated class wires its Click event to the handler of that name on the
object it is given. The handlers are a typing.Protocol, with the sender and
arguments each event's delegate has in PyWinRT's stubs, so pyright checks
that the object has them all, with the right signatures.

First it checks the XAML the way WinUI's compiler would, against the WinRT
signatures the stubs have in comments: every element is a WinUI type, every
attribute a settable property of its element or an attached property of its
owner (Grid.Row), every property element a property, and an event attribute
gets told to be py:. Values aren't checked.

The XAML is read when the class is used, so changing the layout needs no
regenerating; renaming an element, changing its type, or changing an event
or a handler does.

    uv run tools/xaml_types.py [--check]

--check only reports whether the generated files are up to date (the tests
run it).
"""

import argparse
import ast
import functools
import importlib
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ElementTree

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "pybricks_voice"
PRESENTATION = "http://schemas.microsoft.com/winfx/2006/xaml/presentation"
X = "http://schemas.microsoft.com/winfx/2006/xaml"
PY = "https://pybricks.com/pybricks-voice/xaml"  # handlers: py:Event="handler"
# Where WinUI's types for the default XAML namespace are, in the order to look.
MODULES = [
    "winrt.microsoft.ui.xaml",
    "winrt.microsoft.ui.xaml.controls",
    "winrt.microsoft.ui.xaml.controls.primitives",
    "winrt.microsoft.ui.xaml.shapes",
    "winrt.microsoft.ui.xaml.media",
    "winrt.microsoft.ui.xaml.documents",
]

Type = tuple[str, str]  # module, name


def winrt_type(tag: str) -> Type:
    """The module and name of the WinRT type for an element's tag."""
    namespace, _, name = tag[1:].partition("}") if tag.startswith("{") else ("", "", tag)
    if namespace != PRESENTATION:
        raise SystemExit(f"{tag}: only WinUI's own elements can be named here")
    for module in MODULES:
        if hasattr(importlib.import_module(module), name):
            return module, name
    raise SystemExit(f"{name} isn't in any of {', '.join(MODULES)}")


@functools.cache
def _stub(module: str) -> tuple[ast.Module, dict[str, str], list[str]]:
    """A module's .pyi, the modules its import aliases name, and its lines."""
    path = pathlib.Path(importlib.import_module(module).__file__ or "").with_suffix(".pyi")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name] = name.name
    return tree, aliases, text.splitlines()


def _resolve(node: ast.expr, module: str) -> Type:
    """The module and name an annotation in a module's stub refers to."""
    if isinstance(node, ast.Name):
        return module, node.id
    prefix, _, name = ast.unparse(node).rpartition(".")
    return _stub(module)[1].get(prefix, prefix), name


def find_event(element: Type, event: str) -> tuple[str, Type, Type]:
    """
    For an event of a WinRT type, by its WinRT name as XAML uses it: the
    Python method that adds a handler, and the handler's sender and arguments
    types. Each stub method has a comment above it with its WinRT signature,
    which names the event's add method: "...::add_PointerWheelChanged(...)".
    """
    module, type_ = element
    signature = re.compile(rf"^\s*# .*::add_{re.escape(event)}\(")
    for base in getattr(importlib.import_module(module), type_).__mro__:
        if base.__module__ == "winrt.system" or not base.__module__.startswith("winrt."):
            continue  # Object and the like: no events, and no stub of their own
        tree, _, lines = _stub(base.__module__)
        for node in tree.body:
            if not (isinstance(node, ast.ClassDef) and node.name == base.__name__):
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                # The comment is above the def, or above its decorators: lineno is the def's own line.
                first = min([d.lineno for d in item.decorator_list] + [item.lineno])
                if any(signature.match(line) for line in lines[first - 2 : item.lineno - 1]):
                    annotation = (item.args.posonlyargs + item.args.args)[1].annotation  # self, handler
                    assert annotation is not None
                    return item.name, *_delegate(annotation, base.__module__)
    raise SystemExit(f"{type_} has no {event} event")


def _delegate(annotation: ast.expr, module: str) -> tuple[Type, Type]:
    # TypedEventHandler[Sender, Args], or an alias for Callable[[Sender, Args], object].
    if isinstance(annotation, ast.Subscript):
        assert isinstance(annotation.slice, ast.Tuple)
        sender, args = annotation.slice.elts
        return _resolve(sender, module), _resolve(args, module)
    alias_module, name = _resolve(annotation, module)
    for node in _stub(alias_module)[0].body:
        if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name) and node.name.id == name:
            callable_ = node.value
            assert isinstance(callable_, ast.Subscript) and isinstance(callable_.slice, ast.Tuple)
            parameters = callable_.slice.elts[0]
            assert isinstance(parameters, ast.List)
            sender, args = parameters.elts
            return _resolve(sender, alias_module), _resolve(args, alias_module)
    raise SystemExit(f"{name} isn't a delegate in {alias_module}")


_MEMBER = re.compile(r"^\s*# .*::(\w+)\(")


@functools.cache
def members(element: Type) -> frozenset[str]:
    """
    The WinRT names of a type's members, its bases' included, from the
    signatures commented in the stubs: get_MinHeight, put_MinHeight,
    add_Click, SetRow, get_RowProperty...
    """
    module, type_ = element
    names: set[str] = set()
    for base in getattr(importlib.import_module(module), type_).__mro__:
        if base.__module__ == "winrt.system" or not base.__module__.startswith("winrt."):
            continue
        tree, _, lines = _stub(base.__module__)
        for node in tree.body:
            # Statics, attached properties' setters among them, are on a metaclass: Grid_Static.
            if isinstance(node, ast.ClassDef) and node.name in (base.__name__, f"{base.__name__}_Static"):
                names.update(
                    m.group(1) for line in lines[node.lineno - 1 : node.end_lineno] if (m := _MEMBER.match(line))
                )
    return frozenset(names)


def _split(tag: str) -> tuple[str, str]:
    """{namespace}name -> (namespace, name)."""
    namespace, _, name = tag[1:].partition("}") if tag.startswith("{") else ("", "", tag)
    return namespace, name


def _winui(name: str) -> Type | None:
    try:
        return winrt_type(f"{{{PRESENTATION}}}{name}")
    except SystemExit:
        return None


def problems(root: ElementTree.Element) -> list[str]:
    """
    What WinUI's XAML compiler would reject before the app runs: elements
    that aren't WinUI types, and attributes or property elements that aren't
    settable properties of their element, or attached properties of their
    owner. Values aren't checked.
    """
    found = []
    for element in root.iter():
        namespace, tag = _split(element.tag)
        if namespace != PRESENTATION:
            continue  # x:Double and the like
        if "." in tag:  # a property element: <Grid.ColumnDefinitions>
            owner, _, prop = tag.partition(".")
            type_ = _winui(owner)
            if type_ is None:
                found.append(f"<{tag}>: {owner} isn't a WinUI type")
            elif not {f"get_{prop}", f"put_{prop}"} & members(type_):
                found.append(f"<{tag}>: {owner} has no {prop} property")
            continue
        type_ = _winui(tag)
        if type_ is None:
            found.append(f"<{tag}> isn't a WinUI type")
            continue
        names = members(type_)
        label = f'<{tag} Name="{element.get("Name")}">' if element.get("Name") else f"<{tag}>"
        for key in element.attrib:
            namespace, attribute = _split(key)
            if namespace:
                continue  # x:, mc: and py:, which is checked where it is wired
            if "." in attribute:  # an attached property: Grid.Row
                owner, _, prop = attribute.partition(".")
                type_ = _winui(owner)
                if type_ is None or not {f"Set{prop}", f"get_{prop}Property"} & members(type_):
                    found.append(f"{label}: {attribute} isn't an attached property")
            elif f"put_{attribute}" in names:
                continue
            elif f"add_{attribute}" in names:
                found.append(f"{label}: {attribute} is an event, which XamlReader rejects: use py:{attribute}")
            elif f"get_{attribute}" in names:
                found.append(f"{label}: {attribute} is read-only")
            else:
                found.append(f"{label}: {tag} has no {attribute} property")
    return found


def generate(xaml: pathlib.Path) -> str:
    root = ElementTree.parse(xaml).getroot()
    if found := problems(root):
        raise SystemExit(f"{xaml.name}:\n  " + "\n  ".join(found))
    root_module, root_type = winrt_type(root.tag)
    root_attribute = root_type.lower()
    named: list[tuple[str, str, str]] = []  # (name, module, type)
    wiring: list[tuple[str, str, str]] = []  # (attribute, add method, handler)
    handlers: dict[str, tuple[Type, Type]] = {}  # handler: (sender, args)
    for element in root.iter():
        name = element.get("Name") or element.get(f"{{{X}}}Name")
        if element is root:
            name = root_attribute
        elif name:
            named.append((name, *winrt_type(element.tag)))
        for key, handler in element.attrib.items():
            if not key.startswith(f"{{{PY}}}"):
                continue
            event = key.removeprefix(f"{{{PY}}}")
            if not name:
                raise SystemExit(f"{xaml.name}: an element with py:{event} needs a Name")
            add, sender, args = find_event(winrt_type(element.tag), event)
            if handlers.setdefault(handler, (sender, args)) != (sender, args):
                raise SystemExit(f"{xaml.name}: {handler} handles events of different types")
            wiring.append((name, add, handler))
    if len({n for n, _, _ in named}) != len(named):
        raise SystemExit(f"{xaml.name}: two elements have the same name")
    imports: dict[str, set[str]] = {"winrt.microsoft.ui.xaml": {"FrameworkElement"}}
    imports.setdefault(root_module, set()).add(root_type)
    imports.setdefault("winrt.microsoft.ui.xaml.markup", set()).add("XamlReader")
    for _, module, type_ in named:
        imports.setdefault(module, set()).add(type_)
    for sender, args in handlers.values():
        for module, type_ in (sender, args):
            imports.setdefault(module, set()).add(type_)
    class_name = "".join(part.title() for part in xaml.stem.split("_"))
    # A Window's elements are found from its content; any other root, from itself.
    scope = f"self.{root_attribute}.content" if root_type == "Window" else f"self.{root_attribute}"

    lines = [
        f"# Generated by tools/xaml_types.py from {xaml.name}: don't edit. Run it again",
        "# after renaming an element there, changing its type, or changing a handler.",
        f'"""Typed access to {xaml.name}\'s named elements, and its events wired."""',
        "",
        "from __future__ import annotations",
        "",
        "import pathlib",
        "import typing",
        "",
        *(f"from {module} import {', '.join(sorted(names))}" for module, names in sorted(imports.items())),
        "",
        f'XAML = pathlib.Path(__file__).with_name("{xaml.name}")',
        "",
        "",
        f"class {class_name}Handlers(typing.Protocol):",
        f'    """The event handlers {xaml.name} names, as py:Event="handler"."""',
        "",
        *(
            f"    def {handler}(self, sender: {sender[1]}, args: {args[1]}, /) -> object: ..."
            for handler, (sender, args) in handlers.items()
        ),
        "",
        "",
        f"class {class_name}:",
        f'    """{xaml.name}, loaded, with its named elements, and its events wired to handlers."""',
        "",
        f"    def __init__(self, handlers: {class_name}Handlers) -> None:",
        f'        self.{root_attribute} = XamlReader.load(XAML.read_text(encoding="utf-8")).as_({root_type})',
        f"        find = {scope}.as_(FrameworkElement).find_name",
        *(f'        self.{name} = find("{name}").as_({type_})' for name, _, type_ in named),
        *(f"        self.{attribute}.{add}(handlers.{handler})" for attribute, add, handler in wiring),
    ]
    formatted = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", str(xaml.with_suffix(".py")), "-"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        cwd=ROOT,
    )
    return formatted.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="only report generated files that are out of date")
    options = parser.parse_args()
    stale = []
    for xaml in sorted(PACKAGE.glob("*.xaml")):
        target = xaml.with_suffix(".py")
        text = generate(xaml)
        if target.exists() and target.read_text(encoding="utf-8") == text:
            continue
        stale.append(target.name)
        if not options.check:
            target.write_text(text, encoding="utf-8", newline="\n")
    if options.check and stale:
        sys.exit(f"out of date: {', '.join(stale)}; run `uv run tools/xaml_types.py`")
    print(f"{'out of date' if options.check else 'wrote'}: {', '.join(stale) or 'nothing'}", file=sys.stderr)


if __name__ == "__main__":
    main()
