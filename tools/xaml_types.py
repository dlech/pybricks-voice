"""
Generates typed Python classes for XAML files, as WinUI's compiler does for C#.

For each pybricks_voice/*.xaml, writes a .py next to it with a class that
loads the XAML and has an attribute, of the element's WinRT type, for each
element with a Name (or x:Name). So pyright checks every use, and a name
that isn't in the XAML fails here rather than when the app runs.

XamlReader can't take event handlers, which need compiled code-behind, but
it skips attributes in a namespace marked mc:Ignorable. So an element can say
py:Click="on_go", with the py prefix bound to PY below, and the generated
class wires its Click event to the handler of that name on the object it is
given, its code-behind. And py:Text="hud_line", naming a property rather than
an event, binds the property one way to the code-behind's hud_line, as x:Bind
does: the class's update() sets each bound property whose value has changed,
as x:Bind's Bindings.Update() does. The code-behind is a typing.Protocol: its
handlers have the sender and arguments each event's delegate has in PyWinRT's
stubs, and its bound values the type of their property's setter, so pyright
checks that the object has them all, of the right types.

Inside a DataTemplate, py: bindings read from each item rather than from the
code-behind. The template names its items' protocol, <DataTemplate
py:Item="ChatItem">, and the class sets the bound properties of each item's
elements as an ItemsRepeater prepares them (ElementPrepared), with the items
typed by that protocol. The repeater's ItemsSource is any list XAML reads,
such as pybricks_voice.observable_list.ObservableList.

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
import dataclasses
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


def find_setter(element: Type, prop: str) -> tuple[str, Type]:
    """
    For a property of a WinRT type, by its WinRT name as XAML uses it: its
    Python name, and the type its setter takes ("...::put_Text(...)").
    """
    module, type_ = element
    signature = re.compile(rf"^\s*# .*::put_{re.escape(prop)}\(")
    for base in getattr(importlib.import_module(module), type_).__mro__:
        if base.__module__ == "winrt.system" or not base.__module__.startswith("winrt."):
            continue
        tree, _, lines = _stub(base.__module__)
        for node in tree.body:
            if not (isinstance(node, ast.ClassDef) and node.name == base.__name__):
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                first = min([d.lineno for d in item.decorator_list] + [item.lineno])
                if any(signature.match(line) for line in lines[first - 2 : item.lineno - 1]):
                    annotation = (item.args.posonlyargs + item.args.args)[1].annotation  # self, value
                    assert annotation is not None
                    if isinstance(annotation, ast.Name) and annotation.id in BUILTINS:
                        return item.name, ("builtins", annotation.id)
                    return item.name, _resolve(annotation, base.__module__)
    raise SystemExit(f"{type_} has no settable {prop} property")


BUILTINS = {"str", "int", "float", "bool", "bytes"}


def _delegate(annotation: ast.expr, module: str) -> tuple[Type, Type]:
    # TypedEventHandler[Sender, Args], EventHandler[Args] (whose sender is an
    # Object), or an alias for Callable[[Sender, Args], object].
    if isinstance(annotation, ast.Subscript):
        if not isinstance(annotation.slice, ast.Tuple):
            return ("winrt.system", "Object"), _resolve(annotation.slice, module)
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
            if namespace == PY:
                if tag == "DataTemplate" and attribute == "Item":
                    continue  # its items' protocol
                if not {f"add_{attribute}", f"put_{attribute}"} & names:
                    found.append(f"{label}: py:{attribute} is neither an event nor a settable property")
                continue
            if namespace:
                continue  # x: and mc:
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


@dataclasses.dataclass
class _Template:
    """A DataTemplate with py: bindings, and the ItemsRepeater it is the ItemTemplate of."""

    item: str  # its items' protocol
    owner: str  # the repeater's name
    root: Type | None = None
    bindings: list[tuple[str | None, Type, str, str]] = dataclasses.field(default_factory=list)
    # (element's name, None for the root; its type; Python property; item attribute)
    sources: dict[str, Type] = dataclasses.field(default_factory=dict)


def generate(xaml: pathlib.Path) -> str:
    root = ElementTree.parse(xaml).getroot()
    if found := problems(root):
        raise SystemExit(f"{xaml.name}:\n  " + "\n  ".join(found))
    root_module, root_type = winrt_type(root.tag)
    root_attribute = root_type.lower()
    named: list[tuple[str, str, str]] = []  # (name, module, type)
    wiring: list[tuple[str, str, str]] = []  # (attribute, add method, handler)
    handlers: dict[str, tuple[Type, Type]] = {}  # handler: (sender, args)
    bindings: list[tuple[str, str, str]] = []  # (attribute, Python property, source)
    sources: dict[str, Type] = {}  # source: the setter's type
    templates: dict[ElementTree.Element, _Template] = {}
    parents = {child: parent for parent in root.iter() for child in parent}

    def template_of(element: ElementTree.Element) -> ElementTree.Element | None:
        while (element := parents.get(element)) is not None:  # pyright: ignore[reportAssignmentType]
            if element in templates:
                return element
        return None

    for element in root.iter():
        if _split(element.tag) == (PRESENTATION, "DataTemplate"):
            item = element.get(f"{{{PY}}}Item")
            holder = parents.get(element)
            owner = parents.get(holder) if holder is not None else None
            if not item:
                continue  # a template without bindings
            if holder is None or owner is None or _split(holder.tag)[1] != "ItemsRepeater.ItemTemplate":
                raise SystemExit(f"{xaml.name}: py:Item works only in an ItemsRepeater's ItemTemplate, so far")
            if not owner.get("Name"):
                raise SystemExit(f"{xaml.name}: the ItemsRepeater of py:Item={item!r} needs a Name")
            templates[element] = _Template(item, owner.get("Name") or "")
            continue
        if (template := template_of(element)) is not None:
            info = templates[template]
            if "." in _split(element.tag)[1]:
                continue  # a property element
            type_ = winrt_type(element.tag)
            is_root = parents[element] is template
            if is_root:
                info.root = type_
            name = element.get("Name") or element.get(f"{{{X}}}Name")
            for key, target in element.attrib.items():
                if not key.startswith(f"{{{PY}}}"):
                    continue
                member = key.removeprefix(f"{{{PY}}}")
                if f"add_{member}" in members(type_):
                    raise SystemExit(f"{xaml.name}: events in a DataTemplate aren't supported yet (py:{member})")
                if not is_root and not name:
                    raise SystemExit(f"{xaml.name}: an element in a DataTemplate with py:{member} needs a Name")
                prop, value = find_setter(type_, member)
                if info.sources.setdefault(target, value) != value:
                    raise SystemExit(f"{xaml.name}: {info.item}.{target} is bound to properties of different types")
                info.bindings.append((None if is_root else name, type_, prop, target))
            continue
        name = element.get("Name") or element.get(f"{{{X}}}Name")
        if element is root:
            name = root_attribute
        elif name:
            named.append((name, *winrt_type(element.tag)))
        for key, target in element.attrib.items():
            if not key.startswith(f"{{{PY}}}"):
                continue
            member = key.removeprefix(f"{{{PY}}}")
            if not name:
                raise SystemExit(f"{xaml.name}: an element with py:{member} needs a Name")
            type_ = winrt_type(element.tag)
            if f"add_{member}" in members(type_):  # an event, and its handler
                add, sender, args = find_event(type_, member)
                if handlers.setdefault(target, (sender, args)) != (sender, args):
                    raise SystemExit(f"{xaml.name}: {target} handles events of different types")
                wiring.append((name, add, target))
            else:  # a property, and the value it is bound to
                prop, value = find_setter(type_, member)
                if sources.setdefault(target, value) != value:
                    raise SystemExit(f"{xaml.name}: {target} is bound to properties of different types")
                bindings.append((name, prop, target))
    if len({n for n, _, _ in named}) != len(named):
        raise SystemExit(f"{xaml.name}: two elements have the same name")
    imports: dict[str, set[str]] = {"winrt.microsoft.ui.xaml": {"FrameworkElement"}}
    imports.setdefault(root_module, set()).add(root_type)
    imports.setdefault("winrt.microsoft.ui.xaml.markup", set()).add("XamlReader")
    imports.setdefault("typing_extensions", set()).add("Sentinel")  # PEP 661, not yet in typing
    for _, module, type_ in named:
        imports.setdefault(module, set()).add(type_)
    for sender, args in handlers.values():
        for module, type_ in (sender, args):
            imports.setdefault(module, set()).add(type_)
    for module, type_ in sources.values():
        if module != "builtins":
            imports.setdefault(module, set()).add(type_)
    for info in templates.values():
        if info.root is None:
            raise SystemExit(f"{xaml.name}: the DataTemplate of {info.item} is empty")
        imports.setdefault("winrt.microsoft.ui.xaml.controls", set()).add("ItemsRepeaterElementPreparedEventArgs")
        imports.setdefault("winrt.system", set()).add("Object")
        for module, type_ in [info.root, *(t for _, t, _, _ in info.bindings), *info.sources.values()]:
            if module != "builtins":
                imports.setdefault(module, set()).add(type_)

    def template_protocol(info: _Template) -> list[str]:
        return [
            "",
            "",
            f"class {info.item}(typing.Protocol):",
            f'    """An item of {info.owner}, the ItemsRepeater: what its DataTemplate\'s py: bindings read."""',
            "",
            *(f"    @property\n    def {source}(self) -> {value[1]}: ..." for source, value in info.sources.items()),
        ]

    def template_method(info: _Template) -> list[str]:
        assert info.root is not None
        named = list(dict.fromkeys((name, type_) for name, type_, _, _ in info.bindings if name))
        return [
            "",
            f"    def _prepare_{info.owner}(self, sender: Object, args: ItemsRepeaterElementPreparedEventArgs)"
            " -> None:",
            f'        """Sets the bound properties of an element of {info.owner} from its item, as it is prepared."""',
            f"        item = typing.cast({info.item}, self.{info.owner}.items_source_view.get_at(args.index))",
            f"        root = args.element.as_({info.root[1]})",
            *(["        find = root.find_name"] if named else []),
            *(f'        {name} = find("{name}").as_({type_[1]})' for name, type_ in named),
            *(f"        {name or 'root'}.{prop} = item.{source}" for name, _, prop, source in info.bindings),
        ]

    class_name = "".join(part.title() for part in xaml.stem.split("_"))
    # A Window's elements are found from its content; any other root, from itself.
    scope = f"self.{root_attribute}.content" if root_type == "Window" else f"self.{root_attribute}"

    lines = [
        f"# Generated by tools/xaml_types.py from {xaml.name}: don't edit. Run it again",
        "# after renaming an element there, changing its type, or changing a py: attribute.",
        f'"""Typed access to {xaml.name}\'s named elements, its events wired and its bindings."""',
        "",
        "from __future__ import annotations",
        "",
        "import pathlib",
        "import typing",
        "",
        *(f"from {module} import {', '.join(sorted(names))}" for module, names in sorted(imports.items())),
        "",
        f'XAML = pathlib.Path(__file__).with_name("{xaml.name}")',
        '_UNSET = Sentinel("_UNSET")  # a binding not yet set',
        "",
        "",
        f"class {class_name}CodeBehind(typing.Protocol):",
        f'    """What {xaml.name} names: event handlers (py:Click="on_go") and bound values (py:Text="hud_line")."""',
        "",
        *(
            f"    def {handler}(self, sender: {sender[1]}, args: {args[1]}, /) -> object: ..."
            for handler, (sender, args) in handlers.items()
        ),
        *(f"    @property\n    def {source}(self) -> {value[1]}: ..." for source, value in sources.items()),
        *(line for info in templates.values() for line in template_protocol(info)),
        "",
        "",
        f"class {class_name}:",
        f'    """{xaml.name}, loaded, with its named elements, its events wired and its properties bound."""',
        "",
        f"    def __init__(self, code_behind: {class_name}CodeBehind) -> None:",
        "        self._code_behind = code_behind",
        "        self._shown: dict[str, object] = {}",
        f'        self.{root_attribute} = XamlReader.load(XAML.read_text(encoding="utf-8")).as_({root_type})',
        f"        find = {scope}.as_(FrameworkElement).find_name",
        *(f'        self.{name} = find("{name}").as_({type_})' for name, _, type_ in named),
        *(f"        self.{attribute}.{add}(code_behind.{handler})" for attribute, add, handler in wiring),
        *(f"        self.{info.owner}.add_element_prepared(self._prepare_{info.owner})" for info in templates.values()),
        "",
        "    def update(self) -> None:",
        '        """Sets each bound property whose value in the code-behind has changed, as Bindings.Update()."""',
        "        code_behind, shown = self._code_behind, self._shown",
        *(
            f'        if (value := code_behind.{source}) != shown.get("{attribute}.{prop}", _UNSET):\n'
            f'            self.{attribute}.{prop} = shown["{attribute}.{prop}"] = value'
            for attribute, prop, source in bindings
        ),
        *([] if bindings else ["        del code_behind, shown"]),
        *(line for info in templates.values() for line in template_method(info)),
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
