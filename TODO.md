# To do

What needs improving, here and in the projects this app is built on.

## This app

- [ ] **Demo clip:** record the 30-second clip of driving by voice, the last item of
      milestone 1.
- [ ] **Grammar gaps** found in real sessions:
  - "for return." (Whisper mishearing "forward"), "forward at 10.", "forward 16.6"
  - a request split by a pause: "40 squares." … "centimeters."
- [ ] **Background conversation drives the robot:** talk near the microphone is taken
      as commands. "You made a drive in a square" drove one, because "made" isn't one
      of the words that mark a comment. Consider requiring a wake word, or only
      acting on whole commands.
- [x] **A realistic robot:** Driving Base 1 from LDraw's parts (`models/`), without its
      cables. The wheels and hub are placed by measurement rather than snapped
      together in a LEGO CAD program, so small parts are left out; a model
      built in Studio or LDCad would be more faithful.
- [ ] **No side effects on import:** `scene.py` puts `vendor/` on PATH before importing
      pygfx, since freetype-py looks for its DLL when imported, so pygfx must
      be imported after it. Build a local freetype-py wheel with the DLL in it,
      as `get_mujoco.py` does for MuJoCo, until freetype-py#210 ships one.
- [ ] **The 3D view looks slightly soft,** probably pygfx's antialiasing filter. Try
      its pixel filter settings.
- [x] **Screenshot device:** `windows/screenshot.py` uses PyWinRT's `create_direct3d11_device()`
      rather than ctypes.
- [x] **Physics:** the robot is simulated with MuJoCo (rigid bodies, wheel slip,
      pushing things). Still to do: use it for the "goals in the scene" milestone,
      and show where the robot thinks it is (odometry) next to where it is.
- [ ] **Screenshot:** one capture came back blank while the app was running fine;
      if it happens again, wait for a second frame.
- [ ] **Milestone 2:** drive a real hub over Bluetooth LE with PyWinRT.

## Upstream

### PyWinRT

- [x] **`create_direct3d11_device()`** in `winrt-Windows.Graphics.DirectX.Direct3D11.Interop`
      4.1.0 ([pywinrt#159](https://github.com/pywinrt/pywinrt/pull/159), asked for in
      [pywinrt#78](https://github.com/pywinrt/pywinrt/issues/78)).
- [ ] **A pointer from an `InterfaceCapsule`:** `winrt.runtime.interop` makes and wraps
      capsules but can't read one, so `windows/scene_view.py` uses ctypes' `PyCapsule_GetPointer`
      to hand `ISwapChainPanelNative` to wgpu. Something like
      `interface_pointer(capsule) -> int`, for any native library (wgpu, Skia, ctypes).
- [ ] **`winrt-Microsoft.UI.Xaml.Media.DXInterop`,** like the Direct3D11 interop: what
      `microsoft.ui.xaml.media.dxinterop.h` declares, `ISwapChainPanelNative`
      (`set_swap_chain`, which works only on the UI thread) and the
      `ISurfaceImageSourceNative` family, with their IIDs, which `windows/scene_view.py` copies
      from the header by hand. For apps that bring their own Direct3D.
- [ ] **The guest asyncio loop** (`pybricks_voice/windows/loop.py`) in `winrt-runtime`, with its
      tests.
- [ ] **A WinUI startup helper** for unpackaged apps: DPI awareness (`python.exe`'s
      manifest declares none), AppUserModelID, apartment and bootstrap, the
      `IXamlMetadataProvider` boilerplate, and remembering the window's placement
      (`pybricks_voice/windows/placement.py`), which WinUI 3 doesn't do as UWP did.
- [ ] **The guest loop's timer is coarse:** its `DispatcherQueueTimer` fires on the
      system's ~15.6 ms tick, so a 16 ms `asyncio.sleep` often takes 31 or 47 ms
      (see `spikes/drag_stall.py`). The iocp-handoff thread could wait with a
      high-resolution waitable timer instead and enqueue the iteration itself.
- [ ] **Document the drag-start stall:** pressing on the title bar or a border
      blocks the UI thread in `DefWindowProc` (`SC_MOVE`/`SC_SIZE`, in the kernel)
      until the mouse moves, dispatching nothing, so the guest loop stops too;
      inside the move loop that follows it runs fine. Apps that draw
      continuously need a render thread, as this one has. Posting a
      `WM_MOUSEMOVE` on `WM_NCLBUTTONDOWN` ends the wait and keeps double-click
      working, but it is a hack nothing well-known relies on.
- [ ] **Typed XAML for Python:** `tools/xaml_types.py` generates a class with a typed
      attribute for each named element of a `.xaml` file, and wires events to
      handlers named in the XAML as `py:Click="..."` in an `mc:Ignorable`
      namespace (XamlReader rejects plain event attributes), and `py:Text="hud_line"`
      binds a property one way to a value, as `x:Bind` does, set by the class's
      `update()` when it changes. The handlers and values are a Protocol typed from
      the stubs' delegates and setters, so pyright checks them as C#'s compiler
      checks code-behind. It also checks every element, attribute and property
      element against the stubs, as the compiler would. Bindings in an
      `ItemsRepeater`'s `DataTemplate` read from each item. It could be a PyWinRT tool.
- [ ] **XAML metadata from the PyWinRT generator,** for that tool to read rather than
      the stubs' signature comments or the private `_table.pywinrt`: per package, a
      small documented file (JSON, say) with each type's WinRT and Python names, its
      base, its properties (type, settable, dependency or attached), its events with
      their delegates' Python sender and arguments types already resolved, its
      `[ContentProperty]`, and enum members. Then values can be checked too
      (`Visibility="Colapsed"`), and children against content properties.
- [x] **Lists and item templates from Python:** the chat is an `ItemsRepeater` whose
      `DataTemplate` binds with `py:` too, to each item (`py:Item="ChatItem"` names
      their Protocol). The generated class sets them in the repeater's
      `ElementPrepared`, so neither `IDataTemplateComponent` nor `{Binding}` was
      needed. Its `ItemsSource` is `pybricks_voice/windows/observable_list.py`.
- [ ] **Events in a DataTemplate,** a button on a bubble say: the generator rejects
      them. The `_prepare_` method could wire them to a code-behind handler that
      also gets the item, and must not wire one twice as the repeater recycles
      elements.
- [ ] **Bindings that update themselves:** the code-behind calls `update()` each
      tick, as x:Bind's `Bindings.Update()`. For x:Bind's `Mode=OneWay`, Python
      objects would implement `INotifyPropertyChanged` (non-generic, like the
      `INotifyCollectionChanged` Python implements) and the generated class
      would listen; for XAML's own `{Binding}`, PyWinRT would also have to let
      Python objects be `ICustomPropertyProvider`.
- [ ] **An observable list,** `ObservableList` as in `pybricks_voice/windows/observable_list.py`:
      a Python list that implements WinUI's non-generic `IBindableVector` and
      `INotifyCollectionChanged`, as C#'s `ObservableCollection` does. PyWinRT can't
      implement `IObservableVector<T>`, being generic, so every app that shows a
      list would otherwise write this.
- [ ] **An event helper** for Python classes that implement a WinRT event, as
      C++/WinRT's `winrt::event` or C#'s `event` keyword: `Event[H]` with `add()`,
      `remove()` and calling it, as in `pybricks_voice/windows/winrt_event.py`, so no one
      writes the handler dict and tokens again.
- [ ] **Event tokens for implementers:** `EventRegistrationToken` can't be made in
      Python ("WinRT issues a token", say the stubs), but an implementation of
      `add_<event>()` has to issue one. Returning an `int` works, against the stubs.
      The event helper would hide it; the stubs and docs should allow it.
- [ ] **Keyword arguments for property setters** in generated constructors, as C#'s
      object initializers (`TextBlock(text="…", font_size=12)`), plus helpers for
      attached properties, so UI built in code is as short as XAML and typed.
- [ ] **A deferral helper,** something like `complete_after(deferral, coro)`, plus docs
      on splitting an event handler between the callback and the asyncio loop.
- [ ] **Docs and samples:** DPI awareness in the WinUI 3 samples, which get stretched on
      scaled displays.

### WinUI

- [ ] **A window quits silently** when its `Window.SystemBackdrop` (Mica or acrylic)
      comes before content holding an `ItemsRepeater` with an `ItemTemplate`, loaded
      with `XamlReader`: `Application.Start` returns as the window shows, with no
      error or `UnhandledException`. After the content it works, as does a
      `ListView`'s template. The smallest case is a `Window` holding just the
      backdrop, then an `ItemsRepeater` whose template is a `<Border/>`. Check it
      in C# and report it.

### wgpu-py

- [ ] **A supported way to present into a WinUI `SwapChainPanel`:** wgpu-native has
      `WGPUSurfaceSourceSwapChainPanel`, but wgpu-py only makes surfaces from window
      handles, so `windows/scene_view.py` wraps its `get_surface_id_from_info` (and pins wgpu).
      First of the three: rendercanvas's backend needs it.

### rendercanvas

- [ ] **A WinUI backend,** `rendercanvas.winui`, with PyWinRT as an optional
      dependency: `windows/scene_view.py`'s `SwapChainPanelCanvas` and most of `SceneView`, the
      parts every app would otherwise find again. The panel laid out at physical
      size and scaled back down, the display's scale followed through
      `XamlRoot.Changed`, the first frame drawn on the UI thread (`SetSwapChain`
      fails on any other), then drawing from a thread of its own or on demand,
      perhaps scheduled by the guest asyncio loop. Needs the wgpu-py item above.

### wgpu

- [ ] **The swap chain ignores the display scale:** wgpu's Direct3D 12 backend doesn't
      call `IDXGISwapChain2::SetMatrixTransform` with the inverse of the panel's
      composition scale, so `windows/scene_view.py` lays the panel out at physical size and
      scales it back down; rendercanvas's backend would have to as well.

### pygfx

- [ ] **Import FreeType only when text is used:** importing pygfx imports `freetype`,
      which needs a FreeType DLL even in apps that draw no text. Check its issues first.

### freetype-py

- [ ] **Windows ARM64 wheels:** [freetype-py#210](https://github.com/rougier/freetype-py/pull/210)
      adds them and fixes the project's broken CI, but is unreviewed. Until it's
      released, `tools/get_freetype.py` builds the DLL.

### MuJoCo

- [ ] **Windows ARM64 wheels:** neither PyPI nor the GitHub releases have them.
      `tools/get_mujoco.py` builds them, for this repository's mujoco-3.17.0
      release: the library with MSVC, the bindings with
      clang-cl, as MuJoCo's CI does for Windows. With MSVC the bindings fail to
      link, since MSVC gives the code for the unnamed structs inside `mjVisual`
      internal linkage in `structs_wrappers.cc`; a fix there would make MSVC
      builds possible too.
