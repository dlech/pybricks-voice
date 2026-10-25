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
- [ ] **The 3D view looks slightly soft,** probably pygfx's antialiasing filter. Try
      its pixel filter settings.
- [ ] **Screenshot device:** replace the ctypes `D3D11CreateDevice` in `screenshot.py`
      with `create_direct3d11_device()` once PyWinRT releases it (see below).
- [ ] **Physics:** simulate the robot with MuJoCo (rigid bodies, wheel slip, pushing
      things), in progress. MuJoCo has no Windows ARM64 wheels, so it is built from
      source.
- [ ] **Milestone 2:** drive a real hub over Bluetooth LE with PyWinRT.

## Upstream

### PyWinRT

- [ ] **`create_direct3d11_device()`** in `winrt-Windows.Graphics.DirectX.Direct3D11.Interop`:
      branch `d3d11-create-device` is pushed and its PR is not open yet. See
      [pywinrt#78](https://github.com/pywinrt/pywinrt/issues/78), where it was asked for.
- [ ] **The guest asyncio loop** (`pybricks_voice/loop.py`) in `winrt-runtime`, with its
      tests.
- [ ] **A WinUI startup helper** for unpackaged apps: DPI awareness (`python.exe`'s
      manifest declares none), AppUserModelID, apartment and bootstrap, and the
      `IXamlMetadataProvider` boilerplate.
- [ ] **A deferral helper,** something like `complete_after(deferral, coro)`, plus docs
      on splitting an event handler between the callback and the asyncio loop.
- [ ] **Docs and samples:** DPI awareness in the WinUI 3 samples, which get stretched on
      scaled displays.

### wgpu-py

- [ ] **A supported way to present into a WinUI `SwapChainPanel`:** wgpu-native has
      `WGPUSurfaceSourceSwapChainPanel`, but wgpu-py only makes surfaces from window
      handles, so `scene.py` wraps its `get_surface_id_from_info` (and pins wgpu).
- [ ] **The swap chain ignores the display scale:** wgpu doesn't call
      `IDXGISwapChain2::SetMatrixTransform`, so `scene.py` lays the panel out at
      physical size and scales it back down.

### pygfx

- [ ] **Import FreeType only when text is used:** importing pygfx imports `freetype`,
      which needs a FreeType DLL even in apps that draw no text. Check its issues first.

### freetype-py

- [ ] **Windows ARM64 wheels:** [freetype-py#210](https://github.com/rougier/freetype-py/pull/210)
      adds them and fixes the project's broken CI, but is unreviewed. Until it's
      released, `tools/get_freetype.py` builds the DLL.

### MuJoCo

- [ ] **Windows ARM64 wheels:** neither PyPI nor the GitHub releases have them, though
      the library and the Python bindings build fine from source here.
