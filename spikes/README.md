# Spikes

## Composition Scenes

`scenes_spike.py` draws the robot with `Microsoft.UI.Composition.Scenes`, the
Windows App SDK's own 3D, instead of pygfx. It was set aside for pygfx in a
`SwapChainPanel` (see `pybricks_voice/scene.py`). What it found, none of it in
Microsoft's documentation:

- **No camera, and no control over lighting:** a `SceneVisual` takes a whole
  view-projection-viewport matrix as its `TransformMatrix`, as
  [CommunityToolkit/SceneLoader](https://github.com/CommunityToolkit/SceneLoader)'s
  experimental camera shows. The nodes stay in world units. There is some
  lighting, but no light object to control it.
- **Axes:** the renderer flips at least y between the scene and the visual
  (glTF's y-up against Composition's y-down). The spike's correction is only
  right near the origin; away from it, depth still feeds into screen position.
  That's where it was left.
- **Line meshes render black,** whatever their material, emission included.
  Lines are drawn as thin strips of triangles instead.
- **Vertex colours:** the `COLOR` attribute refused every format tried (RGBA
  float, RGB float, RGBA 8-bit).
- **Transforms:** a `SceneModelTransform` rotates about the origin *after* it
  translates, so position and heading need nodes of their own.
- **Logging:** the renderer writes debug output to the console (device setup,
  "Attempting to compute tangents" for every mesh without UVs).

`scenes_probe.py` puts four small scenes side by side to test conventions.

Both need `winrt-Windows.Foundation.Numerics`, which the app doesn't depend on:

```
uv pip install winrt-Windows.Foundation.Numerics
uv run --no-sync python spikes/scenes_spike.py --seconds 8 --screenshot out/scenes.png
```

## Drag stall

`drag_stall.py` finds out why the guest asyncio loop (`pybricks_voice/loop.py`)
stalls for 150-350 ms each time a window drag starts. A WinUI window ticks every
16 ms and logs every gap over 40 ms with what the UI thread did meanwhile: the
loop's work items and timer, each message it takes from its queue or is sent
(thread hooks), and a native stack from `cdb` once it has stalled for 150 ms.

- **The stall is in `DefWindowProc`:** a press on the title bar or a border
  sends `WM_SYSCOMMAND` `SC_MOVE` (or `SC_SIZE`), which waits in the kernel for
  the mouse to move before it starts the move loop, as long as the mouse is
  held still. Nothing is dispatched meanwhile, so no code on the UI thread can
  run, the guest loop's included. Inside the move loop it runs at full rate.
- **`--nudge`** posts a `WM_MOUSEMOVE` on `WM_NCLBUTTONDOWN`, which ends the
  wait at once. Double-clicking the title bar still maximizes and restores, but
  every click now starts and ends a move loop. The app uses a render thread
  instead.
- **The `DispatcherQueueTimer` is coarse:** a 16 ms timer often fires after 31
  or 47 ms, on the system's ~15.6 ms tick.

```
uv run python spikes/drag_stall.py [--nudge] [--no-cdb] [--seconds 60]
```

The report and stacks go to `out/drag_stall/`. `cdb` is the Windows SDK's
(Debugging Tools for Windows); its symbols come from Microsoft's server.
