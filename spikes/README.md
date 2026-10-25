# Composition Scenes spike

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
