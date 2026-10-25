"""
Spike: what the 3D view's GPU use is made of.

Renders the app's RobotScene into a texture of the view's size, 60 times a
second, with one thing turned off at a time, and reads this process's 3D
engine use from Windows' "GPU Engine" performance counter while it does.

    uv run python spikes/gpu_cost.py [--size 816x930] [--seconds 4]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pybricks_voice.physics import Physics
from pybricks_voice.scene import RobotScene, gfx
from pybricks_voice.windows.scene_view import use_d3d12


def gpu_percent(seconds: float, out: list[float]) -> None:
    """This process's 3D engine use, sampled once a second."""
    counter = rf"\GPU Engine(pid_{os.getpid()}_*engtype_3d)\Utilization Percentage"
    command = (
        f"(Get-Counter '{counter}' -SampleInterval 1 -MaxSamples {int(seconds)}).CounterSamples"
        " | Group-Object Timestamp | ForEach-Object { ($_.Group | Measure-Object CookedValue -Sum).Sum }"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)
    out.extend(float(line) for line in result.stdout.split() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="816x930")
    parser.add_argument("--seconds", type=float, default=4)
    options = parser.parse_args()
    width, height = map(int, options.size.split("x"))
    print("on", use_d3d12())

    scene = RobotScene()
    world = Physics().world()
    scene.pose(world)
    target = gfx.Texture(dim=2, size=(width, height, 1), format="rgba8unorm")
    renderer = gfx.WgpuRenderer(target)
    floor = [o for o in scene.scene.children if isinstance(o, gfx.Mesh) and o is not scene.robot]
    grids = [o for o in scene.scene.children if isinstance(o, gfx.Grid | gfx.Line) and o is not scene.trail]

    def measure(name: str, fps: float = 60) -> None:
        samples: list[float] = []
        sampler = threading.Thread(target=gpu_percent, args=(options.seconds, samples))
        sampler.start()
        frames, start = 0, time.perf_counter()
        while sampler.is_alive():
            scene.pose(world)
            renderer.render(scene.scene, scene.camera)
            frames += 1
            time.sleep(max(0.0, start + frames / fps - time.perf_counter()))
        rate = frames / (time.perf_counter() - start)
        gpu = statistics.mean(samples[1:]) if len(samples) > 1 else float("nan")
        print(f"{name:34} {gpu:5.1f}% GPU at {rate:4.0f} fps")

    measure("everything (as the app)")
    scene.sun.cast_shadow = False
    measure("no shadows")
    scene.sun.cast_shadow = True
    scene.robot.visible = False
    measure("no robot")
    scene.robot.visible = True
    for o in floor + grids:
        o.visible = False
    measure("no floor or grids")
    for o in floor + grids:
        o.visible = True
    measure("everything at 30 fps", fps=30)

    # The renderer's own passes: supersampling, its downsampling filter, and
    # post-process antialiasing.
    for ratio, filter_, ppaa in (
        (1, "mitchell", "ddaa"),
        (1, "linear", "none"),
        (0.75, "mitchell", "ddaa"),
        (0.5, "mitchell", "ddaa"),
        (0.5, "linear", "none"),
    ):
        renderer.pixel_ratio, renderer.pixel_filter, renderer.ppaa = ratio, filter_, ppaa
        measure(f"pixel_ratio {ratio}, {filter_}, ppaa {ppaa}")


if __name__ == "__main__":
    main()
