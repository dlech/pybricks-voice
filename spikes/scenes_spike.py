"""Spike: the robot in 3D with Composition Scenes instead of WebView2 and three.js.

A SceneVisual hosted in the window's root Grid draws meshes built here: the
floor, a grid of lines, the robot from boxes and cylinders, and its trail.
SimRobot drives it. The nodes are in world units, and the camera is one matrix
on the SceneVisual's TransformMatrix - view, projection, then a stretch to the
visual's size - as CommunityToolkit/SceneLoader's experimental camera does it.

    uv run --no-sync python tools/scenes_spike.py --seconds 8 --screenshot out/scenes.png

Needs winrt-Windows.Foundation.Numerics, which the app doesn't depend on yet:
    uv pip install winrt-Windows.Foundation.Numerics
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import datetime
import itertools
import math
import pathlib
import sys
import time

import numpy as np
from winrt.microsoft.graphics.directx import DirectXPixelFormat, DirectXPrimitiveTopology
from winrt.microsoft.ui.composition import Compositor, ContainerVisual
from winrt.microsoft.ui.composition.scenes import (
    SceneAttributeSemantic,
    SceneMesh,
    SceneMeshRendererComponent,
    SceneMetallicRoughnessMaterial,
    SceneNode,
    SceneVisual,
)
from winrt.microsoft.ui.xaml import Application, FrameworkElement, Window
from winrt.microsoft.ui.xaml.controls import Grid, XamlControlsResources
from winrt.microsoft.ui.xaml.hosting import ElementCompositionPreview
from winrt.microsoft.ui.xaml.markup import IXamlMetadataProvider
from winrt.microsoft.ui.xaml.xamltypeinfo import XamlControlsXamlMetaDataProvider
from winrt.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import InitializeOptions, initialize
from winrt.runtime import ApartmentType, init_apartment
from winrt.windows.foundation import MemoryBuffer
from winrt.windows.foundation.numerics import Matrix4x4

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pybricks_voice.grammar import parse
from pybricks_voice.robot import Curve, SimRobot, Straight, Turn

# World space is the simulator's, in centimeters and right-handed: x right, y
# forward, z up. Its heading is clockwise seen from above, which is a negative
# rotation about z.

# ----- meshes ---------------------------------------------------------------------


def _buffer(array: np.ndarray) -> MemoryBuffer:
    data = np.ascontiguousarray(array).tobytes()
    buffer = MemoryBuffer(len(data))
    with buffer.create_reference() as reference, memoryview(reference) as view:
        view[: len(data)] = data
    return buffer


class Mesh:
    """A SceneMesh and the MemoryBuffers it was filled from, kept alive with it."""

    def __init__(self, compositor: Compositor, positions, normals=None, indices=None, topology=None) -> None:
        self.mesh = SceneMesh.create(compositor)
        self.buffers = []
        self._fill(SceneAttributeSemantic.VERTEX, DirectXPixelFormat.R32_G32_B32_FLOAT, np.float32(positions))
        if normals is not None:
            self._fill(SceneAttributeSemantic.NORMAL, DirectXPixelFormat.R32_G32_B32_FLOAT, np.float32(normals))
        if indices is not None:
            self._fill(SceneAttributeSemantic.INDEX, DirectXPixelFormat.R32_UINT, np.uint32(indices))
        self.mesh.primitive_topology = topology or DirectXPrimitiveTopology.TRIANGLE_LIST

    def _fill(self, semantic, format, array) -> None:
        buffer = _buffer(array)
        self.buffers.append(buffer)
        self.mesh.fill_mesh_attribute(semantic, format, buffer)


def box(w: float, d: float, h: float):
    """A w (x) by d (y) by h (z) box, centered on the origin, with flat normals."""
    positions, normals, indices = [], [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sign
            u = np.roll(n, 1) if sign > 0 else np.roll(n, 2)
            v = np.cross(n, u)
            base = len(positions)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                positions.append((n + a * u + b * v) * (w / 2, d / 2, h / 2))
                normals.append(n)
            indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return positions, normals, indices


def cylinder_x(radius: float, width: float, segments: int = 32):
    """A cylinder along x, centered on the origin: a wheel."""
    positions, normals, indices = [], [], []
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    ring = np.stack([np.zeros(segments), np.cos(angles), np.sin(angles)], axis=1)
    for i in range(segments):
        j = (i + 1) % segments
        base = len(positions)
        for k in (i, j):
            for x in (-width / 2, width / 2):
                positions.append(ring[k] * radius + (x, 0, 0))
                normals.append(ring[k])
        indices += [base, base + 1, base + 3, base, base + 3, base + 2]
    for x, nx in ((-width / 2, -1.0), (width / 2, 1.0)):
        center = len(positions)
        positions.append((x, 0, 0))
        normals.append((nx, 0, 0))
        for k in range(segments):
            positions.append(ring[k] * radius + (x, 0, 0))
            normals.append((nx, 0, 0))
        for k in range(segments):
            a, b = center + 1 + k, center + 1 + (k + 1) % segments
            indices += [center, a, b] if nx > 0 else [center, b, a]
    return positions, normals, indices


def strips(segments, width: float, z: float):
    """Flat strips lying on the floor, one per segment ((x0, y0), (x1, y1)).
    Line meshes draw black whatever their material, so lines are drawn as
    triangles instead."""
    positions, normals, indices = [], [], []
    for (x0, y0), (x1, y1) in segments:
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length * width / 2, dx / length * width / 2
        base = len(positions)
        positions += [(x0 - nx, y0 - ny, z), (x1 - nx, y1 - ny, z), (x1 + nx, y1 + ny, z), (x0 + nx, y0 + ny, z)]
        normals += [(0, 0, 1)] * 4
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return positions, normals, indices


def grid(half: float, step: float):
    segments = []
    for t in np.arange(-half, half + step / 2, step):
        segments += [((t, -half), (t, half)), ((-half, t), (half, t))]
    return strips(segments, width=0.2, z=0.02)


# ----- the scene -------------------------------------------------------------------


def _srgb(hex_color: int) -> tuple[float, float, float, float]:
    """Base colors are linear; three.js took sRGB hex."""
    rgb = [((hex_color >> s) & 0xFF) / 255 for s in (16, 8, 0)]
    return (*[c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb], 1.0)


class Scene:
    def __init__(self, compositor: Compositor) -> None:
        self.compositor = compositor
        self.keep: list[Mesh] = []
        self.visual = SceneVisual.create(compositor)
        self.world = SceneNode.create(compositor)
        self.visual.root = self.world
        self.size = (1.0, 1.0)

        floor = SceneNode.create(compositor)
        floor.transform.translation = (0, 0, -0.5)
        self.add(floor, Mesh(compositor, *box(400, 400, 1)), 0x2A303B, roughness=0.95)
        self.world.children.append(floor)
        grid_node = SceneNode.create(compositor)
        self.world.children.append(self.add(grid_node, Mesh(compositor, *grid(200, 10)), 0x4A5263, roughness=0.95))

        # Position and heading on nodes of their own: a SceneModelTransform turns
        # about the origin after it translates, which would swing the robot
        # around the world origin instead of turning it in place.
        self.place = SceneNode.create(compositor)
        self.robot = SceneNode.create(compositor)
        self.world.children.append(self.place)
        self.place.children.append(self.robot)
        for size, at, color in (
            ((11.2, 14.4, 4.0), (0, -0.8, 4.4), 0xF2F2F2),  # white base
            ((8.8, 8.8, 4.8), (0, -1.6, 8.8), 0xFFCD03),  # yellow hub
            ((5.6, 4.0, 0.4), (0, -0.4, 11.4), 0x23262B),  # its screen
            ((2.4, 2.4, 2.0), (0, -6.5, 1.0), 0x00A3A6),  # caster
        ):
            self.robot.children.append(self.part(box(*size), at, color))
        self.wheels = []
        for side in (-1, 1):
            wheel = SceneNode.create(compositor)
            wheel.transform.translation = (side * 6.4, 0, 2.8)
            wheel.transform.rotation_axis = (1, 0, 0)
            self.add(wheel, Mesh(compositor, *cylinder_x(2.8, 1.4)), 0x23262B, roughness=0.8)
            wheel.children.append(self.part(box(1.6, 1.5, 3.6), (side * 0.2, 0, 0), 0xF2F2F2))
            self.robot.children.append(wheel)
            self.wheels.append(wheel)
        self.robot.children.append(self.part(box(1.6, 3.0, 0.6), (0, 6.0, 11.4), 0x4F8CFF))  # forward

        if "--probe" in sys.argv:  # orientation check: one box ahead, one to the right
            self.world.children.append(self.part(box(10, 10, 10), (0, 40, 5), 0xFFCD03))  # ahead
            self.world.children.append(self.part(box(6, 6, 6), (25, 0, 3), 0xCC2222))  # right
        self.trail_node = SceneNode.create(compositor)
        self.world.children.append(self.trail_node)
        self.trail: list[tuple[float, float, float]] = []
        self.trail_mesh: Mesh | None = None

    def add(self, node: SceneNode, mesh: Mesh, color: int, roughness: float = 0.5, unlit: bool = False) -> SceneNode:
        material = SceneMetallicRoughnessMaterial.create(self.compositor)
        if unlit:
            # Lines have no normals, so lighting leaves them black: glow instead.
            material.base_color_factor = (0, 0, 0, 1)
            material.emissive_factor = _srgb(color)[:3]
        else:
            material.base_color_factor = _srgb(color)
        material.metallic_factor = 0.0
        material.roughness_factor = roughness
        renderer = SceneMeshRendererComponent.create(self.compositor)
        renderer.mesh = mesh.mesh
        renderer.material = material
        node.components.append(renderer)
        self.keep.append(mesh)
        return node

    def part(self, mesh_data, at, color: int) -> SceneNode:
        node = SceneNode.create(self.compositor)
        node.transform.translation = at
        return self.add(node, Mesh(self.compositor, *mesh_data), color)

    def pose(self, x_mm: float, y_mm: float, heading: float, left: float, right: float) -> None:
        x, y = x_mm / 10, y_mm / 10
        self.place.transform.translation = (x, y, 0)
        self.robot.transform.rotation_axis = (0, 0, 1)
        self.robot.transform.rotation_angle_in_degrees = -heading
        # Rolling forward turns the top of a wheel toward +y: negative about x.
        self.wheels[0].transform.rotation_angle_in_degrees = -left
        self.wheels[1].transform.rotation_angle_in_degrees = -right
        # Chase camera: behind and above the robot, looking just ahead of it.
        h = math.radians(heading)
        forward = np.array([math.sin(h), math.cos(h), 0.0])
        robot = np.array([x, y, 0.0])
        self.visual.transform_matrix = camera_matrix(
            eye=robot - 60 * forward + (0, 0, 45), target=robot + 10 * forward + (0, 0, 4), size=self.size
        )
        if not self.trail or math.dist(self.trail[-1][:2], (x, y)) > 0.5:
            self.trail.append((x, y, 0.1))
            if len(self.trail) % 10 == 2 and "--no-trail" not in sys.argv:  # refilling every point would be wasteful
                self.update_trail()

    def update_trail(self) -> None:
        if len(self.trail) < 2:
            return
        points = [p[:2] for p in self.trail]
        mesh = Mesh(self.compositor, *strips(itertools.pairwise(points), width=0.4, z=0.05))
        self.trail_node.components.clear()
        self.add(self.trail_node, mesh, 0x4F8CFF)
        if self.trail_mesh is not None:
            self.keep.remove(self.trail_mesh)
        self.trail_mesh = mesh

    def layout(self, width: float, height: float) -> None:
        if "--size" in sys.argv:
            self.visual.size = (width, height)
        self.size = (width, height)


# Found by experiment, documented nowhere: the scene renderer flips y and z
# (a half turn about x, the glTF-to-Composition change of axes) before it
# applies the visual's TransformMatrix, so the matrix undoes that first.
_UNFLIP = np.eye(4)


def camera_matrix(eye, target, size, fov_y: float = 50.0, near: float = 1.0, far: float = 2000.0) -> Matrix4x4:
    """View, perspective projection and a stretch to the visual, in the
    row-vector convention Composition's matrices use (v' = v M)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    back = (eye - target) / np.linalg.norm(eye - target)  # the camera looks down -z
    right = np.cross((0.0, 0.0, 1.0), back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    view = np.eye(4)
    view[:3, 0], view[:3, 1], view[:3, 2] = right, up, back
    view[3, :3] = -right @ eye, -up @ eye, -back @ eye

    width, height = size
    f = 1 / math.tan(math.radians(fov_y) / 2)
    projection = np.zeros((4, 4))
    projection[0, 0] = f * height / width
    projection[1, 1] = f
    projection[2, 2] = (far - near) / -(far + near)
    projection[2, 3] = -1
    projection[3, 2] = (-2 * far * near) / -(far + near)

    # From [-1, 1] to the visual's pixels: y down the screen, and z toward the
    # viewer, as Composition has it, so that nearer is larger.
    stretch = np.diag((width / 2, -height / 2, 1 * (width + height) / 2, 1.0))
    stretch[3, :2] = width / 2, height / 2
    return Matrix4x4(*(_UNFLIP @ view @ projection @ stretch).flatten())


# ----- the window ------------------------------------------------------------------


class App(Application, IXamlMetadataProvider):
    def __init__(self, options: argparse.Namespace) -> None:
        super().__init__()
        self._provider = XamlControlsXamlMetaDataProvider()
        self.options = options
        self.robot = SimRobot()
        self.gaps: list[float] = []

    def _on_launched(self, args) -> None:
        self.resources.merged_dictionaries.append(XamlControlsResources())
        self.window = Window()
        self.window.title = "Composition Scenes spike"
        self.window.app_window.resize((1200, 800))
        self.host = Grid()
        self.window.content = self.host

        compositor = ElementCompositionPreview.get_element_visual(self.host).compositor
        self.container: ContainerVisual = compositor.create_container_visual()
        self.scene = Scene(compositor)
        self.container.children.insert_at_top(self.scene.visual)
        ElementCompositionPreview.set_element_child_visual(self.host, self.container)
        self.host.add_size_changed(lambda s, e: self.layout())

        if self.options.say:
            self.robot.run(parse(self.options.say).commands)
        elif not self.options.static:
            self.robot.run([Straight(300), Curve(300, -360), Turn(90), Straight(200)])
        self.started = self.last = time.perf_counter()
        self.timer = self.window.dispatcher_queue.create_timer()
        self.timer.interval = datetime.timedelta(milliseconds=16)
        self.timer.add_tick(lambda t, e: self.tick())
        self.timer.start()
        self.window.activate()

    def layout(self) -> None:
        element = self.host.as_(FrameworkElement)
        width, height = element.actual_width, element.actual_height
        self.container.size = (width, height)
        self.scene.layout(width, height)

    def tick(self) -> None:
        now = time.perf_counter()
        self.gaps.append(now - self.last)
        self.robot.step(min(now - self.last, 0.1))
        self.last = now
        s = self.robot.state()
        self.scene.pose(s.x, s.y, s.heading, s.left_angle, s.right_angle)
        if now - self.started >= self.options.seconds:
            self.timer.stop()
            if self.options.screenshot:
                self.screenshot(self.options.screenshot)
            gaps = np.array(self.gaps[30:]) * 1000
            print(
                f"frames: median {np.median(gaps):.1f} ms, p99 {np.percentile(gaps, 99):.1f} ms, "
                f"max {gaps.max():.1f} ms; {len(self.scene.trail)} trail points",
                flush=True,
            )
            self.exit()

    def screenshot(self, path: pathlib.Path) -> None:
        """The window's own content, whatever is in front of it."""
        from PIL import Image

        hwnd = ctypes.windll.user32.FindWindowW(None, self.window.title)
        user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(ctypes.c_void_p(hwnd), ctypes.byref(rect))
        width, height = rect.right - rect.left, rect.bottom - rect.top
        screen = user32.GetDC(None)
        dc = gdi32.CreateCompatibleDC(screen)
        bitmap = gdi32.CreateCompatibleBitmap(screen, width, height)
        gdi32.SelectObject(dc, bitmap)
        user32.PrintWindow(ctypes.c_void_p(hwnd), dc, 2)  # PW_RENDERFULLCONTENT
        pixels = ctypes.create_string_buffer(width * height * 4)
        header = (ctypes.c_uint32 * 10)(40, width, -height, 1 | (32 << 16), 0, 0, 0, 0, 0, 0)
        gdi32.GetDIBits(dc, bitmap, 0, height, pixels, header, 0)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(dc)
        user32.ReleaseDC(None, screen)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.frombuffer("RGB", (width, height), pixels, "raw", "BGRX").save(path)

    def get_xaml_type(self, type):
        return self._provider.get_xaml_type(type)

    def get_xaml_type_by_full_name(self, full_name):
        return self._provider.get_xaml_type_by_full_name(full_name)

    def get_xmlns_definitions(self):
        return self._provider.get_xmlns_definitions()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--screenshot", type=pathlib.Path)
    parser.add_argument("--probe", action="store_true", help="add a box ahead and one to the right")
    parser.add_argument("--static", action="store_true", help="leave the robot where it starts")
    parser.add_argument("--say", help="drive these commands instead of the demo")
    parser.add_argument("--no-trail", action="store_true")
    options = parser.parse_args()
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    init_apartment(ApartmentType.SINGLE_THREADED)
    with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
        Application.start(lambda _: App(options))


if __name__ == "__main__":
    main()
