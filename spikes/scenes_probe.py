"""
Probe Composition Scenes' conventions: one box in four scenes side by side,
z scaled +1 or -1, seen face-on or tilted 40 degrees about x.
"""

import ctypes
import datetime
import pathlib
import sys

from winrt.microsoft.ui.composition.scenes import (
    SceneMeshRendererComponent,
    SceneMetallicRoughnessMaterial,
    SceneNode,
    SceneVisual,
)
from winrt.microsoft.ui.xaml import Application, Window
from winrt.microsoft.ui.xaml.controls import Grid, XamlControlsResources
from winrt.microsoft.ui.xaml.hosting import ElementCompositionPreview
from winrt.microsoft.ui.xaml.markup import IXamlMetadataProvider
from winrt.microsoft.ui.xaml.xamltypeinfo import XamlControlsXamlMetaDataProvider
from winrt.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import initialize
from winrt.runtime import ApartmentType, init_apartment

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scenes_spike import Mesh, Scene, box

CASES = [(1, 0), (-1, 0), (1, 40), (-1, 40)]  # (z scale sign, tilt)


class App(Application, IXamlMetadataProvider):
    def __init__(self, out):
        super().__init__()
        self._provider = XamlControlsXamlMetaDataProvider()
        self.out = out

    def _on_launched(self, args):
        self.resources.merged_dictionaries.append(XamlControlsResources())
        self.window = Window()
        self.window.app_window.resize((1200, 500))
        host = Grid()
        self.window.content = host
        compositor = ElementCompositionPreview.get_element_visual(host).compositor
        container = compositor.create_container_visual()
        container.size = (1200, 440)
        backdrop = compositor.create_sprite_visual()
        backdrop.size = (1200, 440)

        backdrop.brush = compositor.create_color_brush((255, 230, 230, 230))
        container.children.insert_at_bottom(backdrop)
        self.keep = []
        for i, (z, tilt) in enumerate(CASES):
            scene = Scene.__new__(Scene)  # only its add()
            scene.compositor, scene.keep = compositor, self.keep
            visual = SceneVisual.create(compositor)
            visual.size = (300, 400)
            visual.offset = (i * 300, 0, 0)
            root = SceneNode.create(compositor)
            root.transform.translation = (150, 200, 0)
            root.transform.scale = (8, 8, 8 * z)
            root.transform.rotation_axis = (1, 0, 0)
            root.transform.rotation_angle_in_degrees = tilt
            visual.root = root
            # A flat red plate with a blue post standing up on +z.
            plate = SceneNode.create(compositor)
            scene.add(plate, Mesh(compositor, *box(20, 20, 1)), 0xCC3333)
            post = SceneNode.create(compositor)
            post.transform.translation = (0, 0, 5)
            scene.add(post, Mesh(compositor, *box(4, 4, 10)), 0x3366FF)
            for node in (plate, post):
                material = (
                    node.components.get_at(0)
                    .as_(SceneMeshRendererComponent)
                    .material.as_(SceneMetallicRoughnessMaterial)
                )
                material.emissive_factor = (0.3, 0.3, 0.3)
            root.children.append(plate)
            root.children.append(post)
            container.children.insert_at_top(visual)
        ElementCompositionPreview.set_element_child_visual(host, container)
        self.window.activate()
        timer = self.window.dispatcher_queue.create_timer()
        timer.interval = datetime.timedelta(seconds=2)
        timer.is_repeating = False
        timer.add_tick(lambda t, e: self.done())
        timer.start()
        self.timer = timer

    def done(self):
        from PIL import ImageGrab

        pos, size = self.window.app_window.position, self.window.app_window.size
        ImageGrab.grab((pos.x, pos.y, pos.x + size.width, pos.y + size.height), all_screens=True).save(self.out)
        self.exit()

    def get_xaml_type(self, t):
        return self._provider.get_xaml_type(t)

    def get_xaml_type_by_full_name(self, n):
        return self._provider.get_xaml_type_by_full_name(n)

    def get_xmlns_definitions(self):
        return self._provider.get_xmlns_definitions()


ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
init_apartment(ApartmentType.SINGLE_THREADED)
with initialize():
    Application.start(lambda _: App(sys.argv[1]))
