import pathlib
import unittest

import numpy as np

from pybricks_voice.ldraw import LDU, MAIN_COLOR, Library

MODELS = pathlib.Path(__file__).resolve().parent.parent / "pybricks_voice" / "models"


def bounds(triangles: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(list(triangles.values())).reshape(-1, 3)
    return points.min(0), points.max(0)


class TestLibrary(unittest.TestCase):
    library = Library(MODELS / "ldraw")

    def test_colors(self) -> None:
        self.assertEqual(self.library.color(0).name, "Black")
        self.assertEqual(self.library.color(0x2FF8000).rgb, "#ff8000")  # a direct color

    def test_a_part_is_its_real_size(self) -> None:
        low, high = bounds(self.library.triangles("39367p01.dat"))
        self.assertAlmostEqual((high - low)[0] * LDU, 0.056, delta=0.0005)  # a 56 mm wheel

    def test_main_color_is_whoever_places_the_part(self) -> None:
        beam = "32278.dat"  # a Technic beam, all main color
        self.assertEqual(list(self.library.triangles(beam)), [MAIN_COLOR])
        self.assertEqual(list(self.library.triangles(beam, 0)), [0])

    def test_names_are_case_and_slash_blind(self) -> None:
        self.assertEqual(len(self.library.triangles("S\\39367S01.DAT")), len(self.library.triangles("s/39367s01.dat")))


class TestDrivingBase(unittest.TestCase):
    def test_stands_on_the_table_behind_its_axle(self) -> None:
        library = Library(MODELS / "ldraw")
        triangles = library.triangles(str(MODELS / "driving_base.ldr"))
        _low, high = bounds(triangles)
        self.assertAlmostEqual(high[1], 0, delta=1)  # its lowest point, the caster ball, on the table
        self.assertGreater(high[2], 150)  # most of it behind the axle (+z)
        self.assertTrue({0, 15, 26}.issubset(triangles))  # black beams, white motors, magenta frame


if __name__ == "__main__":
    unittest.main()
