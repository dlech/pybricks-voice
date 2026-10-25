import unittest

import numpy as np

from pybricks_voice.physics import THINGS, Physics
from pybricks_voice.robot import Command, SimRobot, Straight, Turn


def drive(physics: Physics, commands: list[Command], seconds: float) -> SimRobot:
    """Drives as the app does: SimRobot's reference wheel angles, 60 times a second."""
    robot = SimRobot()
    robot.run(commands)
    dt = 1 / 60
    for _ in range(round(seconds / dt)):
        robot.step(dt)
        s = robot.state()
        physics.step(dt, s.left_angle, s.right_angle)
    return robot


class TestPhysics(unittest.TestCase):
    def test_drives_about_as_far_as_told(self) -> None:
        physics = Physics()
        drive(physics, [Straight(200)], 3)
        x, y, z = physics.robot_pose().position * 1000
        self.assertAlmostEqual(y, 200, delta=6)  # within 3%, as a real robot would be
        self.assertAlmostEqual(x, 0, delta=2)
        self.assertLess(abs(z), 2)  # on its wheels, not tipped over

    def test_turns_about_as_far_as_told(self) -> None:
        physics = Physics()
        drive(physics, [Turn(90)], 3)
        self.assertAlmostEqual(physics.heading(), 90, delta=3)
        self.assertLess(np.hypot(*physics.robot_pose().position[:2]) * 1000, 5)  # on the spot

    def test_wheels_follow_their_motors(self) -> None:
        physics = Physics()
        robot = drive(physics, [Straight(100)], 2)
        s = robot.state()
        for actual, reference in zip(physics.wheel_angles(), (s.left_angle, s.right_angle), strict=True):
            self.assertAlmostEqual(actual, reference, delta=2)

    def test_pushes_what_is_in_the_way(self) -> None:
        physics = Physics()
        red = next(thing for thing in THINGS if thing.name == "red block")
        start = np.array(red.at)
        drive(physics, [Straight(500)], 4)
        moved = physics.pose(physics.things["red block"]).position[:2] - start
        self.assertGreater(moved[1], 0.1)  # pushed forward, more than 10 cm

    def test_reset(self) -> None:
        physics = Physics()
        drive(physics, [Straight(300)], 3)
        physics.reset()
        world = physics.world()
        self.assertLess(np.abs(world.robot.position[:2]).max(), 1e-3)
        self.assertEqual(world.heading, 0.0)


if __name__ == "__main__":
    unittest.main()
