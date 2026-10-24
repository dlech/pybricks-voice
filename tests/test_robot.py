import math
import unittest

from pybricks_voice.robot import Drive, SetSpeed, SimRobot, Stop, Straight, Turn

DT = 1 / 120


def run_until_idle(robot: SimRobot, limit_s: float = 30) -> float:
    t = 0.0
    while robot.busy and t < limit_s:
        robot.step(DT)
        t += DT
    return t


class TestSimRobot(unittest.TestCase):
    def test_straight_ends_exactly_there(self) -> None:
        robot = SimRobot()
        robot.run([Straight(500)])
        run_until_idle(robot)
        s = robot.state()
        self.assertAlmostEqual(s.y, 500, places=3)
        self.assertAlmostEqual(s.x, 0, places=3)
        self.assertEqual(s.speed, 0)

    def test_backwards(self) -> None:
        robot = SimRobot()
        robot.run([Straight(-200)])
        run_until_idle(robot)
        self.assertAlmostEqual(robot.state().y, -200, places=3)

    def test_turn_right_is_clockwise(self) -> None:
        robot = SimRobot()
        robot.run([Turn(90), Straight(100)])
        run_until_idle(robot)
        s = robot.state()
        self.assertAlmostEqual(s.heading, 90, places=3)
        self.assertAlmostEqual(s.x, 100, places=2)
        self.assertAlmostEqual(s.y, 0, places=2)

    def test_square_comes_back(self) -> None:
        robot = SimRobot()
        robot.run([Straight(300), Turn(-90)] * 4)
        run_until_idle(robot)
        s = robot.state()
        self.assertAlmostEqual(s.x, 0, places=2)
        self.assertAlmostEqual(s.y, 0, places=2)
        self.assertAlmostEqual(s.heading, 0, places=3)

    def test_turning_in_place_turns_the_wheels_opposite_ways(self) -> None:
        robot = SimRobot()
        robot.run([Turn(360)])
        run_until_idle(robot)
        s = robot.state()
        # A full turn: each wheel rolls the circle of the axle track.
        expected = 112 / 56 * 360
        self.assertAlmostEqual(s.left_angle, expected, places=1)
        self.assertAlmostEqual(s.right_angle, -expected, places=1)

    def test_drive_for_a_time(self) -> None:
        robot = SimRobot()
        robot.run([Drive(100, 0, 2)])
        run_until_idle(robot)
        s = robot.state()
        # 2 s at up to 100 mm/s, with a short ramp up and the stop at the end.
        self.assertGreater(s.y, 150)
        self.assertLess(s.y, 220)

    def test_stop_clears_the_queue_and_brakes(self) -> None:
        robot = SimRobot()
        robot.run([Straight(1000), Turn(90)])
        for _ in range(60):
            robot.step(DT)
        robot.stop()
        run_until_idle(robot)
        s = robot.state()
        self.assertLess(s.y, 1000)
        self.assertEqual(s.heading, 0)
        self.assertFalse(robot.busy)

    def test_faster_takes_less_time(self) -> None:
        slow, fast = SimRobot(), SimRobot()
        slow.run([Straight(1000)])
        fast.run([SetSpeed(2), Straight(1000)])
        self.assertLess(run_until_idle(fast), run_until_idle(slow))

    def test_speed_never_exceeds_the_limit(self) -> None:
        robot = SimRobot()
        robot.run([Straight(2000)])
        top = 0.0
        while robot.busy:
            robot.step(DT)
            top = max(top, robot.state().speed)
        self.assertLessEqual(top, robot.limits.straight_speed + 1e-9)
        self.assertTrue(math.isclose(robot.state().y, 2000, abs_tol=1e-3))


if __name__ == "__main__":
    unittest.main()
