import unittest

from pybricks_voice.grammar import normalize, parse
from pybricks_voice.robot import Drive, SetSpeed, Stop, Straight, Turn


class TestNumbers(unittest.TestCase):
    def test_words(self) -> None:
        self.assertEqual(normalize("twenty five"), "25")
        self.assertEqual(normalize("one hundred and twenty"), "120")
        self.assertEqual(normalize("a hundred"), "100")
        self.assertEqual(normalize("ninety degrees"), "90 degrees")
        self.assertEqual(normalize("half a meter"), "0.5 a meter")

    def test_whisper_styles(self) -> None:
        self.assertEqual(normalize("Forward 20cm."), "forward 20 cm .")
        self.assertEqual(normalize("Turn 90°"), "turn 90 degrees")
        self.assertEqual(normalize("1,000 millimeters"), "1000 millimeters")
        self.assertEqual(normalize("1.5 meters"), "1.5 meters")


class TestParse(unittest.TestCase):
    def check(self, text, *expected, not_understood=()) -> None:
        result = parse(text)
        self.assertEqual(result.commands, list(expected), text)
        self.assertEqual(result.not_understood, list(not_understood), text)

    def test_the_milestone_sentence(self) -> None:
        self.check("Forward 50 centimeters, turn right 90 degrees, then back 20.",
                   Straight(500), Turn(90), Straight(-200))

    def test_distances_and_units(self) -> None:
        self.check("Go forward 30 cm.", Straight(300))
        self.check("Drive forward one meter.", Straight(1000))
        self.check("Move ahead 150 millimeters.", Straight(150))
        self.check("Back up 10 inches.", Straight(-254))
        self.check("Reverse half a meter.", Straight(-500))
        self.check("Forward 20.", Straight(200))  # no unit: centimetres
        self.check("Go forward.", Straight(200))

    def test_turns(self) -> None:
        self.check("Turn left.", Turn(-90))
        self.check("Turn right 45 degrees.", Turn(45))
        self.check("Turn left 180°.", Turn(-180))
        self.check("Turn around.", Turn(180))
        self.check("Spin all the way around to the left.", Turn(-180))
        self.check("Do a full circle to the right.", Turn(360))
        self.check("Rotate right thirty degrees.", Turn(30))

    def test_timed(self) -> None:
        self.check("Drive forward for 2 seconds.", Drive(150, 0, 2))
        self.check("Spin left for three seconds.", Drive(0, -90, 3))

    def test_speed_and_stop(self) -> None:
        self.check("Stop!", Stop())
        self.check("Faster.", SetSpeed(1.5))
        self.check("Slow down a bit.", SetSpeed(1 / 1.5))

    def test_chains(self) -> None:
        self.check("Forward 1 meter and then turn left and then forward 50.",
                   Straight(1000), Turn(-90), Straight(500))
        self.check("Turn right. Forward 10 centimeters. Turn left.",
                   Turn(90), Straight(100), Turn(-90))

    def test_not_understood_is_reported(self) -> None:
        self.check("Forward 20, do a little dance", Straight(200), not_understood=["do a little dance"])
        self.check("What a nice day", not_understood=["what a nice day"])


if __name__ == "__main__":
    unittest.main()
