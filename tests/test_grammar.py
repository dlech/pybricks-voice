import unittest

from pybricks_voice.grammar import Conversation, describe, normalize, parse
from pybricks_voice.robot import Curve, Drive, Home, SetSpeed, Stop, Straight, Turn


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


class TestDescribe(unittest.TestCase):
    def test_plain_words(self) -> None:
        self.assertEqual(
            describe([Straight(500), Turn(90), Straight(-200)]), "Forward 50 cm, then turn right 90°, then back 20 cm"
        )
        self.assertEqual(describe([Turn(360)] * 3), "Spin right all the way around ×3")
        self.assertEqual(describe([Curve(600, -360)]), "Drive a circle to the left (60 cm radius)")
        self.assertEqual(describe([Curve(500, -30)]), "Curve left 30° (50 cm radius)")
        self.assertEqual(describe([Straight(300), Turn(90)] * 4), "4 × (forward 30 cm, then turn right 90°)")
        self.assertEqual(describe([Home()]), "Go back to the start")
        self.assertEqual(
            describe([Straight(100), *parse("zigzag").commands]),
            "Forward 10 cm, then zigzag forward to the right (20 cm legs)",
        )


class TestParse(unittest.TestCase):
    def check(self, text, *expected, not_understood=()) -> None:
        result = parse(text)
        self.assertEqual(result.commands, list(expected), text)
        self.assertEqual(result.not_understood, list(not_understood), text)

    def test_the_milestone_sentence(self) -> None:
        self.check(
            "Forward 50 centimeters, turn right 90 degrees, then back 20.", Straight(500), Turn(90), Straight(-200)
        )

    def test_distances_and_units(self) -> None:
        self.check("Go forward 30 cm.", Straight(300))
        self.check("Drive forward one meter.", Straight(1000))
        self.check("Move ahead 150 millimeters.", Straight(150))
        self.check("Back up 10 inches.", Straight(-254))
        self.check("Reverse half a meter.", Straight(-500))
        self.check("Forward 20.", Straight(200))  # no unit: centimeters
        self.check("Go forward.", Straight(200))

    def test_turns(self) -> None:
        self.check("Turn left.", Turn(-90))
        self.check("Turn right 45 degrees.", Turn(45))
        self.check("Turn left 180°.", Turn(-180))
        self.check("Turn around.", Turn(180))
        self.check("Spin all the way around to the left.", Turn(-360))
        self.check("Spin around.", Turn(360))
        self.check("Rotate right thirty degrees.", Turn(30))

    def test_circles(self) -> None:
        self.check("Drive in a circle.", Curve(300, 360))
        self.check("Make a big circle to the left.", Curve(600, -360))
        self.check("Do a small circle.", Curve(150, 360))
        self.check("Do a full circle to the right.", Curve(300, 360))
        self.check("Drive in a circle with a radius of 40 centimeters.", Curve(400, 360))
        self.check("Spin in a circle.", Turn(360))
        self.check("Drive half a circle to the left.", Curve(300, -180))
        self.check("Drive a half of a circle to the left.", Curve(300, -180))
        self.check("Do a quarter circle.", Curve(300, 90))
        self.check("Do a figure eight.", Curve(300, 360), Curve(300, -360))
        self.check("Figure 8 to the left.", Curve(300, -360), Curve(300, 360))

    def test_curves(self) -> None:
        self.check("Curve left.", Curve(500, -90))
        self.check("Arc right 45 degrees.", Curve(500, 45))
        self.check("Veer to the left.", Curve(500, -30))
        self.check("Vier to the left.", Curve(500, -30))  # Whisper's spelling
        self.check("Make a wide curve to the right.", Curve(800, 90))

    def test_driving_while_turning_is_one_arc(self) -> None:
        self.check("No, like drive and turn left at the same time.", Curve(500, -90))
        self.check("Turn right while driving.", Curve(500, 90))

    def test_repeats(self) -> None:
        self.check("Spin around three times.", Turn(360), Turn(360), Turn(360))
        self.check("Spin around twice.", Turn(360), Turn(360))
        self.check("Drive in a circle 2 times, then stop.", Curve(300, 360), Curve(300, 360), Stop())

    def test_chatter_doesnt_move_the_robot(self) -> None:
        self.check("Okay.")
        self.check("Hello? Oh, there we go.")
        self.check("Spin around three times.", Turn(360), Turn(360), Turn(360))
        self.check("It's been around three times.", Turn(360), Turn(360), Turn(360))

    def test_your_third_session(self) -> None:
        self.check("Drive a quarter of a circle.", Curve(300, 90))
        self.check("Drive two units.", not_understood=["drive 2 units"])
        self.check("Drive in a very small circle.", Curve(150, 360))
        self.check("drive in a two centimeter circle.", Curve(60, 360))  # as tight as it can
        self.check("drive in a 1 meter circle to the left.", Curve(500, -360))
        self.check("Drive backwards in a circle.", Curve(-300, 360))
        self.check("Curve left backwards.", Curve(-500, -90))
        self.check("go back to the starting point.", Home())
        self.check("Go home.", Home())
        self.check("Drive back to where you started.", Home())
        self.check("Return to the start.", Home())
        self.check("Start driving forward.", Straight(200))
        self.check("Oh no.")
        self.check("Oh my")

    def test_shapes(self) -> None:
        self.check("driving a square.", *[Straight(300), Turn(90)] * 4)
        self.check("driving a triangle.", *[Straight(300), Turn(120)] * 3)
        self.check("Drive a small square to the left.", *[Straight(150), Turn(-90)] * 4)
        self.check("Drive a 50 centimeter hexagon.", *[Straight(500), Turn(60)] * 6)

    def test_praise_gets_a_thank_you(self) -> None:
        result = parse("Oh you're such a good robot.")
        self.assertEqual((result.commands, result.not_understood, result.remarks), ([], [], ["Thanks! 😊"]))

    def test_too_far(self) -> None:
        self.check("Go forward three miles.", not_understood=["go forward 3 miles"])

    def test_timed(self) -> None:
        self.check("Drive forward for 2 seconds.", Drive(150, 0, 2))
        self.check("Spin left for three seconds.", Drive(0, -90, 3))

    def test_speed_and_stop(self) -> None:
        self.check("Stop!", Stop())
        self.check("Faster.", SetSpeed(1.5))
        self.check("Slow down a bit.", SetSpeed(1 / 1.5))

    def test_chains(self) -> None:
        self.check("Forward 1 meter and then turn left and then forward 50.", Straight(1000), Turn(-90), Straight(500))
        self.check("Turn right. Forward 10 centimeters. Turn left.", Turn(90), Straight(100), Turn(-90))

    def test_not_understood_is_reported(self) -> None:
        self.check("Forward 20, make me a sandwich", Straight(200), not_understood=["make me a sandwich"])
        self.check("What a nice day", not_understood=["what a nice day"])

    def test_a_breath_before_the_number(self) -> None:
        self.check("Right, 90.", Turn(90))
        self.check("Left, 45.", Turn(-45))
        self.check("Turn left, then forward.", Turn(-90), Straight(200))

    def test_forward_march(self) -> None:
        self.check("Forward march!", Straight(200))
        self.check("March.")  # the second half of "Forward." … "March."

    def test_backward_shapes(self) -> None:
        self.check("Make a square while driving backwards.", *[Straight(-300), Turn(90)] * 4)
        self.check("A backwards triangle to the left.", *[Straight(-300), Turn(-120)] * 3)

    def test_zigzag(self) -> None:
        legs = [Straight(200), Turn(-90), Straight(200), Turn(90), Straight(200), Turn(-90), Straight(200)]
        self.check("Drive in a zigzag.", Turn(45), *legs, Turn(45))
        result = parse("zigzag to the left").commands
        self.assertEqual(sum(c.angle for c in result if isinstance(c, Turn)), 0)  # ends the way it started

    def test_how_whisper_spells_zigzag(self) -> None:
        for text in ("Zig Zag", "zig-zag", "non-sigzag.", "zigzag"):
            self.assertEqual(describe(parse(text).commands), "Zigzag forward to the right (20 cm legs)", text)

    def test_part_of_a_shape(self) -> None:
        self.check("Half a hexagon.", *[Straight(300), Turn(60)] * 3)
        self.check("Half a square backwards.", *[Straight(-300), Turn(90)] * 2)

    def test_wiggle(self) -> None:
        self.check("Shake it.", Turn(-20), Turn(40), Turn(-40), Turn(20))
        self.assertEqual(describe(parse("do a little dance then forward").commands), "Wiggle, then forward 20 cm")

    def test_more_praise(self) -> None:
        self.assertEqual(parse("Such a good little robot.").remarks, ["Thanks! 😊"])
        self.check("Womp womp.")

    def test_back_and_forth(self) -> None:
        self.check("Go back and forth.", Straight(200), Straight(-200))
        self.check("Forward and back 30 cm.", Straight(300), Straight(-300))
        self.check("Welcome back and forth.", Straight(200), Straight(-200))  # Whisper's "Go back and forth"

    def test_talk_about_the_robot_doesnt_move_it(self) -> None:
        self.check("You don't go backwards.")
        self.check("Why can't you draw a backward circle?")
        self.check("You just can't do a backward square.")
        self.check("It disappeared and then came back.")
        self.check("I mean the drawing doesn't always show up while you're driving.")
        self.check("Don't turn left.")
        self.check("Can you drive in a circle?", Curve(300, 360))

    def test_here_we_go(self) -> None:
        self.check("Here we go, forward 10.", Straight(100))


class TestConversation(unittest.TestCase):
    def test_a_request_cut_in_two_by_a_pause(self) -> None:
        talk = Conversation()
        self.assertTrue(talk.parse("Do a", 0).waiting)
        self.assertEqual(talk.parse("100 centimeter circle.", 2).commands, [Curve(500, 360)])
        self.assertTrue(talk.parse("Do a figure.", 10).waiting)
        self.assertEqual(talk.parse("Figure eight.", 13).commands, [Curve(300, 360), Curve(300, -360)])
        self.assertTrue(talk.parse("Give me that zig!", 20).waiting)
        self.assertEqual(describe(talk.parse("Zag.", 22).commands), "Zigzag forward to the right (20 cm legs)")

    def test_only_what_comes_soon_after_is_joined(self) -> None:
        talk = Conversation()
        talk.parse("Do a", 0)
        result = talk.parse("left", 60)
        self.assertEqual((result.commands, result.waiting), ([Turn(-90)], False))

    def test_a_whole_request_is_not_held(self) -> None:
        talk = Conversation()
        self.assertEqual(talk.parse("Turn left.", 0).commands, [Turn(-90)])
        self.assertEqual(talk.parse("Forward.", 1).commands, [Straight(200)])
        self.assertFalse(talk.parse("Hello? Oh, there we go.", 2).waiting)
        self.assertEqual(talk.parse("Drive in a circle.", 3).not_understood, [])


if __name__ == "__main__":
    unittest.main()
