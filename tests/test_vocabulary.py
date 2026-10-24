import json
import unittest

from pybricks_voice.whisper import EOT, MODEL, Vocabulary


@unittest.skipUnless((MODEL / "vocab.json").exists(), "run tools/get_models.py first")
class TestVocabulary(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabulary = Vocabulary(MODEL / "vocab.json")
        cls.ids = json.loads((MODEL / "vocab.json").read_text(encoding="utf-8"))

    def test_words_with_leading_spaces(self) -> None:
        # Ġ is GPT-2's stand-in for a space byte.
        ids = [self.ids["ĠTurn"], self.ids["Ġright"], self.ids["Ġ90"], self.ids["Ġdegrees"], self.ids["."]]
        self.assertEqual(self.vocabulary.decode(ids), " Turn right 90 degrees.")

    def test_multibyte_characters_survive(self) -> None:
        # "°" is the two UTF-8 bytes C2 B0, written "Â°" in the byte-level alphabet,
        # and may come as one token or as one token per byte.
        whole = [self.ids["Ġ90"], self.ids["Â°"]]
        split = [self.ids["Ġ90"], self.ids["Â"], self.ids["°"]]
        self.assertEqual(self.vocabulary.decode(whole), " 90°")
        self.assertEqual(self.vocabulary.decode(split), " 90°")

    def test_special_tokens_are_dropped(self) -> None:
        self.assertEqual(self.vocabulary.decode([EOT, self.ids["Ġstop"], EOT + 1]), " stop")


if __name__ == "__main__":
    unittest.main()
