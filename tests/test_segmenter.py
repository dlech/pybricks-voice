import unittest

import numpy as np

from pybricks_voice.speech import Segmenter, is_noise

RATE = 16000


def noise(seconds: float, db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(RATE * seconds)) * 10 ** (db / 20)).astype(np.float32)


def tone(seconds: float, db: float) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (np.sin(2 * np.pi * 220 * t) * 10 ** (db / 20) * np.sqrt(2)).astype(np.float32)


def feed_in_chunks(segmenter: Segmenter, audio: np.ndarray, chunk: int = 160) -> list[np.ndarray]:
    out = []
    for i in range(0, len(audio), chunk):
        out += segmenter.feed(audio[i : i + chunk])
    return out


class TestSegmenter(unittest.TestCase):
    def test_one_utterance_in_room_noise(self) -> None:
        audio = np.concatenate([noise(2, -55), noise(1.5, -55, 1) + tone(1.5, -25), noise(2, -55, 2)])
        utterances = feed_in_chunks(Segmenter(RATE), audio)
        self.assertEqual(len(utterances), 1)
        length = len(utterances[0]) / RATE
        # 1.5 s of speech, plus up to 300 ms of preroll and the 600 ms of trailing quiet.
        self.assertGreater(length, 1.5)
        self.assertLess(length, 2.5)

    def test_silence_and_noise_make_nothing(self) -> None:
        self.assertEqual(feed_in_chunks(Segmenter(RATE), noise(6, -45)), [])

    def test_a_click_is_too_short(self) -> None:
        audio = np.concatenate([noise(2, -55), tone(0.05, -10), noise(2, -55, 1)])
        self.assertEqual(feed_in_chunks(Segmenter(RATE), audio), [])

    def test_two_phrases_with_a_pause(self) -> None:
        audio = np.concatenate([noise(2, -55), tone(1, -25), noise(1.2, -55, 1), tone(1, -25), noise(2, -55, 2)])
        self.assertEqual(len(feed_in_chunks(Segmenter(RATE), audio)), 2)

    def test_a_short_breath_does_not_split(self) -> None:
        audio = np.concatenate([noise(2, -55), tone(1, -25), noise(0.3, -55, 1), tone(1, -25), noise(2, -55, 2)])
        self.assertEqual(len(feed_in_chunks(Segmenter(RATE), audio)), 1)

    def test_speech_ends_when_the_room_is_louder_than_it_learned(self) -> None:
        # What Windows' voice processing gives: near silence at first (so the
        # noise floor is learned very low), then speech, then ordinary room noise
        # at -48 dBFS that is louder than that floor. Speech must still end.
        audio = np.concatenate([noise(2, -65), tone(1.2, -25), noise(4, -48, 1)])
        utterances = feed_in_chunks(Segmenter(RATE), audio)
        self.assertEqual(len(utterances), 1)
        self.assertLess(len(utterances[0]) / RATE, 2.5)

    def test_quiet_room_noise_does_not_start_speech(self) -> None:
        audio = np.concatenate([noise(2, -65), noise(3, -46, 1), noise(2, -65, 2)])
        self.assertEqual(feed_in_chunks(Segmenter(RATE), audio), [])

    def test_endless_talking_is_cut(self) -> None:
        utterances = feed_in_chunks(Segmenter(RATE), np.concatenate([noise(2, -55), tone(30, -25)]))
        self.assertGreaterEqual(len(utterances), 2)
        self.assertTrue(all(len(u) / RATE <= Segmenter.MAX_S + 0.1 for u in utterances))


class TestNoise(unittest.TestCase):
    def test_room_noise_is_noise(self) -> None:
        for text in ["", "[BLANK_AUDIO]", "(clicking)", "*coughs*", "Um...", "Uh, um.", "Thank you.", "you"]:
            self.assertTrue(is_noise(text), text)

    def test_commands_are_not_noise(self) -> None:
        for text in ["Stop.", "Turn left.", "Um, forward 20.", "[BLANK_AUDIO] go back"]:
            self.assertFalse(is_noise(text), text)


if __name__ == "__main__":
    unittest.main()
