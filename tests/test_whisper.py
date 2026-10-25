import unittest

import numpy as np

from pybricks_voice.whisper import EOT, FIRST_TIMESTAMP, PROMPT, decode_greedy

VOCABULARY = FIRST_TIMESTAMP + 1500


def logits_for(token: int) -> np.ndarray:
    logits = np.zeros(VOCABULARY, np.float16)
    logits[token] = 1.0
    return logits


class TestDecodeGreedy(unittest.TestCase):
    def test_prompt_then_tokens_until_end_of_text(self) -> None:
        fed: list[tuple[int, int]] = []
        answers = [logits_for(7)] * (len(PROMPT) - 1) + [logits_for(t) for t in (100, 200, EOT)]

        def step(token: int, position: int) -> np.ndarray:
            fed.append((token, position))
            return answers[position]

        tokens, steps = decode_greedy(step, max_len=224)
        # The prompt's own predictions (7) are ignored; each token is fed back.
        self.assertEqual(tokens, [100, 200])
        self.assertEqual(fed, [(t, n) for n, t in enumerate([*PROMPT, 100, 200])])
        self.assertEqual(steps, len(PROMPT) + 2)

    def test_timestamps_are_never_chosen(self) -> None:
        def step(token: int, position: int) -> np.ndarray:
            logits = logits_for(EOT if position >= len(PROMPT) else 300)
            logits[FIRST_TIMESTAMP + 5] = 10.0  # louder than anything, but a timestamp
            return logits

        tokens, _ = decode_greedy(step, max_len=224)
        self.assertEqual(tokens, [300])

    def test_stops_at_max_tokens_and_the_cache_length(self) -> None:
        def step(token: int, position: int) -> np.ndarray:
            return logits_for(42)

        self.assertEqual(len(decode_greedy(step, max_len=224, max_tokens=5)[0]), 5)
        # max_len - 1 steps at most: the prompt's 4 and 3 more.
        self.assertEqual(decode_greedy(step, max_len=8)[1], 7)


if __name__ == "__main__":
    unittest.main()
