"""Run with the isolated MLX Python; regular CI skips on non-MLX hosts."""

import importlib.util
import unittest

HAS_MLX = importlib.util.find_spec("mlx") is not None


@unittest.skipUnless(HAS_MLX, "Requires isolated Apple Silicon MLX environment")
class ThinkingBoundaryTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx

        from examples.local_qwen.constrained_server import AfterThinking

        self.mx = mx
        self.calls = []

        def final_processor(tokens, logits):
            self.calls.append(len(tokens))
            return logits + 1

        self.gate = AfterThinking(final_processor, end_token=3, budget=2)
        self.logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

    def test_prefilled_prompt_does_not_extend_reasoning_budget(self):
        mx = self.mx
        self.gate(mx.array([1]), self.logits)
        unchanged = self.gate(mx.array([1, 2]), self.logits)
        self.assertEqual(unchanged.tolist(), self.logits.tolist())
        forced = self.gate(mx.array([1, 2, 2]), self.logits).tolist()[0]
        self.assertEqual(forced[3], 0)
        self.assertTrue(all(v == -float("inf") for i, v in enumerate(forced) if i != 3))
        self.assertEqual(self.calls, [])
        constrained = self.gate(mx.array([1, 2, 2, 3]), self.logits)
        self.assertEqual(constrained.tolist(), (self.logits + 1).tolist())
        self.gate(mx.array([1, 2, 2, 3, 2]), self.logits)
        self.assertEqual(self.calls, [4, 5])

    def test_natural_thinking_end_switches_immediately(self):
        self.gate(self.mx.array([1]), self.logits)
        self.gate(self.mx.array([1, 3]), self.logits)
        self.assertEqual(self.calls, [2])

    def test_initial_callback_can_include_multiple_prompt_tokens(self):
        self.gate(self.mx.array([1] * 10), self.logits)
        self.gate(self.mx.array([1] * 11), self.logits)
        forced = self.gate(self.mx.array([1] * 12), self.logits)
        self.assertEqual(forced.tolist()[0][3], 0)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
