import unittest

from algorithms import algorithm_a, algorithm_b


class AlgorithmsTest(unittest.TestCase):
    def test_correctness_across_sizes(self):
        for n in (1, 2, 17, 2000, 4000):
            self.assertEqual(algorithm_a(n)[0], sum(range(1, n + 1)))
            self.assertEqual(algorithm_b(n)[0], algorithm_a(n)[0])

    def test_operation_counts(self):
        self.assertEqual(algorithm_a(2000)[1], 2000)
        self.assertEqual(algorithm_b(2000)[1], 1)
