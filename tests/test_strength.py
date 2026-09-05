import unittest

from engine.strength import clamp, entropy_of, normalize5, strength_of


class TestStrength(unittest.TestCase):
    def test_normalize_clean(self):
        probs, note = normalize5([0.1, 0.2, 0.3, 0.2, 0.2])
        self.assertIsNone(note)
        self.assertEqual(probs, [0.1, 0.2, 0.3, 0.2, 0.2])

    def test_normalize_rescales(self):
        probs, note = normalize5([0.2, 0.4, 0.6, 0.4, 0.5])
        self.assertIsNotNone(note)
        self.assertAlmostEqual(sum(probs), 1.0, places=9)

    def test_normalize_zero_sum(self):
        probs, note = normalize5([0, 0, 0, 0, 0])
        self.assertEqual(probs, [1.0, 0, 0, 0, 0])
        self.assertIsNotNone(note)

    def test_strength_expectation(self):
        self.assertEqual(strength_of([0, 0, 1, 0, 0]), 2.0)
        self.assertAlmostEqual(strength_of([0.1] * 5), 1.0)  # 0.1*(0+1+2+3+4)
        self.assertEqual(strength_of([1, 0, 0, 0, 0]), 0.0)
        self.assertEqual(strength_of([0, 0, 0, 0, 1]), 4.0)

    def test_entropy(self):
        self.assertEqual(entropy_of([1, 0, 0, 0, 0]), 0.0)
        self.assertAlmostEqual(entropy_of([0.2] * 5), 2.322, places=2)

    def test_clamp(self):
        self.assertEqual(clamp(0.5, -1, 1), (0.5, None))
        self.assertEqual(clamp(1.4, -1, 1), (1.0, "clamped 1.4 to [-1,1]"))
        self.assertEqual(clamp(-2.0, 0, 1), (0.0, "clamped -2.0 to [0,1]"))


if __name__ == "__main__":
    unittest.main()
