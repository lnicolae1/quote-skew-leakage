# test_fundamental_value.py: round-trip volatility test for fundamental_value.py

import math
import unittest

from fundamental_value import generate_value_path, SECONDS_PER_YEAR


def measured_annualized_sigma(path, dt_seconds):
    """Back out annualized volatility from a value path, EXACTLY the way"""
    rets = []
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        rets.append(math.log(p1 / p0))

    n = len(rets)
    mean_r = sum(rets) / n
    var_r = sum((r - mean_r) ** 2 for r in rets) / n
    std_r = math.sqrt(var_r)

    steps_per_year = SECONDS_PER_YEAR / dt_seconds
    return std_r * math.sqrt(steps_per_year)


class TestFundamentalValue(unittest.TestCase):

    def _roundtrip(self, sigma_in, seed):
        dt_seconds = 1.0
        n_steps = 3_000_000
        start_price = 62000.0

        path = generate_value_path(sigma_in, start_price, dt_seconds, n_steps, seed)

        self.assertEqual(len(path), n_steps + 1)

        sigma_out = measured_annualized_sigma(path, dt_seconds)
        rel_err = abs(sigma_out - sigma_in) / sigma_in
        print("    sigma_in=%.4f  sigma_out=%.4f  rel_err=%.4f%%"
              % (sigma_in, sigma_out, rel_err * 100))
        self.assertLess(rel_err, 0.02)
        return path

    def test_roundtrip_at_0_45(self):
        self._roundtrip(0.45, seed=12345)

    def test_roundtrip_at_calibrated_0_3721(self):
        self._roundtrip(0.3721, seed=6789)

    def test_deterministic_given_seed(self):
        a = generate_value_path(0.4, 100.0, 1.0, 10000, seed=99)
        b = generate_value_path(0.4, 100.0, 1.0, 10000, seed=99)
        c = generate_value_path(0.4, 100.0, 1.0, 10000, seed=100)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
