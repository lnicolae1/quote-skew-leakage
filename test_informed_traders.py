# test_informed_traders.py: unit tests for informed_traders.py (fixed seeds, wide tolerances)

import statistics
import unittest

from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow, DEFAULT_LAMBDA as LAM_NOISE
from informed_traders import (InformedTraderFlow, TOXICITY_PRESETS,
                              DEFAULT_LAMBDA_INFORMED, DEFAULT_THRESHOLD,
                              DEFAULT_SIZE, constant_value, path_value)


def seeded_book(mid=62000.0, tick=0.01, levels=20, size=0.5):
    """Engine with `levels` resting orders per side."""
    eng = MatchingEngine()
    for i in range(levels):
        eng.add_limit_order("buy", round(mid - (i + 1) * tick, 2), size, "seed")
        eng.add_limit_order("sell", round(mid + (i + 1) * tick, 2), size, "seed")
    return eng


class TestInformedTraders(unittest.TestCase):

    # 1: directional correctness, both ways
    def test_1_buys_underpriced_ask(self):
        eng = seeded_book(mid=62000.0)
        flow = InformedTraderFlow(seed=1)
        v = 62100.0

        rec = flow.step(eng, v)
        print("    V=%.2f  best_ask=%.2f -> side=%r  edge=%.2f"
              % (v, rec.best_ask, rec.side, rec.edge))

        self.assertEqual(rec.side, "buy")
        self.assertTrue(rec.fills)
        self.assertAlmostEqual(sum(f.size for f in rec.fills), DEFAULT_SIZE)
        for f in rec.fills:
            self.assertLess(f.price, v)

    def test_1b_sells_overpriced_bid(self):
        eng = seeded_book(mid=62000.0)
        flow = InformedTraderFlow(seed=1)
        v = 61900.0

        rec = flow.step(eng, v)
        print("    V=%.2f  best_bid=%.2f -> side=%r  edge=%.2f"
              % (v, rec.best_bid, rec.side, rec.edge))

        self.assertEqual(rec.side, "sell")
        self.assertTrue(rec.fills)
        for f in rec.fills:
            self.assertGreater(f.price, v)

    # 2: no-trade zone
    def test_2_no_trade_when_fairly_priced(self):
        eng = seeded_book(mid=62000.0)
        flow = InformedTraderFlow(seed=2)
        rec = flow.step(eng, 62000.0)

        print("    V=62000.00 inside book [%.2f, %.2f] -> side=%r"
              % (rec.best_bid, rec.best_ask, rec.side))
        self.assertIsNone(rec.side)
        self.assertEqual(rec.fills, [])
        self.assertEqual(flow.n_arrivals, 1)
        self.assertEqual(flow.n_trades, 0)
        self.assertEqual(eng.best_bid(), 61999.99)
        self.assertEqual(eng.best_ask(), 62000.01)

    def test_2b_threshold_boundary_is_respected(self):
        # trade iff V > 62000.06
        flow = InformedTraderFlow(seed=3)
        self.assertAlmostEqual(flow.threshold, DEFAULT_THRESHOLD)

        side, _ = flow.decide(61999.99, 62000.01, 62000.05)
        print("    V=62000.05 (edge 0.04 < threshold 0.05) -> %r" % (side,))
        self.assertIsNone(side)

        side, edge = flow.decide(61999.99, 62000.01, 62000.07)
        print("    V=62000.07 (edge 0.06 > threshold 0.05) -> %r edge=%.4f"
              % (side, edge))
        self.assertEqual(side, "buy")

    # 3: adverse-selection direction, randomized
    def test_3_always_trades_toward_value(self):
        import random as _random
        rng = _random.Random(4242)
        flow = InformedTraderFlow(seed=4)

        n_buy = n_sell = n_none = 0
        for _ in range(20000):
            mid = 62000.0
            half_spread = rng.choice([0.01, 0.05, 0.50, 2.00])
            bb = mid - half_spread
            ba = mid + half_spread
            v = mid + rng.uniform(-50.0, 50.0)

            side, edge = flow.decide(bb, ba, v)
            if side == "buy":
                n_buy += 1
                self.assertLess(ba, v - flow.threshold)
                self.assertGreater(v, ba)
                self.assertAlmostEqual(edge, v - ba)
            elif side == "sell":
                n_sell += 1
                self.assertGreater(bb, v + flow.threshold)
                self.assertLess(v, bb)
                self.assertAlmostEqual(edge, bb - v)
            else:
                n_none += 1
                self.assertFalse(ba < v - flow.threshold)
                self.assertFalse(bb > v + flow.threshold)

        print("    20000 random books: %d buys, %d sells, %d no-trade"
              % (n_buy, n_sell, n_none))
        self.assertGreater(n_buy, 0)
        self.assertGreater(n_sell, 0)
        self.assertGreater(n_none, 0)

    # 4: Poisson arrival rate
    def test_4_interarrival_mean(self):
        lam = DEFAULT_LAMBDA_INFORMED
        flow = InformedTraderFlow(lam_informed=lam, seed=555)
        n = 200_000
        gaps = [flow.next_interarrival() for _ in range(n)]

        observed = sum(gaps) / n
        expected = 1.0 / lam
        rel_err = abs(observed - expected) / expected
        print("    lambda_informed=%.6f/s (%.4f of noise rate)"
              % (lam, lam / LAM_NOISE))
        print("    mean inter-arrival: %.2f s (expected %.2f s, err %.2f%%)"
              % (observed, expected, rel_err * 100))
        self.assertLess(rel_err, 0.02)

        sd = statistics.pstdev(gaps)
        print("    std inter-arrival : %.2f s (expect == mean)" % sd)
        self.assertLess(abs(sd - expected) / expected, 0.03)

        self.assertLess(lam, LAM_NOISE)

    # 5: the toxicity knob actually turns
    def test_5_toxicity_ratio_sweepable(self):
        # deep ask wall: every arrival trades, best ask fixed
        T = 400_000.0

        counts = {}
        for level in ("low", "medium", "high"):
            eng = MatchingEngine()
            eng.add_limit_order("sell", 61000.0, 1000.0, "wall")
            flow = InformedTraderFlow.from_toxicity(level, seed=77)
            recs = flow.run_until(eng, T, constant_value(62000.0))
            trades = [r for r in recs if r.side is not None]
            counts[level] = len(trades)
            print("    toxicity=%-6s ratio=%.2f  lambda=%.6f/s  "
                  "arrivals=%4d trades=%4d (expected ~%.0f)"
                  % (level, flow.toxicity_ratio, flow.lam_informed,
                     len(recs), len(trades), flow.lam_informed * T))
            self.assertEqual(len(trades), len(recs))
            self.assertEqual(eng.best_ask(), 61000.0)

        r_ml = counts["medium"] / counts["low"]
        r_hm = counts["high"] / counts["medium"]
        exp_ml = TOXICITY_PRESETS["medium"] / TOXICITY_PRESETS["low"]
        exp_hm = TOXICITY_PRESETS["high"] / TOXICITY_PRESETS["medium"]
        print("    medium/low = %.2f (expected %.2f); high/medium = %.2f "
              "(expected %.2f)" % (r_ml, exp_ml, r_hm, exp_hm))
        self.assertLess(abs(r_ml - exp_ml) / exp_ml, 0.15)
        self.assertLess(abs(r_hm - exp_hm) / exp_hm, 0.15)

    # 6: integration with noise flow and a real value path
    def test_6_integration_trades_only_when_mispriced(self):
        dt = 1.0
        horizon = 100_000.0
        n_steps = int(horizon / dt)

        path = generate_value_path(sigma=0.3721, start_price=62000.0,
                                   dt_seconds=dt, n_steps=n_steps, seed=31415)
        value_fn = path_value(path, dt)

        eng = seeded_book(mid=62000.0, levels=50, size=0.5)
        noise = NoiseTraderFlow(lam=LAM_NOISE, p_market=0.5, seed=1001,
                                reference_price=62000.0)
        informed = InformedTraderFlow(seed=2002)

        chunk = 200.0
        informed_recs = []
        t = 0.0
        while t < horizon:
            t += chunk
            noise.run_until(eng, t)
            informed_recs.extend(informed.run_until(eng, t, value_fn))

        trades = [r for r in informed_recs if r.side is not None]
        passes = [r for r in informed_recs if r.side is None]
        buys = [r for r in trades if r.side == "buy"]
        sells = [r for r in trades if r.side == "sell"]

        print("    V path: start %.2f -> end %.2f (moved %.2f)"
              % (path[0], path[-1], path[-1] - path[0]))
        print("    noise arrivals: %d   informed arrivals: %d"
              % (noise.n_arrivals, informed.n_arrivals))
        print("    informed: %d traded (%d buy, %d sell), %d passed"
              % (len(trades), len(buys), len(sells), len(passes)))
        if trades:
            print("    mean edge when trading: $%.2f  (max $%.2f)"
                  % (sum(r.edge for r in trades) / len(trades),
                     max(r.edge for r in trades)))

        self.assertGreater(len(trades), 0)

        for r in buys:
            self.assertIsNotNone(r.best_ask)
            self.assertLess(r.best_ask, r.value - informed.threshold)
            self.assertTrue(r.fills)
            for f in r.fills:
                self.assertLess(f.price, r.value)
        for r in sells:
            self.assertIsNotNone(r.best_bid)
            self.assertGreater(r.best_bid, r.value + informed.threshold)
            self.assertTrue(r.fills)
            for f in r.fills:
                self.assertGreater(f.price, r.value)

        for r in passes:
            if r.best_ask is not None:
                self.assertFalse(r.best_ask < r.value - informed.threshold)
            if r.best_bid is not None:
                self.assertFalse(r.best_bid > r.value + informed.threshold)

        if eng.best_bid() is not None and eng.best_ask() is not None:
            self.assertLess(eng.best_bid(), eng.best_ask())

    # extra: from_toxicity wiring and determinism
    def test_extra_from_toxicity_and_determinism(self):
        f = InformedTraderFlow.from_toxicity("medium")
        self.assertAlmostEqual(f.lam_informed, 0.10 * LAM_NOISE)
        self.assertAlmostEqual(f.toxicity_ratio, 0.10)
        f2 = InformedTraderFlow.from_toxicity(0.05)
        self.assertAlmostEqual(f2.toxicity_ratio, 0.05)

        def run(seed):
            eng = seeded_book()
            fl = InformedTraderFlow(seed=seed)
            return fl.run_n(eng, 100, constant_value(62050.0))

        self.assertEqual(run(9), run(9))
        self.assertNotEqual(run(9), run(10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
