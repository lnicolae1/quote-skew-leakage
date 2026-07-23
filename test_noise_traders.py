# test_noise_traders.py: tests for noise_traders.py (fixed seeds)

import array
import os
import statistics
import unittest

from matching_engine import MatchingEngine
import noise_traders as NT
from noise_traders import (NoiseTraderFlow, EmpiricalSizeSampler,
                           DEFAULT_SIZES_PATH, DEFAULT_LAMBDA)


class _SpyEngine(MatchingEngine):
    """MatchingEngine that records every cancel_order call, so a test can"""

    def __init__(self):
        super().__init__()
        self.cancel_calls = []

    def cancel_order(self, order_id):
        self.cancel_calls.append(order_id)
        return super().cancel_order(order_id)


def load_true_sizes(path=DEFAULT_SIZES_PATH):
    """Load the empirical samples file directly, so the size test compares the"""
    n = os.path.getsize(path) // 8
    arr = array.array("d")
    with open(path, "rb") as f:
        arr.fromfile(f, n)
    return list(arr)


class TestNoiseTraders(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.true_sizes = load_true_sizes()
        cls.true_mean = sum(cls.true_sizes) / len(cls.true_sizes)
        cls.true_median = statistics.median(cls.true_sizes)
        cls.true_max = max(cls.true_sizes)
        print("\n  [file] n=%d  mean=%.8f  median=%.8f  max=%.5f"
              % (len(cls.true_sizes), cls.true_mean, cls.true_median,
                 cls.true_max))

    def test_1_interarrival_mean(self):
        lam = DEFAULT_LAMBDA
        flow = NoiseTraderFlow(lam=lam, seed=1234)
        n = 200_000
        gaps = [flow.next_interarrival() for _ in range(n)]

        observed = sum(gaps) / n
        expected = 1.0 / lam
        rel_err = abs(observed - expected) / expected
        print("    mean inter-arrival: %.4f s (expected %.4f s, err %.2f%%)"
              % (observed, expected, rel_err * 100))

        self.assertLess(rel_err, 0.02)

        sd = statistics.pstdev(gaps)
        print("    std inter-arrival : %.4f s (Exponential => std == mean)"
              % sd)
        self.assertLess(abs(sd - expected) / expected, 0.03)

    def test_2_side_balance(self):
        flow = NoiseTraderFlow(p_buy=0.5, seed=99)
        n = 200_000
        sides = [flow.draw_side() for _ in range(n)]
        buy_frac = sum(1 for s in sides if s == "buy") / n
        print("    buy fraction: %.4f (expected 0.5000)" % buy_frac)
        self.assertLess(abs(buy_frac - 0.5), 0.01)

        x = [1 if s == "buy" else -1 for s in sides]
        mean = sum(x) / n
        c0 = sum((v - mean) ** 2 for v in x) / n
        rho1 = sum((x[i] - mean) * (x[i + 1] - mean)
                   for i in range(n - 1)) / n / c0
        print("    lag-1 sign autocorrelation: %.4f "
              "(real data: 0.4102; placeholder is memoryless by design)"
              % rho1)
        self.assertLess(abs(rho1), 0.01)

    def test_3_size_sampling_matches_file(self):
        flow = NoiseTraderFlow(seed=7)
        n = 200_000
        samples = [flow.draw_size() for _ in range(n)]

        s_mean = sum(samples) / n
        s_median = statistics.median(samples)
        s_max = max(samples)
        print("    sampled mean  : %.8f (file %.8f)" % (s_mean, self.true_mean))
        print("    sampled median: %.8f (file %.8f)"
              % (s_median, self.true_median))
        print("    sampled max   : %.5f (file max %.5f)"
              % (s_max, self.true_max))

        self.assertLess(abs(s_mean - self.true_mean) / self.true_mean, 0.10)
        self.assertLess(abs(s_median - self.true_median) / self.true_median,
                        0.05)

        srt = sorted(samples)
        p99 = srt[int(0.99 * (n - 1))]
        print("    sampled p99   : %.8f  (p99/median = %.1f)"
              % (p99, p99 / s_median))
        self.assertGreater(p99 / s_median, 20)

        self.assertLessEqual(s_max, self.true_max)
        file_set = set(self.true_sizes)
        self.assertTrue(all(v in file_set for v in srt[-50:]))

    def test_4_order_type_split(self):
        n = 100_000
        for p_market in (0.5, 0.2, 0.8):
            flow = NoiseTraderFlow(p_market=p_market, seed=42)
            types = [flow.draw_order_type() for _ in range(n)]
            frac = sum(1 for t in types if t == "market") / n
            print("    p_market=%.1f -> observed market fraction %.4f"
                  % (p_market, frac))
            self.assertLess(abs(frac - p_market), 0.01)

    def test_5_integration_market_fills_and_limits_rest(self):
        eng = MatchingEngine()
        tick = 0.01
        mid = 62000.0
        for i in range(20):
            eng.add_limit_order("buy", round(mid - (i + 1) * tick, 2),
                                0.5, "seed")
            eng.add_limit_order("sell", round(mid + (i + 1) * tick, 2),
                                0.5, "seed")

        flow = NoiseTraderFlow(lam=DEFAULT_LAMBDA, p_market=0.5, seed=2024,
                               tick=tick, reference_price=mid)
        orders = flow.run_n(eng, 2000)

        n_market = sum(1 for o in orders if o.order_type == "market")
        n_limit = len(orders) - n_market
        filled_market = [o for o in orders
                         if o.order_type == "market" and o.fills]
        rested_limit = [o for o in orders
                        if o.order_type == "limit" and o.resting_size > 0]
        traded_volume = sum(f.size for o in orders for f in o.fills)

        print("    arrivals: %d  (market %d, limit %d)"
              % (len(orders), n_market, n_limit))
        print("    market orders that produced fills : %d"
              % len(filled_market))
        print("    limit orders that rested on book  : %d" % len(rested_limit))
        print("    total traded volume: %.6f BTC" % traded_volume)
        print("    clock advanced: %.1f s (%.2f h) for %d arrivals"
              % (flow.time, flow.time / 3600.0, len(orders)))

        self.assertGreater(len(filled_market), 0)
        self.assertEqual(len(filled_market), n_market)
        self.assertGreater(traded_volume, 0)

        self.assertGreater(len(rested_limit), 0)
        noise_ids = {o.order_id for o in rested_limit}
        still_resting = [oid for oid in noise_ids if oid in eng.orders]
        print("    noise limit orders still resting at end: %d"
              % len(still_resting))
        self.assertGreater(len(still_resting), 0)

        self.assertIsNotNone(eng.best_bid())
        self.assertIsNotNone(eng.best_ask())
        print("    final book: best_bid=%.2f best_ask=%.2f"
              % (eng.best_bid(), eng.best_ask()))
        self.assertLess(eng.best_bid(), eng.best_ask())

    def test_6_b2_orders_expire_via_engine_cancel(self):
        """A resting noise limit order must die on schedule, and it must die"""
        eng = _SpyEngine()
        flow = NoiseTraderFlow(lam=1.0, p_market=0.0, mean_lifetime=5.0,
                               reference_price=62000.0, seed=31337)

        early = flow.run_until(eng, 60.0)
        rested = [o.order_id for o in early if o.resting_size > 0]
        print("    orders resting after 60s: %d" % len(rested))
        self.assertGreater(len(rested), 0)
        self.assertTrue(any(oid in eng.orders for oid in rested))

        flow.run_until(eng, 600.0)
        survivors = [oid for oid in rested if oid in eng.orders]
        print("    of those, still resting after 600s: %d (expect 0)"
              % len(survivors))
        self.assertEqual(survivors, [])

        cancelled = set(eng.cancel_calls)
        self.assertTrue(set(rested).issubset(cancelled))
        print("    engine.cancel_order calls: %d   flow.n_cancelled: %d"
              % (len(eng.cancel_calls), flow.n_cancelled))
        self.assertGreater(flow.n_cancelled, 0)

        self.assertEqual(flow.n_already_gone, 0)

        self.assert_book_consistent(eng)

    def assert_book_consistent(self, eng):
        """No dangling references: every order in a price-level queue is in the"""
        seen = []
        for book in (eng.bids, eng.asks):
            for price, queue in book.items():
                self.assertGreater(len(queue), 0,
                                   "empty price level %r left on the book"
                                   % (price,))
                for o in queue:
                    self.assertIn(o.id, eng.orders)
                    self.assertEqual(o.price, price)
                    seen.append(o.id)
        self.assertEqual(len(seen), len(set(seen)), "order in two queues")
        self.assertEqual(set(seen), set(eng.orders.keys()),
                         "engine.orders and the price-level queues disagree")

    def test_7_c2b_placement_is_passive(self):
        """C2b prices buys at V*exp(-|eta|) and sells at V*exp(+|eta|), so a"""
        ref = 62000.0
        flow = NoiseTraderFlow(reference_price=ref, seed=4242)
        n = 20_000
        buys = [flow.limit_price(None, "buy") for _ in range(n)]
        sells = [flow.limit_price(None, "sell") for _ in range(n)]
        print("    buys : max %.2f (must be <= %.2f)   min %.2f"
              % (max(buys), ref, min(buys)))
        print("    sells: min %.2f (must be >= %.2f)   max %.2f"
              % (min(sells), ref, max(sells)))
        self.assertLessEqual(max(buys), ref)
        self.assertGreaterEqual(min(sells), ref)
        self.assertLess(min(buys), ref - 1.0)
        self.assertGreater(max(sells), ref + 1.0)

        eng = MatchingEngine()
        flow2 = NoiseTraderFlow(p_market=0.0, reference_price=ref, seed=909)
        orders = flow2.run_n(eng, 2000)
        crossed = [o for o in orders if o.fills]
        not_fully_rested = [o for o in orders if o.resting_size != o.size]
        print("    limit orders crossing on arrival: %d/%d (expect 0)"
              % (len(crossed), len(orders)))
        self.assertEqual(len(crossed), 0)
        self.assertEqual(len(not_fully_rested), 0)
        self.assert_book_consistent(eng)

    def test_8_locked_defaults_and_emergent_trade_rate(self):
        """DEFAULT_LAMBDA is a Phase 3 measurement and must never be retuned to"""
        self.assertEqual(DEFAULT_LAMBDA, 0.0451)
        self.assertEqual(NT.DEFAULT_SUPPLY_MULT, 5.0)
        self.assertEqual(NT.DEFAULT_ORDER_RATE, 0.0451 * 5.0)
        self.assertEqual(NT.DEFAULT_P_MARKET, 0.02)
        self.assertEqual(NT.DEFAULT_MEAN_LIFETIME, 180.0)
        self.assertEqual(NT.DEFAULT_DISP, 0.0020)

        from fundamental_value import generate_value_path
        from informed_traders import InformedTraderFlow, path_value
        from market_maker import MarketMaker

        T, dt = 21600.0, 1.0
        path = generate_value_path(0.3721, 62000.0, dt, int(T / dt), seed=31415)
        value_fn = path_value(path, dt)
        eng = MatchingEngine()
        noise = NoiseTraderFlow(value_fn=value_fn, reference_price=62000.0,
                                seed=1001)
        informed = InformedTraderFlow(seed=2002)
        mm = MarketMaker(horizon=T)

        n_trades = 0
        t = 0.0
        while t < T:
            t += 1.0
            for r in noise.run_until(eng, t):
                if r.fills:
                    n_trades += 1
                    mm.on_fills(r.fills, r.side)
            for r in informed.run_until(eng, t, value_fn):
                if r.fills:
                    n_trades += 1
                    mm.on_fills(r.fills, r.side)
            mm.requote(eng, t)

        rate = n_trades / T
        print("    emergent trade rate: %.4f/s over %.0fh  "
              "(Phase 3 target %.4f/s)" % (rate, T / 3600.0, DEFAULT_LAMBDA))
        self.assertGreater(rate, 0.030)
        self.assertLess(rate, 0.060)
        self.assert_book_consistent(eng)

    def test_extra_missing_sizes_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            EmpiricalSizeSampler(path="definitely_not_a_real_file.bin")

    def test_extra_deterministic_given_seed(self):
        def run(seed):
            eng = MatchingEngine()
            for i in range(10):
                eng.add_limit_order("buy", round(62000 - (i + 1) * 0.01, 2),
                                    0.5, "seed")
                eng.add_limit_order("sell", round(62000 + (i + 1) * 0.01, 2),
                                    0.5, "seed")
            f = NoiseTraderFlow(seed=seed, reference_price=62000.0)
            return f.run_n(eng, 200)

        self.assertEqual(run(5), run(5))
        self.assertNotEqual(run(5), run(6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
