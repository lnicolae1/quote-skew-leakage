# test_matching_engine.py: unit tests for matching_engine.py (python test_matching_engine.py)

import unittest

from matching_engine import MatchingEngine


class TestMatchingEngine(unittest.TestCase):

    def test_1_fifo_partial_fill_keeps_front_priority(self):
        eng = MatchingEngine()
        r1 = eng.add_limit_order("sell", 100, 10, "A")
        r2 = eng.add_limit_order("sell", 100, 10, "B")
        r3 = eng.add_limit_order("sell", 100, 10, "C")

        fills = eng.submit_market_order("buy", 15)

        self.assertEqual(fills, [(100, 10, "A"), (100, 5, "B")])

        self.assertNotIn(r1.order_id, eng.orders)
        self.assertFalse(eng.cancel_order(r1.order_id))

        q = eng.asks[100]
        self.assertEqual(q[0].id, r2.order_id)
        self.assertEqual(q[0].size, 5)

        self.assertEqual(q[1].id, r3.order_id)
        self.assertEqual(q[1].size, 10)

        self.assertEqual(eng.depth_at(100), 15)

    def test_2_crossing_limit_executes_not_rests(self):
        eng = MatchingEngine()
        eng.add_limit_order("sell", 100, 5, "S")

        res = eng.add_limit_order("buy", 101, 5, "T")

        self.assertEqual(res.fills, [(100, 5, "S")])
        self.assertEqual(res.resting_size, 0)
        self.assertNotIn(res.order_id, eng.orders)
        self.assertIsNone(eng.best_bid())
        self.assertIsNone(eng.best_ask())
        self.assertEqual(eng.depth_at(100), 0)
        self.assertEqual(eng.depth_at(101), 0)

    def test_3_partial_cross_fills_then_rests_remainder(self):
        eng = MatchingEngine()
        eng.add_limit_order("sell", 100, 3, "S")

        res = eng.add_limit_order("buy", 100, 5, "T")

        self.assertEqual(res.fills, [(100, 3, "S")])
        self.assertEqual(res.resting_size, 2)
        self.assertIsNone(eng.best_ask())
        self.assertEqual(eng.best_bid(), 100)
        self.assertEqual(eng.depth_at(100), 2)
        self.assertIn(res.order_id, eng.orders)

    def test_4_cancel_middle_promotes_those_behind(self):
        eng = MatchingEngine()
        rx = eng.add_limit_order("buy", 100, 1, "X")
        ry = eng.add_limit_order("buy", 100, 2, "Y")
        rz = eng.add_limit_order("buy", 100, 3, "Z")

        self.assertTrue(eng.cancel_order(ry.order_id))

        q = eng.bids[100]
        self.assertEqual([o.id for o in q], [rx.order_id, rz.order_id])

        fills = eng.submit_market_order("sell", 4)
        self.assertEqual(fills, [(100, 1, "X"), (100, 3, "Z")])

    def test_5_top_of_book_after_sequence(self):
        eng = MatchingEngine()
        eng.add_limit_order("buy", 99, 5, "A")
        rb = eng.add_limit_order("buy", 98, 3, "B")
        eng.add_limit_order("sell", 101, 4, "C")
        eng.add_limit_order("sell", 102, 2, "D")

        self.assertEqual(eng.best_bid(), 99)
        self.assertEqual(eng.best_ask(), 101)
        self.assertEqual(eng.mid(), 100.0)
        self.assertEqual(eng.depth_at(99), 5)
        self.assertEqual(eng.depth_at(101), 4)

        eng.add_limit_order("buy", 99, 2, "E")
        self.assertEqual(eng.depth_at(99), 7)

        self.assertTrue(eng.cancel_order(rb.order_id))
        self.assertEqual(eng.best_bid(), 99)

        fills = eng.submit_market_order("buy", 5)
        self.assertEqual(fills, [(101, 4, "C"), (102, 1, "D")])
        self.assertEqual(eng.best_ask(), 102)
        self.assertEqual(eng.depth_at(102), 1)
        self.assertEqual(eng.mid(), (99 + 102) / 2.0)

    def test_extra_market_order_exhausts_book(self):
        eng = MatchingEngine()
        eng.add_limit_order("sell", 100, 2, "S")
        fills = eng.submit_market_order("buy", 5)
        self.assertEqual(fills, [(100, 2, "S")])
        self.assertIsNone(eng.best_ask())

    def test_extra_cancel_unknown_returns_false(self):
        eng = MatchingEngine()
        self.assertFalse(eng.cancel_order(999999))

    def test_modify_price_change_loses_priority(self):
        eng = MatchingEngine()
        rp = eng.add_limit_order("buy", 100, 5, "P")
        rq = eng.add_limit_order("buy", 101, 1, "Q")

        res = eng.modify_order(rp.order_id, 101, 5)
        self.assertIsNotNone(res)

        self.assertNotIn(100, eng.bids)
        self.assertEqual(eng.depth_at(101), 6)

        q = eng.bids[101]
        self.assertEqual(q[0].id, rq.order_id)
        self.assertEqual(q[1].id, res.order_id)

        fills = eng.submit_market_order("sell", 6)
        self.assertEqual(fills, [(101, 1, "Q"), (101, 5, "P")])

    def test_modify_same_price_size_change_loses_priority(self):
        eng = MatchingEngine()
        rp = eng.add_limit_order("buy", 100, 5, "P")
        rq = eng.add_limit_order("buy", 100, 1, "Q")

        self.assertEqual([o.id for o in eng.bids[100]], [rp.order_id, rq.order_id])

        res = eng.modify_order(rp.order_id, 100, 3)
        self.assertIsNotNone(res)

        q = eng.bids[100]
        self.assertEqual(q[0].id, rq.order_id)
        self.assertEqual(q[1].id, res.order_id)
        self.assertEqual(q[1].size, 3)

    def test_modify_unknown_id_returns_none(self):
        eng = MatchingEngine()
        self.assertIsNone(eng.modify_order(123456, 100, 5))

        rs = eng.add_limit_order("sell", 100, 2, "S")
        eng.submit_market_order("buy", 2)
        self.assertNotIn(rs.order_id, eng.orders)
        self.assertIsNone(eng.modify_order(rs.order_id, 101, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
