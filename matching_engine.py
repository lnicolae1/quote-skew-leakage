# matching_engine.py: limit order book matching engine
# - price-time (FIFO) priority; incoming orders match against resting ones; book never crossed

from collections import deque, namedtuple

# plain tuple: Fill(100, 5, "A") == (100, 5, "A")
Fill = namedtuple("Fill", ["price", "size", "counterparty_id"])

# result of add_limit_order: id, fills, resting size (a crossing order can fill and then rest)
LimitResult = namedtuple("LimitResult", ["order_id", "fills", "resting_size"])


class Order:
    """Resting limit order; size is the remaining size."""

    __slots__ = ("id", "side", "price", "size", "agent_id")

    def __init__(self, id, side, price, size, agent_id):
        self.id = id
        self.side = side
        self.price = price
        self.size = size
        self.agent_id = agent_id

    def __repr__(self):
        return ("Order(id=%r, side=%r, price=%r, size=%r, agent=%r)"
                % (self.id, self.side, self.price, self.size, self.agent_id))


class MatchingEngine:
    def __init__(self):
        # side: price -> deque of Orders (FIFO; append right, consume left)
        self.bids = {}
        self.asks = {}
        # order_id -> resting Order, for cancel/modify
        self.orders = {}
        self._next_id = 1

    # top of book
    def best_bid(self):
        """Highest price anyone is currently willing to buy at, or None"""
        return max(self.bids) if self.bids else None

    def best_ask(self):
        """Lowest price anyone is currently willing to sell at, or None"""
        return min(self.asks) if self.asks else None

    def mid(self):
        """(best_bid + best_ask) / 2, or None if either side is empty"""
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def depth_at(self, price):
        """Total resting size at a price (both sides summed)."""
        total = 0
        q = self.bids.get(price)
        if q is not None:
            total += sum(o.size for o in q)
        q = self.asks.get(price)
        if q is not None:
            total += sum(o.size for o in q)
        return total

    # matching
    def _match(self, aggressor_side, size, limit_price):
        """Match an aggressor best-price-first, FIFO within a price; limit_price=None means no bound."""
        fills = []
        remaining = size
        buying = (aggressor_side == "buy")
        book = self.asks if buying else self.bids

        while remaining > 0 and book:
            best = min(book) if buying else max(book)
            if limit_price is not None:
                if buying and best > limit_price:
                    break
                if (not buying) and best < limit_price:
                    break

            queue = book[best]
            while queue and remaining > 0:
                head = queue[0]
                traded = head.size if head.size < remaining else remaining
                fills.append(Fill(best, traded, head.agent_id))
                head.size -= traded
                remaining -= traded
                if head.size == 0:
                    queue.popleft()
                    del self.orders[head.id]
                # partial fill keeps its place at the front of the queue

            if not queue:
                del book[best]

        return fills, remaining

    # operations
    def add_limit_order(self, side, price, size, agent_id):
        """Submit a limit order: crossing part executes, remainder rests. Returns LimitResult."""
        if size <= 0:
            raise ValueError("size must be positive, got %r" % (size,))
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell', got %r" % (side,))

        fills, remaining = self._match(side, size, limit_price=price)

        # the remainder cannot cross: matching consumed everything it overlapped
        order_id = self._new_id()
        if remaining > 0:
            order = Order(order_id, side, price, remaining, agent_id)
            self.orders[order_id] = order
            book = self.bids if side == "buy" else self.asks
            book.setdefault(price, deque()).append(order)

        return LimitResult(order_id, fills, remaining)

    def submit_market_order(self, side, size):
        """Market order: take size from the opposite side, no price bound. Returns list of Fill."""
        if size <= 0:
            raise ValueError("size must be positive, got %r" % (size,))
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell', got %r" % (side,))
        fills, _remaining = self._match(side, size, limit_price=None)
        return fills

    def cancel_order(self, order_id):
        """Remove a resting order; True if removed, False if the id is unknown."""
        order = self.orders.get(order_id)
        if order is None:
            return False
        book = self.bids if order.side == "buy" else self.asks
        queue = book.get(order.price)
        if queue is not None:
            try:
                queue.remove(order)
            except ValueError:
                pass
            if not queue:
                del book[order.price]
        del self.orders[order_id]
        return True

    def modify_order(self, order_id, new_price, new_size):
        """Cancel + re-add at the new price (loses time priority). None if the id is unknown."""
        order = self.orders.get(order_id)
        if order is None:
            return None
        side = order.side
        agent_id = order.agent_id
        self.cancel_order(order_id)
        return self.add_limit_order(side, new_price, new_size, agent_id)

    # helpers
    def _new_id(self):
        oid = self._next_id
        self._next_id += 1
        return oid


if __name__ == "__main__":
    print("matching_engine.py is a library module; run test_matching_engine.py")
