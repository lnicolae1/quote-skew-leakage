# informed_traders.py: informed traders (the source of adverse selection)
# - observe V_t; trade only when the book is mispriced against V by more than a threshold
# - take (market order) or post (limit at V -/+ threshold), per take_fraction
# - lambda_informed / lambda_noise sets toxicity; not measurable from the data, so it is swept

from collections import namedtuple
import math
import random

from noise_traders import DEFAULT_LAMBDA as DEFAULT_LAMBDA_NOISE

# venue tick (Phase 3)
DEFAULT_TICK = 0.01

# toxicity presets: informed rate / noise rate (chosen, not measured)
TOXICITY_PRESETS = {
    "low": 0.02,
    "medium": 0.10,
    "high": 0.30,
}

# default: medium toxicity against the Phase 3 noise rate 0.0451/s
DEFAULT_LAMBDA_INFORMED = TOXICITY_PRESETS["medium"] * DEFAULT_LAMBDA_NOISE

# mispricing threshold in price units: 5 ticks = $0.05
DEFAULT_THRESHOLD = 5 * DEFAULT_TICK

# fraction of mispriced arrivals that take rather than post (chosen)
DEFAULT_TAKE_FRACTION = 0.5

# informed order size, BTC (chosen, not measured)
DEFAULT_SIZE = 0.00076

# one informed arrival; side None = looked and left without trading
InformedOrder = namedtuple(
    "InformedOrder",
    ["time", "side", "action", "size", "fills", "price", "order_id",
     "resting_size", "value", "best_bid", "best_ask", "edge"],
)


class InformedTraderFlow:
    """Poisson stream of informed arrivals acting on a matching engine, given V_t."""

    def __init__(self, lam_informed=DEFAULT_LAMBDA_INFORMED,
                 threshold=DEFAULT_THRESHOLD, size=DEFAULT_SIZE,
                 take_fraction=DEFAULT_TAKE_FRACTION, tick=DEFAULT_TICK,
                 seed=0, agent_id_prefix="informed"):
        if lam_informed <= 0:
            raise ValueError("lam_informed must be positive, got %r"
                             % (lam_informed,))
        if threshold < 0:
            raise ValueError("threshold must be non-negative, got %r"
                             % (threshold,))
        if size <= 0:
            raise ValueError("size must be positive, got %r" % (size,))
        if not 0.0 <= take_fraction <= 1.0:
            raise ValueError("take_fraction must be in [0, 1], got %r"
                             % (take_fraction,))
        if tick <= 0:
            raise ValueError("tick must be positive, got %r" % (tick,))

        self.lam_informed = lam_informed
        self.threshold = threshold
        self.size = size
        self.take_fraction = take_fraction
        self.tick = tick
        self.agent_id_prefix = agent_id_prefix
        # agent_id appears only on posted limit orders (market orders carry none)
        self.agent_id = agent_id_prefix

        self.rng = random.Random(seed)
        self.time = 0.0
        self.n_arrivals = 0
        self.n_trades = 0
        self.n_takes = 0
        self.n_posts = 0

    # construction by toxicity ratio
    @classmethod
    def from_toxicity(cls, toxicity, lam_noise=DEFAULT_LAMBDA_NOISE, **kwargs):
        """Build from a toxicity preset (low/medium/high) or a float ratio: lam_informed = ratio * lam_noise."""
        ratio = TOXICITY_PRESETS[toxicity] if isinstance(toxicity, str) \
            else float(toxicity)
        if ratio <= 0:
            raise ValueError("toxicity ratio must be positive, got %r"
                             % (ratio,))
        return cls(lam_informed=ratio * lam_noise, **kwargs)

    @property
    def toxicity_ratio(self):
        """Informed rate as a fraction of the Phase 3 noise rate."""
        return self.lam_informed / DEFAULT_LAMBDA_NOISE

    # arrivals and trading decision
    def next_interarrival(self):
        """Seconds until the next arrival ~ Exponential(lam_informed)"""
        return self.rng.expovariate(self.lam_informed)

    def decide(self, best_bid, best_ask, value):
        """Returns (side, edge): buy if best_ask < V - threshold, sell if best_bid > V + threshold, else (None, 0.0)."""
        if best_ask is not None and best_ask < value - self.threshold:
            return "buy", value - best_ask
        if best_bid is not None and best_bid > value + self.threshold:
            return "sell", best_bid - value
        return None, 0.0

    def draw_action(self):
        """"take" (market order) or "post" (limit order), per take_fraction"""
        return "take" if self.rng.random() < self.take_fraction else "post"

    def post_price(self, side, value):
        """Edge-preserving limit price (V -/+ threshold), rounded conservatively to the tick."""
        raw = value - self.threshold if side == "buy" else value + self.threshold
        # 1e-9 guard against float error flooring to the level below
        n = (math.floor(raw / self.tick + 1e-9) if side == "buy"
             else math.ceil(raw / self.tick - 1e-9))
        return round(n * self.tick, 10)

    def _submit(self, engine, value):
        """Look at the book at the current clock time and act."""
        self.n_arrivals += 1
        bb = engine.best_bid()
        ba = engine.best_ask()
        side, edge = self.decide(bb, ba, value)

        if side is None:
            return InformedOrder(self.time, None, None, 0.0, [], None, None,
                                 0.0, value, bb, ba, 0.0)

        self.n_trades += 1
        action = self.draw_action()

        if action == "take":
            self.n_takes += 1
            fills = engine.submit_market_order(side, self.size)
            return InformedOrder(self.time, side, "take", self.size, fills,
                                 None, None, 0.0, value, bb, ba, edge)

        self.n_posts += 1
        price = self.post_price(side, value)
        res = engine.add_limit_order(side, price, self.size, self.agent_id)
        return InformedOrder(self.time, side, "post", self.size, res.fills,
                             price, res.order_id, res.resting_size, value,
                             bb, ba, edge)

    def step(self, engine, value):
        """Advance by one Exponential gap and process one arrival."""
        self.time += self.next_interarrival()
        return self._submit(engine, value)

    def run_n(self, engine, n, value_fn):
        """Process n arrivals"""
        out = []
        for _ in range(n):
            self.time += self.next_interarrival()
            out.append(self._submit(engine, value_fn(self.time)))
        return out

    def run_until(self, engine, t_end, value_fn):
        """Process arrivals until t_end."""
        out = []
        while True:
            gap = self.next_interarrival()
            if self.time + gap > t_end:
                self.time = t_end
                return out
            self.time += gap
            out.append(self._submit(engine, value_fn(self.time)))


def constant_value(v):
    """value_fn for a fixed V."""
    return lambda t: v


def path_value(path, dt_seconds):
    """value_fn(t) from a generate_value_path() result (piecewise constant, clamped at the end)."""
    n = len(path)

    def value_fn(t):
        i = int(t / dt_seconds)
        if i < 0:
            i = 0
        elif i >= n:
            i = n - 1
        return path[i]

    return value_fn


if __name__ == "__main__":
    print("informed_traders.py is a library module; run test_informed_traders.py")
