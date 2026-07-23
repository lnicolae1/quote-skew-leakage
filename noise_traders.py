# noise_traders.py: noise trader order flow (the maker's profitable flow)
# - Poisson arrivals; side is a 50/50 coin flip; sizes bootstrapped from real Phase 3 trade sizes
# - market orders take; limit orders rest at a noisy view of V (C2b):
#   buys at V * exp(-|eta|), sells at V * exp(+|eta|), eta ~ N(0, disp)
# - resting orders cancel after an Exponential(mean_lifetime) lifetime (B2)
# - limitation: prices carry a weak signal about V (noise traders are not fully uninformed)

import array
import heapq
import math
import os
from collections import namedtuple
import random

# Phase 3 measured trade rate (trades/s, whole-sample average); a measurement, not tuned
DEFAULT_LAMBDA = 0.0451

# submission rate = DEFAULT_LAMBDA * multiplier; raises resting depth without raising the trade rate
DEFAULT_SUPPLY_MULT = 5.0
DEFAULT_ORDER_RATE = DEFAULT_LAMBDA * DEFAULT_SUPPLY_MULT

# fraction of arrivals that are market orders
DEFAULT_P_MARKET = 0.02

# Phase 3 venue constants, BTCUSD on Binance.US
DEFAULT_TICK = 0.01
DEFAULT_SIZES_PATH = r"C:\Users\lnico\Desktop\trade_sizes.bin"

# log-scale dispersion of limit prices around V; price SD ~ V * disp
DEFAULT_DISP = 0.0020

# mean resting lifetime, s; per-order lifetime ~ Exponential(mean)
DEFAULT_MEAN_LIFETIME = 180.0

NoiseOrder = namedtuple(
    "NoiseOrder",
    ["time", "side", "order_type", "size", "price", "fills", "order_id",
     "resting_size"],
)


class EmpiricalSizeSampler:
    """Bootstrap sampler over real Phase 3 trade sizes (trade_sizes.bin, raw float64)."""

    def __init__(self, path=DEFAULT_SIZES_PATH, rng=None):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Empirical trade-size samples not found at %r. This file is "
                "required -- noise traders bootstrap from the REAL Phase 3 "
                "sizes and must not fall back to any invented distribution. "
                "Regenerate it with trade_size_dist.py." % (path,)
            )
        nbytes = os.path.getsize(path)
        if nbytes == 0 or nbytes % 8 != 0:
            raise ValueError(
                "%r is %d bytes, not a whole number of float64 values."
                % (path, nbytes)
            )
        n = nbytes // 8
        arr = array.array("d")
        with open(path, "rb") as f:
            arr.fromfile(f, n)
        self.sizes = arr
        self.n = n
        self.rng = rng if rng is not None else random.Random()

    def sample(self):
        """One size, drawn by uniformly random index (with replacement)"""
        return self.sizes[self.rng.randrange(self.n)]


class NoiseTraderFlow:
    """Poisson stream of noise-trader arrivals acting on a matching engine."""

    def __init__(self, lam=DEFAULT_ORDER_RATE, p_market=DEFAULT_P_MARKET,
                 p_buy=0.5,
                 value_fn=None, disp=DEFAULT_DISP,
                 mean_lifetime=DEFAULT_MEAN_LIFETIME,
                 sizes_path=DEFAULT_SIZES_PATH, tick=DEFAULT_TICK,
                 reference_price=62000.0, seed=0, agent_id_prefix="noise"):
        if lam <= 0:
            raise ValueError("lam must be positive, got %r" % (lam,))
        if not (0.0 <= p_market <= 1.0):
            raise ValueError("p_market must be in [0,1], got %r" % (p_market,))
        if not (0.0 <= p_buy <= 1.0):
            raise ValueError("p_buy must be in [0,1], got %r" % (p_buy,))
        if disp <= 0:
            raise ValueError("disp must be positive, got %r" % (disp,))
        if mean_lifetime <= 0:
            raise ValueError("mean_lifetime must be positive, got %r"
                             % (mean_lifetime,))

        self.lam = lam
        self.p_market = p_market
        self.p_buy = p_buy
        self.value_fn = value_fn
        self.disp = disp
        self.mean_lifetime = mean_lifetime
        self.tick = tick
        self.reference_price = reference_price
        self.agent_id_prefix = agent_id_prefix

        self.rng = random.Random(seed)
        self.sizes = EmpiricalSizeSampler(sizes_path, rng=self.rng)

        # separate RNG so lifetimes do not shift the arrival/side/size/price stream
        self.rng_lifetime = random.Random(seed + 90001)

        # min-heap of (expiry_time, order_id)
        self._expiries = []

        self.time = 0.0
        self.n_arrivals = 0
        self.n_scheduled = 0
        self.n_cancelled = 0
        self.n_already_gone = 0

    # random draws
    def next_interarrival(self):
        """Seconds until the next arrival ~ Exponential(lam), mean 1/lam"""
        return self.rng.expovariate(self.lam)

    def draw_side(self):
        """Independent coin flip (real trade signs have rho(1)=0.41, H=0.79; not modelled)."""
        return "buy" if self.rng.random() < self.p_buy else "sell"

    def draw_size(self):
        """Empirical bootstrap from the real Phase 3 samples"""
        return self.sizes.sample()

    def draw_order_type(self):
        """'market' (take liquidity) or 'limit' (add liquidity)"""
        return "market" if self.rng.random() < self.p_market else "limit"

    def draw_lifetime(self):
        """Resting lifetime ~ Exponential(mean_lifetime), from the lifetime RNG."""
        return self.rng_lifetime.expovariate(1.0 / self.mean_lifetime)

    # cancellation (B2)
    def expire_due(self, engine, t):
        """Cancel every order whose expiry is <= t, via engine.cancel_order."""
        n = 0
        while self._expiries and self._expiries[0][0] <= t:
            _t_exp, order_id = heapq.heappop(self._expiries)
            if engine.cancel_order(order_id):
                self.n_cancelled += 1
                n += 1
            else:
                self.n_already_gone += 1
        return n

    # limit price (C2b)
    def _round_to_tick(self, price):
        # round to tick, trim float noise so equal prices are equal dict keys
        return round(round(price / self.tick) * self.tick, 8)

    def reference(self):
        """Reference price: current V if a path was given, else the static fallback."""
        if self.value_fn is None:
            return self.reference_price
        return self.value_fn(self.time)

    def limit_price(self, engine, side):
        """Limit price: buys V*exp(-|eta|), sells V*exp(+|eta|), eta ~ N(0, disp), snapped to tick."""
        eta = abs(self.rng.gauss(0.0, self.disp))
        ref = self.reference()
        return self._round_to_tick(
            ref * math.exp(-eta if side == "buy" else eta))

    # stepping
    def _submit(self, engine):
        """Draw and submit one arrival at the current clock time."""
        self.n_arrivals += 1
        agent_id = "%s_%d" % (self.agent_id_prefix, self.n_arrivals)

        side = self.draw_side()
        size = self.draw_size()
        order_type = self.draw_order_type()

        if order_type == "market":
            fills = engine.submit_market_order(side, size)
            return NoiseOrder(self.time, side, "market", size, None, fills,
                              None, 0.0)

        price = self.limit_price(engine, side)
        res = engine.add_limit_order(side, price, size, agent_id)
        # schedule expiry only if something rested
        if res.resting_size > 0:
            heapq.heappush(self._expiries,
                           (self.time + self.draw_lifetime(), res.order_id))
            self.n_scheduled += 1
        return NoiseOrder(self.time, side, "limit", size, price, res.fills,
                          res.order_id, res.resting_size)

    def step(self, engine):
        """Advance by one Exponential gap and submit one arrival. Returns a NoiseOrder."""
        self.time += self.next_interarrival()
        self.expire_due(engine, self.time)
        return self._submit(engine)

    def run_n(self, engine, n):
        """Submit n arrivals; return the list of NoiseOrder records"""
        return [self.step(engine) for _ in range(n)]

    def run_until(self, engine, t_end):
        """Submit arrivals until t_end; expiries processed in time order before each arrival."""
        out = []
        while True:
            gap = self.next_interarrival()
            if self.time + gap > t_end:
                self.time = t_end
                self.expire_due(engine, t_end)
                return out
            self.time += gap
            self.expire_due(engine, self.time)
            out.append(self._submit(engine))


if __name__ == "__main__":
    print("noise_traders.py is a library module; run test_noise_traders.py")
