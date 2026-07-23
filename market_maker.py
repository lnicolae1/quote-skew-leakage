# market_maker.py: Avellaneda-Stoikov market maker, no defenses (baseline)
# - quotes are a deterministic function of inventory q
# - each update: observe mid s; r = s - q * gamma * sigma^2 * (T - t)
#   spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k); bid/ask = r -/+ spread/2
# - (T - t) in years (sigma is annualized); sigma converted to dollars at the mid

import math
from collections import namedtuple

# same convention as fundamental_value.py / compute_volatility.py
SECONDS_PER_YEAR = 365 * 24 * 3600

# parameters (chosen, not measured)

# risk aversion; swept
DEFAULT_GAMMA = 1e-6

# k: fill-intensity decay, A * exp(-k * delta), units 1/$; placeholder default
DEFAULT_K = 1.5

# MM volatility estimate = Phase 3 calibrated sigma
DEFAULT_SIGMA = 0.3721

# quoting horizon, s (default one day)
DEFAULT_HORIZON = 86400.0

# size per side per quote, BTC (near-touch depth ~0.02-0.06 BTC/side within 1 bp)
DEFAULT_QUOTE_SIZE = 0.02

DEFAULT_TICK = 0.01


# public / private boundary: the sniffer only ever gets PublicQuote

# what an outside observer sees: timestamp, bid, ask, mid
PublicQuote = namedtuple(
    "PublicQuote", ["timestamp", "quoted_bid", "quoted_ask", "observed_mid"]
)

_LogRowBase = namedtuple(
    "LogRow",
    ["timestamp", "true_inventory_q", "quoted_bid", "quoted_ask",
     "observed_mid", "mm_cash", "mm_mark_to_market_pnl"],
)


class LogRow(_LogRowBase):
    """Quote-update record incl. private ground truth; pass only .public_view() to the sniffer."""

    __slots__ = ()

    PUBLIC_FIELDS = ("timestamp", "quoted_bid", "quoted_ask", "observed_mid")
    PRIVATE_FIELDS = ("true_inventory_q", "mm_cash", "mm_mark_to_market_pnl")

    def public_view(self):
        """Fresh PublicQuote for this row."""
        return PublicQuote(self.timestamp, self.quoted_bid, self.quoted_ask,
                           self.observed_mid)


class QuoteLog:
    """Ground-truth log: public_view() for the sniffer, private_view() for scoring only."""

    def __init__(self):
        self._rows = []

    def __len__(self):
        return len(self._rows)

    def append(self, row):
        self._rows.append(row)

    def public_view(self):
        """List of PublicQuote"""
        return [r.public_view() for r in self._rows]

    def private_view(self):
        """LogRows incl. true_inventory_q (scoring only)."""
        return list(self._rows)

    def inventory_series(self):
        """Ground-truth q series."""
        return [r.true_inventory_q for r in self._rows]


# market maker

class MarketMaker:
    """Avellaneda-Stoikov market maker, no defenses."""

    def __init__(self, gamma=DEFAULT_GAMMA, sigma=DEFAULT_SIGMA, k=DEFAULT_K,
                 horizon=DEFAULT_HORIZON, quote_size=DEFAULT_QUOTE_SIZE,
                 agent_id="MM", tick=DEFAULT_TICK, round_to_tick=False,
                 fallback_mid=62000.0):
        if gamma <= 0:
            raise ValueError("gamma must be positive, got %r" % (gamma,))
        if sigma < 0:
            raise ValueError("sigma must be non-negative, got %r" % (sigma,))
        if k <= 0:
            raise ValueError("k must be positive, got %r" % (k,))
        if quote_size <= 0:
            raise ValueError("quote_size must be positive, got %r"
                             % (quote_size,))

        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.horizon = horizon
        self.quote_size = quote_size
        self.agent_id = agent_id
        self.tick = tick
        self.round_to_tick = round_to_tick
        self.fallback_mid = fallback_mid

        self.q = 0.0
        self.cash = 0.0
        self.last_mid = None

        # ids of resting quotes, for cancel
        self.bid_order_id = None
        self.ask_order_id = None

        self.log = QuoteLog()
        self.n_buy_fills = 0
        self.n_sell_fills = 0
        self.n_requotes = 0

    # AS formulas
    def time_remaining_years(self, t):
        """(T - t) in years, floored at zero."""
        return max(self.horizon - t, 0.0) / SECONDS_PER_YEAR

    def sigma_absolute(self, mid):
        """Relative annualized sigma -> dollar volatility at mid."""
        return self.sigma * mid

    def reservation_price(self, mid, t):
        """r = s - q * gamma * sigma^2 * (T - t)."""
        sig = self.sigma_absolute(mid)
        tau = self.time_remaining_years(t)
        return mid - self.q * self.gamma * sig * sig * tau

    def total_spread(self, mid, t):
        """spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)."""
        sig = self.sigma_absolute(mid)
        tau = self.time_remaining_years(t)
        return (self.gamma * sig * sig * tau
                + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k))

    def quotes(self, mid, t):
        """(bid, ask) = r -/+ spread/2, unrounded."""
        r = self.reservation_price(mid, t)
        half = self.total_spread(mid, t) / 2.0
        return r - half, r + half

    def inventory_skew(self, mid, t):
        """(bid + ask)/2 - mid = -q * gamma * sigma^2 * (T - t): the inventory skew."""
        sig = self.sigma_absolute(mid)
        tau = self.time_remaining_years(t)
        return -self.q * self.gamma * sig * sig * tau

    # book interaction
    def observe_mid(self, engine):
        """Mid from the book; call after own quotes are cancelled."""
        m = engine.mid()
        if m is None:
            # one-sided or empty book: last real mid
            m = self.last_mid if self.last_mid is not None else self.fallback_mid
        self.last_mid = m
        return m

    def cancel_quotes(self, engine):
        """Cancel both resting quotes; returns how many were still resting."""
        n = 0
        if self.bid_order_id is not None and engine.cancel_order(self.bid_order_id):
            n += 1
        if self.ask_order_id is not None and engine.cancel_order(self.ask_order_id):
            n += 1
        self.bid_order_id = None
        self.ask_order_id = None
        return n

    def _snap(self, price, side):
        """Optional tick rounding: bids down, asks up (never tightens the spread)."""
        if not self.round_to_tick:
            return price
        if side == "buy":
            return round(math.floor(price / self.tick) * self.tick, 8)
        return round(math.ceil(price / self.tick) * self.tick, 8)

    def requote(self, engine, t):
        """Cancel, observe, compute, post, log. Returns the private LogRow."""
        # cancel first so the observed mid excludes own quotes
        self.cancel_quotes(engine)

        mid = self.observe_mid(engine)

        bid, ask = self.quotes(mid, t)
        bid = self._snap(bid, "buy")
        ask = self._snap(ask, "sell")

        # post (back of queue); a marketable quote executes and its fills are booked
        res_bid = engine.add_limit_order("buy", bid, self.quote_size,
                                         self.agent_id)
        for f in res_bid.fills:
            self._apply_fill("buy", f.price, f.size)
        self.bid_order_id = res_bid.order_id if res_bid.resting_size > 0 else None

        res_ask = engine.add_limit_order("sell", ask, self.quote_size,
                                         self.agent_id)
        for f in res_ask.fills:
            self._apply_fill("sell", f.price, f.size)
        self.ask_order_id = res_ask.order_id if res_ask.resting_size > 0 else None

        self.n_requotes += 1

        row = LogRow(
            timestamp=t,
            true_inventory_q=self.q,
            quoted_bid=bid,
            quoted_ask=ask,
            observed_mid=mid,
            mm_cash=self.cash,
            mm_mark_to_market_pnl=self.mark_to_market(mid),
        )
        self.log.append(row)
        return row

    # fill accounting
    def _apply_fill(self, side, price, size):
        """Buy: q up, cash down. Sell: q down, cash up."""
        if side == "buy":
            self.q += size
            self.cash -= price * size
            self.n_buy_fills += 1
        elif side == "sell":
            self.q -= size
            self.cash += price * size
            self.n_sell_fills += 1
        else:
            raise ValueError("side must be 'buy' or 'sell', got %r" % (side,))

    def on_fills(self, fills, aggressor_side):
        """Book fills where the MM was the passive side; aggressor_side is the crossing trader's side."""
        if aggressor_side == "buy":
            own_side = "sell"
        elif aggressor_side == "sell":
            own_side = "buy"
        else:
            raise ValueError("aggressor_side must be 'buy' or 'sell', got %r"
                             % (aggressor_side,))

        n = 0
        for f in fills:
            if f.counterparty_id == self.agent_id:
                self._apply_fill(own_side, f.price, f.size)
                n += 1
        return n

    def mark_to_market(self, mid):
        """PnL = cash + q * mid."""
        return self.cash + self.q * mid


if __name__ == "__main__":
    print("market_maker.py is a library module; run test_market_maker.py")
