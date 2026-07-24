# sim_run.py: parameterized driver for the gamma sweep.
# - run_sim(gamma, horizon_s, duration_s, seed) -> dict; main() prints it as JSON
# - gamma and horizon are MarketMaker constructor args; module defaults never edited
# - all else from locked module defaults: noise lam=DEFAULT_ORDER_RATE (0.2255 = 0.0451 x 5),
#   p_market=0.02, mean_lifetime=180.0 s, disp=0.0020; MM sigma=0.3721,
#   k=1.5 (flagged placeholder), quote_size=0.02
# - DEFAULT_LAMBDA (0.0451): measured BTCUSD trade rate, target for emergent_trade_rate
# - wiring from diag_final.py and explore_with_mm.py; loop order noise -> informed -> requote
# - spread capture vs adverse selection not reported (see 'notes')
# - --selftest: determinism check

import argparse
import json
import math
import statistics

from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from noise_traders import NoiseTraderFlow
from informed_traders import InformedTraderFlow, path_value
from market_maker import MarketMaker, DEFAULT_SIGMA

# seed=0 reproduces the RNG streams of explore_with_mm.py / diag_final.py (31415 / 1001 / 2002);
# stride keeps derived streams apart (noise_traders also uses seed+90001)
SEED_V_BASE = 31415
SEED_NOISE_BASE = 1001
SEED_INF_BASE = 2002
SEED_STRIDE = 1000

REFERENCE_PRICE = 62000.0
# Phase 3 annualized vol, shared with the MM default
SIGMA = DEFAULT_SIGMA
DT = 1.0                # simulation step, seconds
Q_SAMPLE_SECONDS = 60   # AR(1) on 60s-sampled q


# --- stats helpers (stdlib only; no numpy) ---
def _percentile(sorted_vals, p):
    """Index percentile on a sorted list; p in [0,1]."""
    if not sorted_vals:
        return None
    i = int(p * (len(sorted_vals) - 1))
    return sorted_vals[i]


def _median(vals):
    return statistics.median(vals) if vals else None


def _ols(xs, ys):
    """OLS of y on x -> (slope, r_squared); (None, None) if x or y is flat (e.g. tiny gamma)."""
    n = len(xs)
    if n < 3:
        return None, None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None, None
    slope = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy)
    return slope, r2


def _ar1(series, demean):
    """Fit x_t = phi * x_{t-1} -> (phi, stderr_of_phi, half_life_samples).
    - demean=True: reversion toward the sample mean
    - demean=False: no intercept, reversion toward 0 (the AS target)
    - half-life is very nonlinear in phi near 1; read it with the SE"""
    n = len(series)
    if n < 10:
        return None, None, None
    if demean:
        mu = sum(series) / n
        x = [v - mu for v in series]
    else:
        x = list(series)
    lag = x[:-1]
    cur = x[1:]
    den = sum(v * v for v in lag)
    if den <= 0.0:
        return None, None, None
    phi = sum(a * b for a, b in zip(lag, cur)) / den
    m = len(cur)
    # dof: 1 parameter toward zero, 2 when demeaned
    dof = m - (2 if demean else 1)
    if dof > 0:
        ssr = sum((c - phi * l) ** 2 for c, l in zip(cur, lag))
        s2 = ssr / dof
        stderr = math.sqrt(s2 / den) if s2 > 0 else 0.0
    else:
        stderr = None
    half_life = math.log(0.5) / math.log(phi) if 0.0 < phi < 1.0 else None
    return phi, stderr, half_life


def _seed_book(eng, mid=REFERENCE_PRICE):
    """Five levels a side, 50c apart (as diag_final.py); otherwise the book starts empty."""
    for i in range(5):
        eng.add_limit_order("buy", round(mid - (i + 1) * 0.5, 2), 0.02, "seed")
        eng.add_limit_order("sell", round(mid + (i + 1) * 0.5, 2), 0.02, "seed")


# --- the run ---
def run_sim(gamma, horizon_s, duration_s, seed):
    """One full simulation; identical arguments give identical output.

    gamma      : MM risk aversion, MarketMaker(gamma=...)
    horizon_s  : MM quoting horizon T, seconds; if duration_s > horizon_s, skew is 0 after T
                 (flagged as 'duration_exceeds_horizon')
    duration_s : simulated length, seconds
    seed       : selects RNG streams for value path, noise, informed
    """
    n_steps = int(duration_s / DT)

    seed_v = SEED_V_BASE + SEED_STRIDE * seed
    seed_noise = SEED_NOISE_BASE + SEED_STRIDE * seed
    seed_inf = SEED_INF_BASE + SEED_STRIDE * seed

    path = generate_value_path(SIGMA, REFERENCE_PRICE, DT, n_steps, seed=seed_v)
    value_fn = path_value(path, DT)

    eng = MatchingEngine()
    _seed_book(eng)

    # locked noise defaults; no overrides
    noise = NoiseTraderFlow(value_fn=value_fn, reference_price=REFERENCE_PRICE,
                            seed=seed_noise)
    informed = InformedTraderFlow(seed=seed_inf)
    mm = MarketMaker(gamma=gamma, horizon=horizon_s)

    birth = {oid: 0.0 for oid in eng.orders}
    ages = []
    mid_dev = []            # |mid - V| per second, where mid exists
    two_sided = 0
    touch_spreads = []      # full-book (includes MM) touch spread
    resid_spreads = []      # residual (non-MM) touch spread, incl. seed orders
    resid_spreads_ns = []   # same, excl. seed orders

    # fix #2: seed orders never expire and can define the residual touch; count how often
    n_resid_bid = n_resid_ask = 0          # seconds a residual bid/ask existed
    n_resid_bid_seed = n_resid_ask_seed = 0

    # fix #3: birth keyed by add_limit_order's order_id, read by Order.id; hit rate checked
    birth_lookups = birth_hits = 0
    n_trade_events = 0      # records with >=1 fill
    total_volume = 0.0      # BTC executed, all fills

    mm_fill_times = []
    mm_passive_volume = 0.0
    mm_aggressive_volume = 0.0
    n_ambiguous_aggressive = 0   # requotes where both sides filled

    t = 0.0
    while t < duration_s:
        t += DT

        for r in noise.run_until(eng, t):
            oid = getattr(r, "order_id", None)
            if oid is not None:
                birth.setdefault(oid, t)
            fills = getattr(r, "fills", None) or []
            if fills:
                n_trade_events += 1
                for f in fills:
                    total_volume += f.size
                    if f.counterparty_id == mm.agent_id:
                        mm_passive_volume += f.size
                        mm_fill_times.append(t)
                mm.on_fills(fills, r.side)

        for r in informed.run_until(eng, t, value_fn):
            oid = getattr(r, "order_id", None)
            if oid is not None:
                birth.setdefault(oid, t)
            fills = getattr(r, "fills", None) or []
            if fills:
                n_trade_events += 1
                for f in fills:
                    total_volume += f.size
                    if f.counterparty_id == mm.agent_id:
                        mm_passive_volume += f.size
                        mm_fill_times.append(t)
                mm.on_fills(fills, r.side)

        # MM aggressive fills inside requote() are not exposed; inferred from fill counters and |dq|.
        # both sides in one requote: |dq| is not the volume, counted as ambiguous
        nb0, ns0, q0 = mm.n_buy_fills, mm.n_sell_fills, mm.q
        mm.requote(eng, t)
        db, ds = mm.n_buy_fills - nb0, mm.n_sell_fills - ns0
        if db or ds:
            mm_fill_times.append(t)
            if db and ds:
                n_ambiguous_aggressive += 1
            else:
                vol = abs(mm.q - q0)
                mm_aggressive_volume += vol
                total_volume += vol

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is not None and ba is not None:
            two_sided += 1
            touch_spreads.append(ba - bb)
        m = eng.mid()
        if m is not None:
            mid_dev.append(abs(m - value_fn(t)))

        # residual book = all but MM quotes; bo/ao incl. seed orders, bo_ns/ao_ns excl.
        bo = ao = bo_ns = ao_ns = None
        for o in eng.orders.values():
            aid = str(o.agent_id)
            if aid == mm.agent_id:
                continue
            is_seed = aid.startswith("seed")
            if o.side == "buy":
                if bo is None or o.price > bo.price:
                    bo = o
                if not is_seed and (bo_ns is None or o.price > bo_ns.price):
                    bo_ns = o
            else:
                if ao is None or o.price < ao.price:
                    ao = o
                if not is_seed and (ao_ns is None or o.price < ao_ns.price):
                    ao_ns = o

        for touch in (bo, ao):
            if touch is not None:
                birth_lookups += 1
                if touch.id in birth:
                    birth_hits += 1
                ages.append(t - birth.get(touch.id, t))

        if bo is not None:
            n_resid_bid += 1
            if str(bo.agent_id).startswith("seed"):
                n_resid_bid_seed += 1
        if ao is not None:
            n_resid_ask += 1
            if str(ao.agent_id).startswith("seed"):
                n_resid_ask_seed += 1

        if bo is not None and ao is not None:
            resid_spreads.append(ao.price - bo.price)
        if bo_ns is not None and ao_ns is not None:
            resid_spreads_ns.append(ao_ns.price - bo_ns.price)

    # --- derived from the MM private log (one row per requote) ---
    rows = mm.log.private_view()
    qs = [r.true_inventory_q for r in rows]
    skews = [(r.quoted_bid + r.quoted_ask) / 2.0 - r.observed_mid for r in rows]
    mm_spreads = [r.quoted_ask - r.quoted_bid for r in rows]

    # C(t) = gamma * sigma_dollar^2 * (T-t), coefficient on -q in the skew; via MM methods
    cs = [mm.gamma * mm.sigma_absolute(r.observed_mid) ** 2
          * mm.time_remaining_years(r.timestamp) for r in rows]

    n_sign_changes = sum(1 for i in range(len(qs) - 1)
                         if (qs[i] > 0) != (qs[i + 1] > 0))
    frac_small_q = (sum(1 for q in qs if abs(q) < 0.1) / len(qs)) if qs else None

    q_sampled = qs[::Q_SAMPLE_SECONDS]
    phi_dm, phi_dm_se, hl_dm = _ar1(q_sampled, demean=True)
    phi_raw, phi_raw_se, hl_raw = _ar1(q_sampled, demean=False)
    hl_dm_s = hl_dm * Q_SAMPLE_SECONDS if hl_dm is not None else None
    hl_raw_s = hl_raw * Q_SAMPLE_SECONDS if hl_raw is not None else None

    abs_skews = sorted(abs(s) for s in skews)
    med_mm_spread = _median(mm_spreads)
    med_resid_spread = _median(resid_spreads)
    med_resid_spread_ns = _median(resid_spreads_ns)

    # fix #3: a low hit rate means every touch age defaults to 0
    birth_hit_rate = (birth_hits / birth_lookups) if birth_lookups else None
    assert birth_hit_rate is None or birth_hit_rate > 0.5, (
        "birth-lookup hit rate %.3f too low; order_id/Order.id join broken, "
        "touch ages default to 0" % birth_hit_rate)

    n_seed_resting_end = sum(1 for o in eng.orders.values()
                             if str(o.agent_id).startswith("seed"))

    slope_raw, r2_raw = _ols(skews, qs)
    tick_skews = [round(s / 0.01) * 0.01 for s in skews]
    slope_tick, r2_tick = _ols(tick_skews, qs)

    total_fills = mm.n_buy_fills + mm.n_sell_fills
    fill_times = sorted(mm_fill_times)
    inter = [b - a for a, b in zip(fill_times, fill_times[1:])]
    final_mid = rows[-1].observed_mid if rows else None
    pnl = mm.mark_to_market(final_mid) if final_mid is not None else None

    return {
        # what was run
        "gamma": gamma,
        "horizon_s": horizon_s,
        "duration_s": duration_s,
        "seed": seed,
        "seeds_used": {"value": seed_v, "noise": seed_noise, "informed": seed_inf},
        "duration_exceeds_horizon": duration_s > horizon_s,

        # skew scale (recorded, not inferred)
        "C_at_t0": cs[0] if cs else None,
        "C_at_end": cs[-1] if cs else None,
        "mean_C": statistics.mean(cs) if cs else None,

        # inventory
        "mean_q": statistics.mean(qs) if qs else None,
        "sd_q": statistics.pstdev(qs) if len(qs) > 1 else None,
        "min_q": min(qs) if qs else None,
        "max_q": max(qs) if qs else None,
        "max_abs_q": max(abs(q) for q in qs) if qs else None,
        "terminal_q": qs[-1] if qs else None,
        "n_sign_changes": n_sign_changes,
        "frac_time_abs_q_below_0.1": frac_small_q,
        # toward-zero (raw) fit is the AS reading; demeaned kept for continuity
        "ar1_phi_raw": phi_raw,
        "ar1_phi_raw_stderr": phi_raw_se,
        "implied_half_life_raw_s": hl_raw_s,
        "ar1_phi_demeaned": phi_dm,
        "ar1_phi_demeaned_stderr": phi_dm_se,
        "implied_half_life_demeaned_s": hl_dm_s,
        "ar1_n_samples": len(q_sampled),

        # skew
        "skew_mean": statistics.mean(skews) if skews else None,
        "skew_sd": statistics.pstdev(skews) if len(skews) > 1 else None,
        "skew_mean_abs": statistics.mean(abs_skews) if abs_skews else None,
        "skew_p95_abs": _percentile(abs_skews, 0.95),
        "skew_sd_over_median_mm_spread": (
            statistics.pstdev(skews) / med_mm_spread
            if len(skews) > 1 and med_mm_spread else None),

        # diagnostic only, not the Phase 6 sniffer: in-sample OLS of true q on skew
        # (leak present; tick-grid loss). R^2 is not a leakage result.
        "DIAGNOSTIC_ols_q_on_skew_slope": slope_raw,
        "DIAGNOSTIC_ols_q_on_skew_r2": r2_raw,
        "DIAGNOSTIC_ols_q_on_tickrounded_skew_slope": slope_tick,
        "DIAGNOSTIC_ols_q_on_tickrounded_skew_r2": r2_tick,

        # fills
        "mm_fills_total": total_fills,
        "mm_fills_per_hour": total_fills / (duration_s / 3600.0),
        "mm_buy_fills": mm.n_buy_fills,
        "mm_sell_fills": mm.n_sell_fills,
        "mm_volume_btc": mm_passive_volume + mm_aggressive_volume,
        "mm_passive_volume_btc": mm_passive_volume,
        "mm_aggressive_volume_btc": mm_aggressive_volume,
        "total_executed_volume_btc": total_volume,
        "mm_share_of_volume": ((mm_passive_volume + mm_aggressive_volume)
                               / total_volume) if total_volume > 0 else None,
        "mean_inter_fill_time_s": statistics.mean(inter) if inter else None,
        "n_ambiguous_aggressive_requotes": n_ambiguous_aggressive,

        # pnl
        "terminal_mark_to_market_pnl": pnl,
        "pnl_per_fill": (pnl / total_fills) if (pnl is not None and total_fills) else None,
        "mm_cash": mm.cash,
        "spread_capture_vs_adverse_selection": None,   # see 'notes'

        # competitiveness
        "median_mm_quoted_spread": med_mm_spread,
        "median_residual_touch_spread": med_resid_spread,
        "median_residual_touch_spread_excluding_seed": med_resid_spread_ns,
        "mm_spread_over_residual_spread": (
            med_mm_spread / med_resid_spread
            if med_mm_spread and med_resid_spread else None),

        # seed-book contamination (fix #2): high fractions mean the residual touch
        # is partly stranded seed book
        "frac_residual_bid_is_seed": (n_resid_bid_seed / n_resid_bid)
                                     if n_resid_bid else None,
        "frac_residual_ask_is_seed": (n_resid_ask_seed / n_resid_ask)
                                     if n_resid_ask else None,
        "n_seed_orders_resting_at_end": n_seed_resting_end,

        # age-join integrity (fix #3)
        "birth_lookup_hit_rate": birth_hit_rate,

        # price-formation guards (should not move with gamma)
        "mean_abs_mid_minus_V": statistics.mean(mid_dev) if mid_dev else None,
        "book_two_sided_fraction": two_sided / n_steps if n_steps else None,
        "emergent_trade_rate_per_s": n_trade_events / duration_s,
        "median_touch_order_age_s": _median(ages),
        "median_full_book_touch_spread": _median(touch_spreads),

        "notes": (
            "spread_capture_vs_adverse_selection not computed: market_maker.py "
            "records no reference mid per fill and has no per-fill hook for "
            "its own aggressive fills; needs new instrumentation and a "
            "mark-out horizon. | emergent_trade_rate_per_s counts trade "
            "events (records with >=1 fill), as in diag_final.py, comparable "
            "to the 0.0416/s figure. | k=1.5 is a flagged placeholder. | "
            "DIAGNOSTIC_* keys are not the Phase 6 sniffer."
        ),
    }


# --- command line ---
def _selftest():
    """Determinism: same arguments twice give equal dicts; seed changes the outcome."""
    print("determinism selftest: gamma=1e-6 horizon=86400 duration=3600 seed=0")
    a = run_sim(1e-6, 86400.0, 3600.0, 0)
    b = run_sim(1e-6, 86400.0, 3600.0, 0)
    assert a == b, "non-deterministic: identical arguments gave different dicts"
    print("  assert a == b   PASSED (%d keys compared)" % len(a))
    # compare outcomes; 'seed' and 'seeds_used' always differ
    c = run_sim(1e-6, 86400.0, 3600.0, 1)
    assert (c["terminal_q"] != a["terminal_q"]
            or c["mm_fills_total"] != a["mm_fills_total"]), (
        "seed had no effect on terminal_q or fills; seeding not wired through")
    print("  assert seed 1 outcome != seed 0 outcome   PASSED (seeding is live)")
    print("    seed0 terminal_q=%+.5f fills=%d | seed1 terminal_q=%+.5f fills=%d"
          % (a["terminal_q"], a["mm_fills_total"],
             c["terminal_q"], c["mm_fills_total"]))
    return True


def main():
    p = argparse.ArgumentParser(description="Parameterized LOB simulation run.")
    p.add_argument("--gamma", type=float, default=1e-6)
    p.add_argument("--horizon", type=float, default=86400.0,
                   help="MM quoting horizon T, seconds")
    p.add_argument("--duration", type=float, default=86400.0,
                   help="simulated run length, seconds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true",
                   help="run the determinism check and exit")
    a = p.parse_args()
    if a.selftest:
        _selftest()
        return
    out = run_sim(a.gamma, a.horizon, a.duration, a.seed)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
