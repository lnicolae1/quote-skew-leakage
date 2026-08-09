# toxicity_sweep.py: MM loss with no adversary: toxicity sweep

import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import sim_run
from matching_engine import MatchingEngine
from fundamental_value import generate_value_path
from informed_traders import (InformedTraderFlow, DEFAULT_LAMBDA_NOISE,
                              TOXICITY_PRESETS)
from informed_traders import path_value
from clipped_placement import ClippedNoiseTraderFlow
from market_maker import MarketMaker, DEFAULT_GAMMA, DEFAULT_QUOTE_SIZE
from damage_ab import (T, SEEDS, LAM, P_MARKET, LIFE, DISP, K, REF_MID,
                       C_TARGET, GAMMA, C_PER_GAMMA, MTM_EVERY, SEC_PER_YEAR,
                       sharpe, mean_se)
from clipped_mm_check import MM_ID

REAL_MEDIAN_BP = 0.2873

RATIOS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

BASELINE_PNL = -525.18
BASELINE_SE = 162.43


class TouchInvMM(MarketMaker):
    """as skew unchanged; the 2/k liquidity term replaced by the market"""

    INCLUDE_INV = True

    def total_spread(self, mid, t):
        ms = REAL_MEDIAN_BP * 1e-4 * mid
        if not self.INCLUDE_INV:
            return ms
        sig = self.sigma_absolute(mid)
        tau = self.time_remaining_years(t)
        return self.gamma * sig * sig * tau + ms


class TouchMM(TouchInvMM):
    """Quotes literally at the market touch"""

    INCLUDE_INV = False


def make_world_tox(seed, ratio):
    """Mirrors clipped_placement.make_world exactly, same seed bases, but with"""
    n = int(T)
    path = generate_value_path(sim_run.SIGMA, sim_run.REFERENCE_PRICE, 1.0, n,
                               seed=sim_run.SEED_V_BASE + 1000 * seed)
    vf = path_value(path, 1.0)
    eng = MatchingEngine()
    sim_run._seed_book(eng)
    noise = ClippedNoiseTraderFlow(
        clip_mode="join", lam=LAM, p_market=P_MARKET, mean_lifetime=LIFE,
        disp=DISP, value_fn=vf, reference_price=sim_run.REFERENCE_PRICE,
        seed=sim_run.SEED_NOISE_BASE + 1000 * seed)
    informed = InformedTraderFlow(
        lam_informed=ratio * DEFAULT_LAMBDA_NOISE,
        seed=sim_run.SEED_INF_BASE + 1000 * seed)
    return vf, eng, noise, informed


def run(seed, ratio, mm_cls):
    vf, eng, noise, informed = make_world_tox(seed, ratio)
    mm = mm_cls(horizon=T, k=K, gamma=GAMMA, quote_size=DEFAULT_QUOTE_SIZE)

    tot_vol = 0.0
    mm_vol = 0.0
    mm_fills = 0
    mm_fills_inf = 0
    mm_vol_inf = 0.0
    spread_pnl = 0.0
    mm_spreads = []
    mtm = []
    last_mid = REF_MID

    t = 0.0
    while t < T:
        t += 1.0
        for src, recs in (("noise", noise.run_until(eng, t)),
                          ("inf", informed.run_until(eng, t, vf))):
            for r in recs:
                fills = getattr(r, "fills", None)
                if not fills:
                    continue
                for f in fills:
                    tot_vol += f.size
                    if f.counterparty_id == MM_ID:
                        sgn = 1.0 if r.side == "buy" else -1.0
                        spread_pnl += sgn * (f.price - last_mid) * f.size
                        mm_vol += f.size
                        mm_fills += 1
                        if src == "inf":
                            mm_fills_inf += 1
                            mm_vol_inf += f.size
                mm.on_fills(fills, r.side)
        mm.requote(eng, t)

        bb, ba = eng.best_bid(), eng.best_ask()
        if bb is None or ba is None:
            continue
        last_mid = 0.5 * (bb + ba)
        if int(t) % MTM_EVERY == 0:
            mtm.append(mm.mark_to_market(last_mid))

        mmb = mma = None
        for o in eng.orders.values():
            if o.agent_id != MM_ID:
                continue
            if o.side == "buy":
                mmb = o.price if mmb is None else max(mmb, o.price)
            else:
                mma = o.price if mma is None else min(mma, o.price)
        if mmb is not None and mma is not None:
            mm_spreads.append(mma - mmb)

    pnl = mm.mark_to_market(last_mid)
    return {
        "pnl": pnl,
        "spread": spread_pnl,
        "inv": pnl - spread_pnl,
        "sharpe": sharpe(mtm, MTM_EVERY),
        "fills_day": mm_fills / (T / 86400.0),
        "vol_share": (100.0 * mm_vol / tot_vol) if tot_vol > 0 else 0.0,
        "mm_spread": (statistics.median(mm_spreads) if mm_spreads
                      else float("nan")),
        "frac_inf_fills": (100.0 * mm_fills_inf / mm_fills) if mm_fills else 0.0,
        "frac_inf_vol": (100.0 * mm_vol_inf / mm_vol) if mm_vol > 0 else 0.0,
    }


def measure(ratio, mm_cls):
    per = None
    t0 = time.time()
    for s in SEEDS:
        x = run(s, ratio, mm_cls)
        if per is None:
            per = {k: [] for k in x}
        for k, v in x.items():
            per[k].append(v)
    out = {"ratio": ratio, "secs": time.time() - t0, "raw": per}
    for k, v in per.items():
        m, se, _n = mean_se(v)
        out[k], out[k + "_se"] = m, se
    return out


def cross(rows, key, want_positive=True, two_se=False):
    """Log-interpolate the ratio at which `key` crosses zero (or at which its"""
    def val(r):
        return (r[key] - 2.0 * r[key + "_se"]) if two_se else r[key]
    prev = None
    for r in rows:
        v = val(r)
        if prev is not None:
            pv = val(prev)
            if (pv <= 0.0 < v) or (pv > 0.0 >= v):
                if v == pv:
                    return r["ratio"]
                f = (0.0 - pv) / (v - pv)
                lo, hi = math.log(prev["ratio"]), math.log(r["ratio"])
                return math.exp(lo + f * (hi - lo))
        prev = r
    return None


def main():
    print("=" * 130)
    print("Toxicity sweep; is the Phase 5 item 3 failure caused by informed "
          "flow, or by the AS spread formula?")
    print("=" * 130)
    print("Control arm only, no sniffer. %d seeds x %.0fs (%.0f days). "
          "k=%.5f, C=%.2f, derived gamma=%.4e."
          % (len(SEEDS), T, T / 86400.0, K, C_TARGET, GAMMA))
    print("Working point lam=%.4f p_market=%.4f mean_lifetime=%.0fs "
          "disp=%.4f, JOIN clipping." % (LAM, P_MARKET, LIFE, DISP))
    print("")
    print("The current ratio is %.2f, NOT 2.4. '2.4' in the plan's 'the "
          "toxicity ratio (2.4) is too high'"
          % TOXICITY_PRESETS["medium"])
    print("is a section cross-reference. Presets are low=%.2f medium=%.2f "
          "high=%.2f, and every run"
          % (TOXICITY_PRESETS["low"], TOXICITY_PRESETS["medium"],
             TOXICITY_PRESETS["high"]))
    print("committed so far sat at medium = %.2f. The sweep brackets that."
          % TOXICITY_PRESETS["medium"])
    print("lam_informed <= 0 raises, so 'nearly absent' is ratio=%.3f: about "
          "%.0f informed arrivals in"
          % (RATIOS[0], RATIOS[0] * DEFAULT_LAMBDA_NOISE * T))
    print("the whole run, against ~%.0f/day at the current default."
          % (TOXICITY_PRESETS["medium"] * DEFAULT_LAMBDA_NOISE * 86400.0))
    print("")

    rows = []
    for r in RATIOS:
        row = measure(r, MarketMaker)
        rows.append(row)
        print("  ratio=%.3f  %4.0fs   PnL = %+9.2f +/- %-8.2f   INV = %+9.2f  "
              "spread = %+7.2f   fills/day %.0f"
              % (r, row["secs"], row["pnl"], row["pnl_se"], row["inv"],
                 row["spread"], row["fills_day"]), flush=True)
    print("")

    base = [x for x in rows if abs(x["ratio"] - 0.10) < 1e-12]
    print("=" * 130)
    print("Cross-check; ratio=0.10 with the base MM must reproduce "
          "2eb46f1's control arm")
    print("=" * 130)
    if base:
        b = base[0]
        print("  2eb46f1 control-arm PnL : %+9.2f +/- %.2f"
              % (BASELINE_PNL, BASELINE_SE))
        print("  this file, ratio=0.10   : %+9.2f +/- %.2f"
              % (b["pnl"], b["pnl_se"]))
        ok = abs(b["pnl"] - BASELINE_PNL) < 0.01
        print("  |difference| = %.4f  ->  %s"
              % (abs(b["pnl"] - BASELINE_PNL),
                 "reproduces; the world built here matches make_world"
                 if ok else
                 "does not reproduce; the reproduction has drifted, do not "
                 "trust anything below"))
    print("")

    print("=" * 130)
    print("A. The sweep; does control-arm PnL cross zero at any achievable "
          "toxicity?")
    print("=" * 130)
    print("  %8s %20s %11s %11s %8s %10s %11s %11s %10s"
          % ("ratio", "MM PnL ($)", "spread", "INV", "INV %", "signs",
             "fills/day", "vol share", "spread $"))
    for r in rows:
        neg = sum(1 for v in r["raw"]["pnl"] if v < 0)
        tot = abs(r["spread"]) + abs(r["inv"])
        print("  %8.3f %9.2f +/- %-7.2f %11.2f %11.2f %7.1f%% %5d/%-4d "
              "%11.0f %10.2f%% %10.2f"
              % (r["ratio"], r["pnl"], r["pnl_se"], r["spread"], r["inv"],
                 (100.0 * abs(r["inv"]) / tot) if tot > 0 else float("nan"),
                 neg, len(SEEDS), r["fills_day"], r["vol_share"],
                 r["mm_spread"]))
    print("")
    print("  signs = how many of the %d seeds were negative." % len(SEEDS))
    print("")
    print("  %8s %18s %18s %14s"
          % ("ratio", "MM fills from inf", "MM vol from inf", "Sharpe"))
    for r in rows:
        print("  %8.3f %17.2f%% %17.2f%% %14.2f"
              % (r["ratio"], r["frac_inf_fills"], r["frac_inf_vol"],
                 r["sharpe"]))

    zc = cross(rows, "pnl", two_se=False)
    zc2 = cross(rows, "pnl", two_se=True)
    print("")
    print("=" * 130)
    print("B. The verdict")
    print("=" * 130)
    print("  mean PnL crosses zero at ratio         : %s"
          % ("%.4f" % zc if zc is not None else
             "never in this grid (0.001 to 0.30)"))
    print("  lower 2-SE bound clears zero at ratio  : %s"
          % ("%.4f" % zc2 if zc2 is not None else
             "never in this grid (0.001 to 0.30)"))
    print("  Only the second is a PASS by the gate's own rule; a mean above "
          "zero whose interval")
    print("  straddles it does not demonstrate profitability.")
    print("")
    if zc2 is not None:
        print("  Outcome (a): the gate is passable. Toxicity is the cause. "
              "That ratio becomes the")
        print("  calibration, and every Phase 6 number; the fill-only "
              "floor, the skew baseline, the")
        print("  saturation curve, the level/velocity split and the step 7 "
              "damage A/B; has to be")
        print("  re-run there before any of it can be written down.")
    else:
        print("  Outcome (b): toxicity is not the cause. PnL stays negative "
              "even with informed flow")
        print("  essentially switched off, so the informed:noise ratio does "
              "not explain the Phase 5")
        print("  failure and lowering it will not fix the maker. The "
              "remaining candidate is the as")
        print("  spread formula against the measured k = 0.17763. See "
              "section C.")

    print("")
    print("=" * 130)
    print("C. The discriminating comparison; same decomposition for an MM "
          "quoting at the market touch")
    print("=" * 130)
    print("  Skew (reservation_price) unchanged in both variants; only the "
          "width moves. market_spread")
    print("  = %.4fbp of mid, the Phase 3 median the book is calibrated to."
          % REAL_MEDIAN_BP)
    print("  Run at the current toxicity ratio %.2f so the only thing "
          "changing is the MM's width."
          % TOXICITY_PRESETS["medium"])
    print("")
    variants = [("BASE  2/k = $11.26 (as committed)", MarketMaker),
                ("TOUCH+INV  gamma*sig^2*tau + mkt", TouchInvMM),
                ("TOUCH  mkt spread only", TouchMM)]
    vrows = []
    for label, cls in variants:
        if cls is MarketMaker and base:
            vr = base[0]
        else:
            vr = measure(TOXICITY_PRESETS["medium"], cls)
        vrows.append((label, vr))
        print("  %-34s  PnL = %+9.2f +/- %-8.2f  measured"
              % (label, vr["pnl"], vr["pnl_se"]), flush=True)
    print("")
    print("  %-34s %20s %11s %11s %8s %10s %11s %10s"
          % ("variant", "MM PnL ($)", "spread", "INV", "INV %", "signs",
             "fills/day", "spread $"))
    for label, r in vrows:
        neg = sum(1 for v in r["raw"]["pnl"] if v < 0)
        tot = abs(r["spread"]) + abs(r["inv"])
        print("  %-34s %9.2f +/- %-7.2f %11.2f %11.2f %7.1f%% %5d/%-4d "
              "%11.0f %10.2f"
              % (label, r["pnl"], r["pnl_se"], r["spread"], r["inv"],
                 (100.0 * abs(r["inv"]) / tot) if tot > 0 else float("nan"),
                 neg, len(SEEDS), r["fills_day"], r["mm_spread"]))
    print("")
    print("  if THE INV bleed largely disappears when the MM quotes "
          "competitively, the Phase 5 failure is")
    print("  a selection effect of quoting ~9x outside the touch; the maker "
          "fills only when price")
    print("  sweeps out to it, which is adverse by construction; and it "
          "implicates the k mismatch,")
    print("  not toxicity. If the bleed persists at the touch too, neither "
          "candidate explains it and")
    print("  the cause is elsewhere.")
    print("")
    print("  Note: TOUCH+INV still carries the as inventory term, which at "
          "C=%.2f leaves it several" % C_TARGET)
    print("  times wider than the market. TOUCH removes that too. The pair "
          "brackets the answer.")
    print("")
    print("  No committed default changed. gamma, k and quote_size are "
          "constructor arguments; gamma")
    print("  is derived from C. No sniffer anywhere in this file.")


if __name__ == "__main__":
    main()
