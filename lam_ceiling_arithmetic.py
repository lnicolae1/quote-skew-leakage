# lam_ceiling_arithmetic.py: arithmetic only; can the (lam, mean_lifetime) lever (78701e4) reach
# life ~60s at fixed book size under the lam ceiling?
# - claim: 720 -> 60 needs lam x12 = 21.6/s; book updates at ~3.09 IDs/s incl. cancels and trades,
#   so lam must be a fraction of 3.09 (1.804 is 58% of it)
# - 3.09 is unsourced (bare figure in clipped_lam_sweep.py, f9293d8); recorded rates used instead
# - lam counts submissions only: ceiling is a fraction of the update rate (addition share 0.501132)
# - Binance aggregates per level per diff: update counts undercount orders (not quantified)
# - ratios computed from stored inputs, never from rounded display values

import math

# ---------------------------------------------------------------- inputs -----

# lifetime_sweep_results.txt sections B and C
# life_s, levels_per_side, pnl, pnl_se, width_bp
SWEEP = [
    (   60.0,    52.2,   14.91, 193.95, 2.7100),
    (  180.0,   151.4, -244.36, 168.01, 1.1463),
    (  360.0,   292.5, -391.65, 168.15, 0.6506),
    (  720.0,   556.9, -525.18, 162.43, 0.3976),
    ( 1440.0,  1043.3, -571.21, 184.13, 0.2284),
    ( 2880.0,  1932.9, -803.67, 239.96, 0.1159),
]

LAM_HELD = 1.8040          # lifetime_sweep_results.txt
P_MARKET_HELD = 0.0126
COMMITTED_LIFE = 720.0     # committed default, in-band row
REAL_LEVELS_TARGET = 666.34   # bid side

USER_GRID = [60.0, 180.0, 360.0, 720.0, 1440.0, 2880.0]

CEIL_QUOTED = 3.09         # clipped_lam_sweep.py; unsourced
NOTESMD_IDS = 2257.0      # 07-13 reboot window
NOTESMD_SECS = 108.0

# 2-day span between two recorded update IDs
ID_0709 = 4491244963       # diagnose_crossing_results.txt
ID_0711 = 4494915679       # research_log.md
# 2026-07-09T00:02:20.962636 -> 2026-07-11T00:04:10.925672
SPAN_0709_0711_S = 2 * 86400.0 + (
    (0 * 3600 + 4 * 60 + 10.925672) - (0 * 3600 + 2 * 60 + 20.962636))

# book_change_mix.py section A, cloud chain 07-11..07-13
CLOUD_IDS = 2795579.0      # observed update IDs (span minus 220 lost)
CLOUD_SECS = 194813.671    # first to last event
# book_change_mix.py sections C and D, within 50bp
MEAS_ADD_SHARE = 0.501132  # (CREATED+UP)/(CREATED+UP+DOWN+REMOVED)
MEAS_ADD_RATE_R1 = 6.3235  # Route 1: additions counted / wall span
MEAS_ADD_RATE_R2 = 7.1913  # Route 2: share x update-ID rate

SAFETY = 0.90              # "maximum defensible lam" = 90% of a ceiling


def rule(ch="="):
    print(ch * 100)


def interp_pnl(life):
    """PnL at a lifetime between grid points, linear in log(life) (geometric grid); interpolated, not measured."""
    pts = sorted((s[0], s[2]) for s in SWEEP)
    if life <= pts[0][0]:
        return pts[0][1], pts[0], pts[0]
    if life >= pts[-1][0]:
        return pts[-1][1], pts[-1], pts[-1]
    for (l0, p0), (l1, p1) in zip(pts, pts[1:]):
        if l0 <= life <= l1:
            f = (math.log(life) - math.log(l0)) / (math.log(l1) - math.log(l0))
            return p0 + f * (p1 - p0), (l0, p0), (l1, p1)
    raise AssertionError("unreachable")


# ------------------------------------------------------- 1. the grid ---------

rule()
print("1. Lifetime grid behind the PnL series")
rule()
grid = [s[0] for s in SWEEP]
print("  from lifetime_sweep_results.txt sections B and C:")
print("    " + "  ".join("%.0fs" % g for g in grid))
print("  claimed grid:")
print("    " + "  ".join("%.0fs" % g for g in USER_GRID))
print("  match: %s" % ("YES" if grid == USER_GRID else "NO"))
print()
print("  PnL series (section B, MM PnL $ +/- se):")
for life, lv, pnl, se, w in SWEEP:
    zero = "  <-- straddles zero" if abs(pnl) < se else ""
    print("    life=%6.0fs   PnL = %+8.2f +/- %6.2f%s" % (life, pnl, se, zero))
print()
print("  +14.91 +/- 193.95 at 60s: not distinguishable from zero or from -244.36 at 180s;")
print("  reaching 60s buys not-significantly-negative, not profitable")
print()

# ------------------------------- 2. is book size proportional to lifetime? ---

rule()
print("2. levels/side vs mean_lifetime: proportional or sublinear?")
rule()
print("  N ~ lam*(1-p_market)*life exactly => levels/life constant")
print()
print("    life (s)   levels/side   levels/life   ratio to life=720 row")
base = [s for s in SWEEP if s[0] == COMMITTED_LIFE][0]
for life, lv, pnl, se, w in SWEEP:
    print("    %8.0f   %11.1f   %11.6f   %+.4f" %
          (life, lv, lv / life, lv / base[1]))
print()
lo, hi = SWEEP[0], SWEEP[-1]
life_span = hi[0] / lo[0]
lvl_span = hi[1] / lo[1]
slope = math.log(lvl_span) / math.log(life_span)
print("  over the full grid: life x%.4f  ->  levels x%.4f" % (life_span, lvl_span))
print("  log-log slope = ln(%.4f)/ln(%.4f) = %.4f" % (lvl_span, life_span, slope))
print("  sublinear (slope < 1): %s" % ("YES" if slope < 1 else "NO"))
print()
print("  pairwise local slopes:")
for (l0, v0, _a, _b, _c), (l1, v1, _d, _e, _f) in zip(SWEEP, SWEEP[1:]):
    print("    %6.0f -> %6.0f s   slope = %.4f" %
          (l0, l1, math.log(v1 / v0) / math.log(l1 / l0)))
print()
print("  claimed x12 = 720/60; a lam multiplier must undo the measured levels ratio:")
print("    720s row / 60s row  =  %.1f / %.1f  =  %.4fx" %
      (base[1], lo[1], base[1] / lo[1]))
print("    the naive lifetime ratio 720/60                = %.4fx" % (720.0 / 60.0))
print("  so the required multiplier is below x12 (claim conservative here)")
print()

# ---------------------------------------------- 3. the ceiling's provenance --

rule()
print("3. Provenance of the 3.09 IDs/sec ceiling")
rule()
print("  provenance (git log -S, grep):")
print("    first appearance   clipped_lam_sweep.py:36, commit f9293d8")
print("    stated as          bare figure in a comment, no results file")
print("    absent from        every *_results.txt, research_log.md and the")
print("                       Phase 1-3 calibration scripts")
print("    later uses         clipped_depth_sweep.py, clipped_pmarket_grid.py")
print("                       (inherited, not re-derived)")
print("    verdict            unsourced constant")
print()
r_notesmd = NOTESMD_IDS / NOTESMD_SECS
print("  (a) 07-13 reboot window: %.0f update IDs in ~%.0fs" %
      (NOTESMD_IDS, NOTESMD_SECS))
print("      %.0f / %.0f = %.4f IDs/sec" % (NOTESMD_IDS, NOTESMD_SECS, r_notesmd))
d_id = float(ID_0711 - ID_0709)
r_2day = d_id / SPAN_0709_0711_S
print("  (b) 2-day span between two recorded IDs:")
print("      %d - %d = %.0f update IDs" % (ID_0711, ID_0709, d_id))
print("      over %.6f s  ->  %.4f IDs/sec" % (SPAN_0709_0711_S, r_2day))
print()
r_cloud = CLOUD_IDS / CLOUD_SECS
print("  (c) measured over the cloud chain 07-11..07-13, 2.25 days")
print("      (book_change_mix.py, section A -- run 2026-08-15):")
print("      %.0f observed update IDs over %.3f s  ->  %.4f IDs/sec"
      % (CLOUD_IDS, CLOUD_SECS, r_cloud))
print()
print("      the three figures disagree:")
print("        (a) %8.4f IDs/s   a 108-SECOND window, between two reconnects" %
      r_notesmd)
print("        (b) %8.4f IDs/s   07-09 -> 07-11, ~2 days, LOCAL chain" % r_2day)
print("        (c) %8.4f IDs/s   07-11 -> 07-13, 2.25 days, CLOUD chain" % r_cloud)
print()
print("      (a) and (b) cover adjacent periods (not independent); (a) is a burst")
print("      (c), the longest clean window, sits %.2f%% below (b) on the immediately adjacent days" %
      (100.0 * (1.0 - r_cloud / r_2day)))
print()
print("      update rate is not a scalar: varies ~%.0f%% across adjacent multi-day windows" %
      (100.0 * (r_2day / r_cloud - 1.0)))
print()
print("      every '% of ceiling' figure below is a point estimate with ~50% spread;")
print("      a verdict turning on less than that spread is not decided")
print("      3.09 is below all three")
print("      spread_depth_profile.py:26 says '~20 events/sec' independently.")
print()
print("  quoted ceiling too low by:")
print("      %.4f / %.4f = %.4fx   (vs the NOTES.md figure)" %
      (r_notesmd, CEIL_QUOTED, r_notesmd / CEIL_QUOTED))
print("      %.4f / %.4f = %.4fx   (vs the 2-day derivation)" %
      (r_2day, CEIL_QUOTED, r_2day / CEIL_QUOTED))
print()
print("  an update ID is any book change; submissions are a fraction of it:")
print("  each submitted order is eventually removed by cancel or trade,")
print("  so in steady state submissions are at most half the update stream")
print()

print("  divide-by-2 measured:")
print("  book_change_mix.py classified %s level entries over the cloud chain"
      % "2,921,638")
print("  and found the addition share, within 50bp of the mid, to be %.6f"
      % MEAS_ADD_SHARE)
print("  against the assumed 0.500000 -- a ratio of %.4fx."
      % (MEAS_ADD_SHARE / 0.5))
print()
print("  1,246,383 level creations vs 1,246,386 removals over 2.25 days: additions are half of book changes")
print()

CEILINGS = [
    ("QUOTED 3.09, unsourced", CEIL_QUOTED),
    ("NOTES.md rate, raw", r_notesmd),
    ("NOTES.md rate, half (submissions only)", r_notesmd / 2.0),
    ("2-day rate, half (submissions only)", r_2day / 2.0),
    ("MEASURED additions/s, Route 2 (share x IDs)", MEAS_ADD_RATE_R2),
    ("MEASURED additions/s, Route 1 (direct count)", MEAS_ADD_RATE_R1),
]
print("  candidate ceilings:")
for name, c in CEILINGS:
    print("    %-42s %8.4f /s   (lam=%.4f is %6.2f%% of it)" %
          (name, c, LAM_HELD, 100.0 * LAM_HELD / c))
print()

# ------------------------------------- 4. required lam at fixed book size ----

rule()
print("4. lam holding levels/side at the life=720 value (%.1f), per life" %
      base[1])
rule()
print("  model: levels ~ lam*(1-p_market)*life; at fixed p_market")
print("  lam needed at lifetime L = lam_held * (target_levels / levels(L))  (lam_lin)")
print()
print("  lam_sub (speculative bound, not conservative) applies")
print("  section 2's LIFETIME exponent (%.4f) to lam: lam_needed =" % slope)
print("  lam_held * (target/levels)**(1/%.4f); assumes lam and lifetime saturate alike" % slope)
print("  untested: lifetime saturates as orders are consumed or expire before the book fills,")
print("  lam adds fresh orders, so lam may be closer to linear than lifetime; direction unknown")
print()
print("    life(s)   levels    lam_lin    lam_sub    " +
      "  ".join("%-11s" % n.split(",")[0][:11] for n, _ in CEILINGS))
print("                                              " +
      "  ".join("%-11s" % ("%% of %.3f" % c) for _, c in CEILINGS))
for life, lv, pnl, se, w in SWEEP:
    need_lin = LAM_HELD * (base[1] / lv)
    need_sub = LAM_HELD * (base[1] / lv) ** (1.0 / slope)
    cells = "  ".join("%-11s" % ("%.0f%%" % (100.0 * need_lin / c))
                      for _, c in CEILINGS)
    print("    %7.0f   %6.1f   %8.4f   %8.4f    %s" %
          (life, lv, need_lin, need_sub, cells))
print()
print("  (% columns use lam_lin)")
print()
print("  claimed lam x12 = 21.6/s at life=60; from measured levels instead:")
n60_lin = LAM_HELD * (base[1] / lo[1])
n60_sub = LAM_HELD * (base[1] / lo[1]) ** (1.0 / slope)
print("    linear    lam = %.4f * %.4f = %.4f /s  (x%.4f)" %
      (LAM_HELD, base[1] / lo[1], n60_lin, n60_lin / LAM_HELD))
print("    sublinear lam = %.4f /s  (x%.4f)" % (n60_sub, n60_sub / LAM_HELD))
print("    user's    lam = %.4f /s  (x%.4f)" % (LAM_HELD * 12.0, 12.0))
print("  modest correction; direction unchanged")
print()

# ------------------------------- 5. reverse: what life is reachable at all ---

rule()
print("5. Reverse: lifetime and PnL reachable at maximum defensible lam, fixed book size")
rule()
print("  max defensible lam = %.0f%% of a ceiling; fixed levels hold lam*life:" %
      (100.0 * SAFETY))
print("  life_reachable = %.0f * lam_held / lam_max" %
      COMMITTED_LIFE)
print()
for name, c in CEILINGS:
    lam_max = SAFETY * c
    life_reach = COMMITTED_LIFE * LAM_HELD / lam_max
    pnl, blo, bhi = interp_pnl(life_reach)
    print("  ceiling %-42s %8.4f /s" % (name, c))
    print("    lam_max = %.4f, life reachable = %.1f s" % (lam_max, life_reach))
    if life_reach <= SWEEP[0][0]:
        print("    -> below the grid floor of %.0fs: 60s row reachable" %
              SWEEP[0][0])
        print("       PnL there = %+.2f +/- %.2f" % (SWEEP[0][2], SWEEP[0][3]))
    else:
        print("    -> PnL interpolated (log-life) = %+.2f" % pnl)
        print("       bracketed by  life=%.0fs PnL=%+.2f   and  life=%.0fs PnL=%+.2f"
              % (blo[0], blo[1], bhi[0], bhi[1]))
        print("       distance still to the 60s row: %.1f s of lifetime, and a" %
              (life_reach - SWEEP[0][0]))
        print("       further lam factor of x%.4f on top of lam_max." %
              (life_reach / SWEEP[0][0]))
    print()

# ------------------------------------------------------------ 6. verdict -----

rule()
print("6. Verdict: can the (lam, life) lever reach life=60 at fixed book size?")
rule()
for name, c in CEILINGS:
    need = LAM_HELD * (base[1] / lo[1])
    lam_max = SAFETY * c
    ok = need <= lam_max
    print("  ceiling %-42s  need %.4f vs lam_max %.4f  ->  %s" %
          (name, need, lam_max, "REACHABLE" if ok else "NOT REACHABLE"))
    print("      required lam is %.4fx the full ceiling, %.4fx the 90%% bound" %
          (need / c, need / lam_max))
print()
print("  NOT REACHABLE on every candidate ceiling; margins differ widely:")
need_ = LAM_HELD * (base[1] / lo[1])
worst = max(CEILINGS, key=lambda nc: need_ / (SAFETY * nc[1]))
best = min(CEILINGS, key=lambda nc: need_ / (SAFETY * nc[1]))
print("    widest  %-42s fails by %.2fx" %
      (worst[0], need_ / (SAFETY * worst[1])))
print("    tightest %-41s fails by %.2fx" %
      (best[0], need_ / (SAFETY * best[1])))
print("  tightest row misses by %.0f%%: a near miss, not a structural margin" %
      (100.0 * (need_ / (SAFETY * best[1]) - 1.0)))
print()
print("  measured addition rates harden the verdict:")
print("  cloud-chain addition rate %.4f/s (Route 1,"
      % MEAS_ADD_RATE_R1)
print("  direct count) to %.4f/s (Route 2), both below the assumed %.4f/s;"
      % (MEAS_ADD_RATE_R2, r_notesmd / 2.0))
print("  the life=60 requirement of %.4f/s"
      % need_)
print("  misses by more:")
print("      assumed half-rate  %8.4f /s   ->  short by %.2fx" %
      (r_notesmd / 2.0, need_ / (SAFETY * r_notesmd / 2.0)))
print("      measured Route 2   %8.4f /s   ->  short by %.2fx" %
      (MEAS_ADD_RATE_R2, need_ / (SAFETY * MEAS_ADD_RATE_R2)))
print("      measured Route 1   %8.4f /s   ->  short by %.2fx" %
      (MEAS_ADD_RATE_R1, need_ / (SAFETY * MEAS_ADD_RATE_R1)))
print()
print("  the 1.02x near-miss exists only on the raw update rate,")
print("  the one row that counts cancels and trades as submissions;")
print("  they are about half the stream, so that row is not a lam ceiling")
print()

# ---------------------------- 7. consequence for what is already committed ---

rule()
print("7. Consequences for COMMITTED results")
rule()
print()
print("  (a) recorded bound in NOTES.md (verbatim):")
print()
print('        "the real book updates at ~3.09 IDs/sec, and that count includes')
print('         submissions AND cancels AND trades, so lam must be a FRACTION of')
print('         3.09, not a multiple. lam=1.804 is already at 58% of that')
print('         ceiling. DEPTH CANNOT BE BOUGHT WITH lam ALONE."')
print()
print("      58% recomputed per candidate ceiling:")
print()
print("        %-42s  %8s  %9s  %s" % ("ceiling", "IDs/s", "lam share", "reading"))
for name, c in CEILINGS:
    share = 100.0 * LAM_HELD / c
    if share >= 50.0:
        reading = "at/over half the ceiling -- bound BITES"
    elif share >= 25.0:
        reading = "a quarter to a half -- bound SOFTENS"
    else:
        reading = "a small fraction -- bound DOES NOT BITE"
    print("        %-42s  %8.4f  %8.2f%%  %s" % (name, c, share, reading))
print()
print("      clause by clause:")
print()
print("        'the real book updates at ~3.09 IDs/sec'")
print("            FALSE: recorded rates are near %.1f IDs/sec" %
      r_notesmd)
print("            (section 3), and 3.09 has no measuring artifact at all.")
print()
print("        'that count includes submissions AND cancels AND trades'")
print("            TRUE: makes a submission ceiling a fraction of the update rate")
print()
print("        'lam=1.804 is already at 58% of that ceiling'")
print("            correct against 3.09 (%.2f%%); wrong" %
      (100.0 * LAM_HELD / CEIL_QUOTED))
print("            against every sourced figure: %.2f%% of the raw measured" %
      (100.0 * LAM_HELD / r_notesmd))
print("            rate, %.2f%% of the half-rate submission bound." %
      (100.0 * LAM_HELD / (r_notesmd / 2.0)))
print()
print("        'DEPTH CANNOT BE BOUGHT WITH lam ALONE'")
print("            not established by this argument; rests on the divide-by-2 step, since measured (see (c))")
print()
print("      separately: the")
print("      fixed book size, %.1f levels/side at life=720, is" %
      base[1])
print("      itself only %.2f%% of the real %.2f target (lifetime_sweep_results" %
      (100.0 * base[1] / REAL_LEVELS_TARGET, REAL_LEVELS_TARGET))
print("      .txt:49). It passes the +/-20% band and it is the calibration's own")
print("      working point; reaching the target too would need a")
print("      further lam factor of x%.4f on everything in section 4." %
      (REAL_LEVELS_TARGET / base[1]))
print()
print("  (b) COMMITTED sites using 3.09 as a live constant (flag only):")
print("      prints the figure, or constrains a search?")
print()
print("      SITE 1  clipped_lam_sweep.py (Phase 3, f9293d8)")
print("        line 67   REAL_UPDATE_RATE = 3.09   -- the ORIGIN of the figure")
print("        line 216  used to set first_309 when lam >= REAL_UPDATE_RATE")
print("        line 242  printed as 'lam crosses the real ~3.09 IDs/s ...'")
print("        verdict: reports only; top row lam=4.5100 is above 3.09, so the sweep went past it")
print()
print("      SITE 2  clipped_depth_sweep.py (Phase 4)")
print("        line 76   LAM_CEILING = 3.09")
print("        line 294  printed as 'lam is 58% of the ~3.09 IDs/s ... ceiling'")
print("        header reasons from it to 'lam cannot be the depth knob'")
print("        line 74   LAM = 1.804 pinned; LIVES and DISPS swept")
print("        verdict: constrains the search, worst case:")
print("        lam REMOVED from the search space; Phase 4 swept (life, disp) only")
print("        on an unsourced number")
print()
print("      SITE 3  clipped_pmarket_grid.py (Phase 5)")
print("        line 50   imports LAM and LAM_CEILING from clipped_depth_sweep")
print("        line 141  prints the ceiling text")
print("        header line 37  'lam = 1.804 held throughout'")
print("        verdict: inherits Site 2's pin")
print()
print("      SITE 4  lifetime_sweep.py (78701e4) -- INHERITS THE CONSEQUENCE")
print("        without naming the constant")
print("        line 87   LAM, P_MARKET, DISP = 1.804, 0.0126, 0.0055 hardcoded")
print("        verdict: lam held at Site 2's pinned value")
print()
print("      summary: 1 site reports, 1 removed an axis from the search, 2 inherited the removal")
print()
print("  (c) claim REFUTED in its premise")
print()
print("      the (lam, mean_lifetime) lever arithmetic is right, but the ceiling is an")
print("      unsourced constant, never measured, cited by three later files")
print()
print("      right by accident, not vindicated: 'arithmetically dead' relied on a ~7x gap;")
print("      against the raw measured update rate the gap is %.0f%%" %
      (100.0 * (need_ / (SAFETY * r_notesmd) - 1.0)))
print("      the verdict survives only via the divide-by-2 step")
print()
print("      durable: an update ID is any")
print("      book change, lam counts submissions only, so lam is bounded by")
print("      some fraction of the update rate")
print()
print("      divide-by-2 measured (book_change_mix.py): %.6f against the assumed"
      % MEAS_ADD_SHARE)
print("      0.500000; the verdict no longer rests on an unmeasured step")
print()
print("      the add-vs-non-add split is measurable from depth (level quantity UP or DOWN);")
print("      only the add / cancel / trade split needs more than depth")
print()
print("      remaining soft spot: update rate spans %.4f to %.4f IDs/s (~%.0f%% apart on adjacent"
      % (r_cloud, r_2day, 100.0 * (r_2day / r_cloud - 1.0)))
print("      multi-day windows); addition rate inherits it:")
print("      the ceiling is a distribution; a verdict decided by less than ~50% spread is not decided")
print()
print("      the life=60 verdict is short by")
print("      %.2fx even on the most generous measured row, well" %
      (need_ / (SAFETY * MEAS_ADD_RATE_R2)))
print("      outside the spread; that verdict survives. A verdict turning on")
print("      the difference between, say, 6.3 and 7.2 would not.")
print()
print("      not attempted: the aggTrade stream could split DOWN and REMOVED into cancels vs trades")
rule()
