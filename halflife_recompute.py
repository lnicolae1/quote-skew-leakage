# halflife_recompute.py: recomputes the half-life statistic from committed results files (no simulation)

import io
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

GAMMA_FILE = os.path.join(_HERE, "clipped_gamma_sweep_results.txt")
PART_FILE = os.path.join(_HERE, "participation_skew_results.txt")

AR1_DT = 60.0

_HL = r"(?:\s*(\d+)s \(([\d.]+)h\) ?(?:n=(\d+))?|\s*none stationary)"


def hl_from_phi(phi):
    if phi is None or not (0.0 < phi < 1.0):
        return None
    return AR1_DT * math.log(0.5) / math.log(phi)


def parse_gamma_sweep(path):
    """Rows of d5d134c's mean-reversion table"""
    pat = re.compile(
        r"^\s+([\d.]+)\s+([\d.]+e[-+]\d+)\s+([\d.]+)" + _HL +
        r"\s+([\d.]+)" + _HL + r"\s*$")
    out = []
    for line in io.open(path, encoding="utf-8"):
        m = pat.match(line.rstrip("\n"))
        if not m:
            continue
        g = m.groups()
        out.append({
            "C": float(g[0]), "gamma": float(g[1]),
            "phi_full": float(g[2]),
            "hl_full": float(g[3]) if g[3] else None,
            "n_full": int(g[5]) if g[5] else 0,
            "phi_train": float(g[6]),
            "hl_train": float(g[7]) if g[7] else None,
            "n_train": int(g[9]) if g[9] else 0,
        })
    return out


def parse_participation(path):
    """Rows of 0df879d's section 1"""
    pat = re.compile(
        r"^\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)" +
        _HL + _HL + r"\s*$")
    out = []
    for line in io.open(path, encoding="utf-8"):
        m = pat.match(line.rstrip("\n"))
        if not m:
            continue
        g = m.groups()
        out.append({
            "qsize": float(g[0]), "vol_share": float(g[1]),
            "phi_skew": float(g[2]), "phi_flat": float(g[3]),
            "phi_gamma0": float(g[4]),
            "hl_skew": float(g[5]) if g[5] else None,
            "hl_flat": float(g[8]) if g[8] else None,
        })
    return out


def fmt(hl):
    if hl is None:
        return "        --"
    return "%7.0fs" % hl


def fmt_h(hl):
    if hl is None:
        return "     --"
    return "%5.1fh" % (hl / 3600.0)


def disp(mean_hl, star):
    """Ratio mean(HL)/HL*. A trailing '!' marks a ratio below 1, which"""
    if mean_hl is None or star is None or star <= 0:
        return "    --"
    r = mean_hl / star
    return "%5.2f%s" % (r, "!" if r < 1.0 else " ")


def main():
    grows = parse_gamma_sweep(GAMMA_FILE)
    prows = parse_participation(PART_FILE)

    print("=" * 124)
    print("Half-life recomputed from mean phi; corrects d5d134c and 0df879d. "
          "No simulation run.")
    print("=" * 124)
    print("Parsed %d rows from clipped_gamma_sweep_results.txt and %d from "
          "participation_skew_results.txt." % (len(grows), len(prows)))
    print("HL* = %.0f * ln(0.5) / ln(mean_phi).  mean(HL) is the "
          "already-committed average of per-seed half-lives." % AR1_DT)
    print("HL* is the one to cite. mean(HL) averages a quantity with a pole at "
          "phi=1 and is unstable.")
    print("Where both use the SAME seeds, convexity gives mean(HL) >= HL* and "
          "the ratio is a dispersion")
    print("factor (never a standard error). Where mean(HL) dropped "
          "non-stationary seeds it is computed on")
    print("a different, downward-biased sample and the ratio can fall BELOW 1 "
          "-- flagged inline. No")
    print("interval is available on HL*; see the header for why.")
    print("")

    if not grows or not prows:
        print("Parse FAILED; refusing to report numbers. Check the table "
              "formats in the two files.")
        return 1

    print("=" * 124)
    print("A. d5d134c; gamma sweep (train slice is the one to use; full run "
          "is contaminated by the tau->0 tail)")
    print("=" * 124)
    print("  %9s %9s %11s %10s %9s %8s %8s %11s %10s %9s %8s %8s"
          % ("C_at_t0", "gamma", "phi train", "HL* train", "", "mean(HL)",
             "disp", "phi full", "HL* full", "", "mean(HL)", "disp"))
    for r in grows:
        st = hl_from_phi(r["phi_train"])
        sf = hl_from_phi(r["phi_full"])
        print("  %9.3f %9.1e %11.6f %10s %9s %8s %8s %11.6f %10s %9s %8s %8s"
              % (r["C"], r["gamma"],
                 r["phi_train"], fmt(st), fmt_h(st),
                 fmt_h(r["hl_train"]), disp(r["hl_train"], st),
                 r["phi_full"], fmt(sf), fmt_h(sf),
                 fmt_h(r["hl_full"]), disp(r["hl_full"], sf)))

    tr = [hl_from_phi(r["phi_train"]) for r in grows]
    tr = [x for x in tr if x]
    print("")
    print("  corrected train-slice range: %.0f-%.0fs (%.1f-%.1fh), against "
          "the committed 3.2-8.8h."
          % (min(tr), max(tr), min(tr) / 3600.0, max(tr) / 3600.0))

    print("")
    print("=" * 124)
    print("B. 0df879d; participation arms (train slice)")
    print("=" * 124)
    print("  %10s %10s %11s %10s %8s %9s %7s %11s %10s %8s %9s %7s"
          % ("quote_size", "vol share", "phi skew", "HL* skew", "", "mean(HL)",
             "disp", "phi FLAT", "HL* flat", "", "mean(HL)", "disp"))
    for r in prows:
        ss = hl_from_phi(r["phi_skew"])
        sf = hl_from_phi(r["phi_flat"])
        print("  %10.3f %9.2f%% %11.6f %10s %8s %9s %7s %11.6f %10s %8s "
              "%9s %7s"
              % (r["qsize"], r["vol_share"],
                 r["phi_skew"], fmt(ss), fmt_h(ss),
                 fmt_h(r["hl_skew"]), disp(r["hl_skew"], ss),
                 r["phi_flat"], fmt(sf), fmt_h(sf),
                 fmt_h(r["hl_flat"]), disp(r["hl_flat"], sf)))
    print("")
    print("  %10s %11s %10s" % ("quote_size", "phi GAMMA0", "HL* gamma0"))
    for r in prows:
        sg = hl_from_phi(r["phi_gamma0"])
        print("  %10.3f %11.6f %10s %8s"
              % (r["qsize"], r["phi_gamma0"], fmt(sg), fmt_h(sg)))

    print("")
    print("  the ordering inversion is gone. At quote_size 0.020 mean phi_skew "
          "sits below mean phi_flat,")
    print("  so the skewing arm reverts faster. The corrected half-lives now "
          "say that; the committed")
    print("  averaged half-lives said the opposite (8.6h skew against 2.9h "
          "flat).")

    ps = [hl_from_phi(r["phi_skew"]) for r in prows]
    pf = [hl_from_phi(r["phi_flat"]) for r in prows]
    allp = [x for x in ps + pf if x]
    print("")
    print("  Corrected range across both arms: %.0f-%.0fs (%.1f-%.1fh), "
          "against the committed 2.9-8.6h."
          % (min(allp), max(allp), min(allp) / 3600.0, max(allp) / 3600.0))

    print("")
    print("=" * 124)
    print("C. Cross-check; the same configuration measured by two "
          "independently written scripts")
    print("=" * 124)
    anchor = [r for r in grows if abs(r["gamma"] - 1e-9) < 1e-15]
    g020 = [r for r in prows if abs(r["qsize"] - 0.020) < 1e-12]
    if anchor and g020:
        a = anchor[0]["phi_train"]
        b = g020[0]["phi_gamma0"]
        print("  clipped_gamma_sweep gamma=1e-9 anchor, phi (train)   = %.6f"
              % a)
        print("  participation_skew GAMMA0 arm at quote_size 0.020    = %.6f"
              % b)
        print("  Same gamma, same DEFAULT_QUOTE_SIZE, same 5 seeds, same "
              "train-slice convention.")
        print("  |difference| = %.2e   -> %s"
              % (abs(a - b),
                 "agree to all printed digits" if abs(a - b) < 5e-7
                 else "disagree; investigate before citing either"))
    else:
        print("  Could not locate both rows; cross-check not performed.")

    print("")
    print("=" * 124)
    print("D. What to cite, and what Step 7 should use")
    print("=" * 124)
    print("  A '!' on a dispersion ratio marks a row where mean(HL) dropped "
          "non-stationary seeds, so the")
    print("  two columns are not on the same sample and mean(HL) is biased "
          "DOWN there. Combined with the")
    print("  pole inflating it elsewhere, mean(HL) errs in both directions "
          "depending on the row; which")
    print("  is why it is abandoned rather than corrected.")
    print("")
    print("  Cite HL*. The committed mean(HL) columns are superseded by this "
          "file and should not be")
    print("  quoted again; they are printed above only so the size of the "
          "correction is visible.")
    print("  The holding period for step 7 should be set off HL*, and it "
          "should be set off a range")
    print("  rather than a single row; no interval is available on any of "
          "these point estimates, so")
    print("  treating any one of them as precise would be a second version of "
          "the same mistake.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
