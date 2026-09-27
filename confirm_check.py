# confirm_check.py: checks whether the working-point confirmation (clipped_confirm) reproduces

import math
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from clipped_mm_check import ABSENT

OUTPUT_FILE = os.path.join(_HERE, "clipped_confirm_results.txt")
SOURCE_COMMIT = "847a7fa"
TARGET_KEYS = (("median width bp", "median_bp", "width"),
               ("levels/side", "levels_per_side", "levels"),
               ("aggTrades/s", "agg_per_s", "rate"),
               ("mean/median shape", "shape", "shape"))
DIAG_KEYS = (("near-touch BTC (<=1bp)", "near_btc"),
             ("near-touch levels (<=1bp)", "near_lvl"),
             ("touch-order age (s)", "touch_age"))


def _num(s, what):
    v = float(s)
    if not math.isfinite(v):
        raise ValueError("NON-FINITE %s: %r" % (what, s))
    return v, (len(s.split(".")[1]) if "." in s else 0)


def parse_output(lines):
    """The figures clipped_confirm.py prints: {key: (value, decimals)} plus"""
    out = {"verdict": {}, "margin": {}}
    names = "|".join(re.escape(n) for n, _k, _m in TARGET_KEYS)
    tpat = re.compile(r"^\s{2}(%s)\s+(-?[0-9.]+) \+/- ([0-9.]+)\s+95%% \[.*\]"
                      r"\s+band \[.*\]\s+(\w+)\s*$" % names)
    mpat = re.compile(r"^\s{2}(%s)\s+([0-9.]+) SE from the (lower|upper) edge"
                      % names)
    dnames = "|".join(re.escape(n) for n, _k in DIAG_KEYS)
    dpat = re.compile(r"^\s{2}(%s)\s+([0-9.]+) \+/- " % dnames)
    opat = re.compile(r"^\s{2}orders/side ([0-9.]+) \+/- ([0-9.]+)\s+levels/side"
                      r" ([0-9.]+) \+/- ([0-9.]+)\s+mean bp ([0-9.]+)")
    key_of = {n: k for n, k, _m in TARGET_KEYS}
    dkey_of = dict(DIAG_KEYS)
    for ln in lines:
        m = tpat.match(ln)
        if m:
            k = key_of[m.group(1)]
            out[k] = _num(m.group(2), k)
            out[k + "_se"] = _num(m.group(3), k + " SE")
            out["verdict"][k] = m.group(4)
            continue
        m = mpat.match(ln)
        if m:
            out["margin"][key_of[m.group(1)]] = _num(m.group(2), "margin")
            continue
        m = dpat.match(ln)
        if m:
            out[dkey_of[m.group(1)]] = _num(m.group(2), m.group(1))
            continue
        m = opat.match(ln)
        if m:
            out["orders_per_side"] = _num(m.group(1), "orders/side")
            out["mean_bp"] = _num(m.group(5), "mean bp")
            continue
        m = re.match(r"^\s{2}-> (\d)/4 MET\s*$", ln)
        if m:
            out["met"] = int(m.group(1))
    for _n, k, _m in TARGET_KEYS:
        if k not in out or k not in out["margin"]:
            raise ValueError("output is missing target %s" % k)
    if "met" not in out:
        raise ValueError("output has no MET count")
    return out


def parse_message(text):
    """Margins and the MET count recorded in the 847a7fa message"""
    flat = " ".join(text.split())
    m = re.search(r"width ([0-9.]+) SE, levels ([0-9.]+) SE, rate ([0-9.]+) "
                  r"SE, shape ([0-9.]+) SE", flat)
    if not m:
        raise ValueError("847a7fa message: margins sentence not found")
    met = re.search(r"(\d)/4 targets MET", flat)
    if not met:
        raise ValueError("847a7fa message: MET count not found")
    return {"margin": {"width": float(m.group(1)), "levels": float(m.group(2)),
                       "rate": float(m.group(3)), "shape": float(m.group(4))},
            "met": int(met.group(1))}


def _ref_decimals(v):
    s = repr(v)
    return len(s.split(".")[1]) if "." in s else 0


def compare(out, absent, msg):
    """Prints every comparison; returns (reproduces, diag_ok)"""
    ok_a = True
    print("  (a) targets vs clipped_mm_check.ABSENT")
    for name, k, _m in TARGET_KEYS:
        for kk in (k, k + "_se"):
            got, d_out = out[kk]
            want = absent[kk]
            tol = 0.5 * 10.0 ** (-min(d_out, _ref_decimals(want)))
            ok = abs(got - want) <= tol
            ok_a &= ok
            print("      %-22s %-4s got %12.5f  committed %12.5f  tol %.0e  %s"
                  % (name, "SE" if kk.endswith("_se") else "", got, want, tol,
                     "OK" if ok else "*** miss ***"))
    ok_b = out["met"] == 4 and msg["met"] == 4 and \
        all(v == "MET" for v in out["verdict"].values())
    print("  (b) verdicts: %s; %d/4 MET (message: %d/4)  %s"
          % (", ".join("%s %s" % (k, v) for k, v in out["verdict"].items()),
             out["met"], msg["met"], "OK" if ok_b else "*** miss ***"))
    ok_c = True
    print("  (c) margins vs the %s message" % SOURCE_COMMIT)
    for name, k, mk in TARGET_KEYS:
        got = out["margin"][k][0]
        want = msg["margin"][mk]
        ok = round(got, 2) == round(want, 2)
        ok_c &= ok
        print("      %-22s got %6.2f SE  recorded %6.2f SE  %s"
              % (name, got, want, "OK" if ok else "*** miss ***"))
    diag_ok = True
    print("  diagnostics (reported, not part of the reading)")
    for k in ("near_btc", "near_lvl", "touch_age", "orders_per_side",
              "mean_bp"):
        got, d_out = out[k]
        want = absent[k]
        tol = 0.5 * 10.0 ** (-min(d_out, _ref_decimals(want)))
        ok = abs(got - want) <= tol
        diag_ok &= ok
        print("      %-22s got %12.4f  committed %12.4f  tol %.0e  %s"
              % (k, got, want, tol, "match" if ok else "differs"))
    rep = ok_a and ok_b and ok_c
    return rep, diag_ok


def _planted_output(absent, margins, met=4):
    lines = []
    for name, k, _m in TARGET_KEYS:
        lines.append("  %-20s %10.5f +/- %-9.5f  95%% [%9.5f, %9.5f]  band "
                     "[%9.5f, %9.5f]  %s" % (name, absent[k], absent[k + "_se"],
                                            0, 0, 0, 0, "MET"))
    lines.append("")
    lines.append("  -> %d/4 MET" % met)
    for name, _k, mk in TARGET_KEYS:
        lines.append("  %-20s %8.2f SE from the lower edge" % (name, margins[mk]))
    for label, k in DIAG_KEYS:
        lines.append("  %-28s %10.4f +/- %-8.4f  vs %9.4f   %.2fx"
                     % (label, absent[k], 0.01, 1.0, 1.0))
    lines.append("  orders/side %.1f +/- %.1f   levels/side %.1f +/- %.1f   "
                 "mean bp %.4f   clipped %.1f%%   coherence %.3f"
                 % (absent["orders_per_side"], 1.0, 551.2, 2.5,
                    absent["mean_bp"], 30.0, 0.5))
    return lines


def selftest():
    import contextlib
    import io
    print("=" * 100)
    print("Self-tests; confirm_check.py. clipped_confirm has not been re-run.")
    print("=" * 100)
    text = subprocess.run(["git", "show", "-s", "--format=%B", SOURCE_COMMIT],
                          cwd=_HERE, capture_output=True, text=True,
                          check=True).stdout
    msg = parse_message(text)
    assert msg["met"] == 4
    assert msg["margin"] == {"width": 4.39, "levels": 7.15, "rate": 5.55,
                             "shape": 4.15}, msg
    print("  T-message   PASS  the %s message parses: 4/4 MET, margins 4.39 / "
          "7.15 / 5.55 / 4.15 SE." % SOURCE_COMMIT)
    buf = io.StringIO()
    out = parse_output(_planted_output(ABSENT, msg["margin"]))
    with contextlib.redirect_stdout(buf):
        rep, diag = compare(out, ABSENT, msg)
    assert rep and diag, "planted exact output did not reproduce"
    bad = dict(ABSENT, levels_per_side=551.24755)
    out2 = parse_output(_planted_output(bad, msg["margin"]))
    with contextlib.redirect_stdout(buf):
        rep2, _d = compare(out2, ABSENT, msg)
    assert not rep2, "a 0.01 shift in levels/side passed"
    m3 = dict(msg["margin"], rate=5.56)
    out3 = parse_output(_planted_output(ABSENT, m3))
    with contextlib.redirect_stdout(buf):
        rep3, _d = compare(out3, ABSENT, msg)
    assert not rep3, "a margin off by 0.01 SE passed"
    out4 = parse_output(_planted_output(ABSENT, msg["margin"], met=3))
    with contextlib.redirect_stdout(buf):
        rep4, _d = compare(out4, ABSENT, msg)
    assert not rep4, "3/4 MET passed"
    try:
        parse_output(["nothing here"])
        raise AssertionError("an empty output parsed")
    except ValueError:
        pass
    print("  T-compare   PASS  planted output in clipped_confirm's exact format "
          "reproduces; a 0.01 shift")
    print("                    in levels/side, a margin 0.01 SE off, and 3/4 MET "
          "each fail; an empty output raises.")
    print("")
    print("  All self-tests PASSED.")
    return msg


def main(msg):
    print("=" * 100)
    print("Does the working-point confirmation reproduce?; %s vs %s"
          % (os.path.basename(OUTPUT_FILE), SOURCE_COMMIT))
    print("=" * 100)
    with open(OUTPUT_FILE, encoding="utf-8-sig") as f:
        out = parse_output(f.read().splitlines())
    rep, diag = compare(out, ABSENT, msg)
    print("")
    print("  The confirmation %s.%s" % (
        "reproduces" if rep else "Does not reproduce",
        "" if diag else "  (Some diagnostics differ; see above.)"))
    return rep


if __name__ == "__main__":
    m = selftest()
    if "--selftest" in sys.argv:
        sys.exit(0)
    print("")
    sys.exit(0 if main(m) else 1)
