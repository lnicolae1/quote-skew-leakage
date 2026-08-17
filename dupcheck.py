# dupcheck.py: detector for duplicated print blocks and related defects

import ast
import re
import sys

CONV = re.compile(r"%[-+ #0]*[0-9*]*(?:\.[0-9*]+)?[hlL]?[diouxXeEfFgGcrsa%]")

BENIGN = {'print("")', "print()", "pass", "continue", "break", "else:", "try:"}

MIN_BLOCK = 2
MAX_BLOCK = 15


def read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def check_syntax(path, src):
    """ast.parse"""
    try:
        return ast.parse(src, filename=path), []
    except SyntaxError as e:
        return None, [("ERROR", e.lineno or 0,
                       "SyntaxError: %s (offset %s)" % (e.msg, e.offset))]


def check_dupes(src):
    """Consecutive duplicate lines and duplicate consecutive blocks"""
    lines = src.split("\n")
    out = []
    claimed = set()
    for L in range(MAX_BLOCK, MIN_BLOCK - 1, -1):
        for i in range(0, len(lines) - 2 * L + 1):
            first = lines[i:i + L]
            second = lines[i + L:i + 2 * L]
            if first != second:
                continue
            if not any(ln.strip() for ln in first):
                continue
            if any(n in claimed for n in range(i, i + 2 * L)):
                continue
            for n in range(i, i + 2 * L):
                claimed.add(n)
            out.append(("ERROR", i + 1,
                        "DUPLICATED BLOCK of %d lines: %d-%d repeats %d-%d "
                        "| first line: %s"
                        % (L, i + L + 1, i + 2 * L, i + 1, i + L,
                           first[0].strip()[:70])))
    for i in range(len(lines) - 1):
        if i in claimed or (i + 1) in claimed:
            continue
        a, b = lines[i], lines[i + 1]
        s = a.strip()
        if not s or a != b or s in BENIGN or len(s) < 8:
            continue
        out.append(("WARN", i + 1,
                    "consecutive identical lines at the same indent: %s"
                    % s[:70]))
    out.sort(key=lambda r: r[1])
    return out


def check_percent(tree):
    """Every string constant, classified by whether it is a format string"""
    fmt = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            left = node.left
            if isinstance(left, ast.Constant) and isinstance(left.value, str):
                fmt.add(id(left))
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        s = node.value
        if "%" not in s:
            continue
        ln = getattr(node, "lineno", 0)
        if id(node) in fmt:
            i = 0
            while i < len(s):
                if s[i] != "%":
                    i += 1
                    continue
                m = CONV.match(s, i)
                if not m:
                    out.append(("ERROR", ln,
                                "format string has a %% that opens no valid "
                                "conversion at offset %d: %s"
                                % (i, s[max(0, i - 25):i + 25])))
                    break
                i = m.end()
        else:
            if "%%" in s:
                j = s.index("%%")
                out.append(("ERROR", ln,
                            "UNFORMATTED string contains '%%%%' -- it will "
                            "print literally: %s"
                            % s[max(0, j - 25):j + 25]))
    out.sort(key=lambda r: r[1])
    return out


def run(path):
    src = read(path)
    tree, findings = check_syntax(path, src)
    findings = list(findings)
    findings += check_dupes(src)
    if tree is not None:
        findings += check_percent(tree)
    findings.sort(key=lambda r: (r[1], r[0]))

    nerr = sum(1 for f in findings if f[0] == "ERROR")
    nwarn = len(findings) - nerr
    print("=" * 100)
    print("dupcheck %s; %d lines, %d ERROR, %d warn"
          % (path, len(src.split("\n")), nerr, nwarn))
    print("  ast.parse: %s" % ("OK" if tree is not None else "FAILED"))
    print("=" * 100)
    if not findings:
        print("  clean: no duplicated blocks, no percent-literal defects.")
    for lvl, ln, msg in findings:
        print("  %-5s line %5d  %s" % (lvl, ln, msg))
    print("")
    return nerr


def selftest():
    """The detector must catch the exact defects it was written for"""
    print("=" * 100)
    print("dupcheck self-test; planted defects must be found")
    print("=" * 100)

    bad1 = ('x = 1\n'
            'print("aaaaaaaaaaaa")\n'
            'print("bbbbbbbbbbbb")\n'
            'print("cccccccccccc")\n'
            'print("aaaaaaaaaaaa")\n'
            'print("bbbbbbbbbbbb")\n'
            'print("cccccccccccc")\n')
    hits = check_dupes(bad1)
    got = [h for h in hits if h[0] == "ERROR"]
    print("  planted duplicated block (compiles fine) -> %d ERROR" % len(got))
    for h in got:
        print("      %s" % h[2])
    assert got, "detector missed a duplicated block"

    bad2 = 'print("hello "\nprint("hello ")\n'
    tr, fs = check_syntax("<planted>", bad2)
    print("  planted unterminated call -> ast.parse %s, %d finding"
          % ("FAILED" if tr is None else "OK", len(fs)))
    assert tr is None and fs, "detector missed an unterminated call"

    bad3 = ('print("asserts ~28%% in prose")\n'
            'print("value %d and 50%" % n)\n'
            'print("ok literal 28% here")\n'
            'print("ok %d pct and a real %% sign" % n)\n')
    tr3, _ = check_syntax("<planted>", bad3)
    hits3 = check_percent(tr3)
    print("  planted percent defects -> %d ERROR" % len(hits3))
    for h in hits3:
        print("      line %d: %s" % (h[1], h[2]))
    assert len(hits3) == 2, "expected exactly 2 percent findings, got %d" \
                            % len(hits3)
    assert {h[1] for h in hits3} == {1, 2}, \
        "flagged the wrong lines: %s" % [h[1] for h in hits3]
    print("  self-test PASS; both defect classes detected, both correct "
          "forms left alone.")
    print("")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--selftest"]
    selftest()
    bad = 0
    for p in args:
        bad += run(p)
    sys.exit(1 if bad else 0)
