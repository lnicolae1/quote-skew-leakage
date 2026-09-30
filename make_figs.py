"""make_figs.py [results_dir]: paper figures 1-4 into figs/; all values via fig_data.py."""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import fig_data

for p in font_manager.findSystemFonts():
    if "texgyrepagella" in p.lower():
        font_manager.fontManager.addfont(p)

INK = "#1a1a1a"
MUTED = "#6b6b6b"
BAND = "#e9e9e9"
SEED = "#a3a3a3"
QUANT = "#0072b2"    # colour-vision-safe pair
JITTER = "#d55e00"
Z = 1.96

plt.rcParams.update({
    "font.family": "TeX Gyre Pagella",
    "mathtext.fontset": "custom",
    "mathtext.rm": "TeX Gyre Pagella",
    "mathtext.it": "TeX Gyre Pagella:italic",
    "mathtext.bf": "TeX Gyre Pagella:bold",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "lines.linewidth": 1.3,
    "pdf.fonttype": 3,
})

D = fig_data.all_data()


def panel_label(ax, s):
    ax.text(0.0, 1.05, s, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=INK)


_hidden = []


def fig_calibration():
    c = D["calibration"]
    rows = [("Maker absent", c["absent"]), ("Maker present (C)", c["cfg_c"]),
            ("Maker present (B)", c["cfg_b"])]
    fig, axs = plt.subplots(1, 4, figsize=(6.3, 1.6), gridspec_kw={"wspace": 0.3})
    for j, ax in enumerate(axs):
        lo, hi = c["bands"][j]
        ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
        ax.axvline(c["real"][j], color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.set_ylim(-0.6, 3.0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(c["titles"][j], fontsize=8.5, color=INK, pad=3)
        ax.text(c["real"][j], 2.62, " empirical", fontsize=7, color=MUTED, ha="left", va="bottom")
        pts = [lo, hi, c["real"][j]] + [v[j][0] + s * Z * v[j][1] for _, v in rows for s in (-1, 1)]
        span = max(pts) - min(pts)
        ax.set_xlim(min(pts) - 0.08 * span, max(pts) + 0.08 * span)
        ax.locator_params(axis="x", nbins=3)
        pts_per_unit = ax.get_position().width * fig.get_figwidth() * 72 / (ax.get_xlim()[1] - ax.get_xlim()[0])
        for i, (_, vals) in enumerate(rows):
            m, se = vals[j]
            if Z * se * pts_per_unit > 1.6:          # else hidden by marker (caption notes)
                ax.errorbar(m, 2 - i, xerr=Z * se, fmt="o", ms=3.2, color=INK, elinewidth=0.9,
                            capsize=2, zorder=3)
            else:
                ax.plot(m, 2 - i, "o", ms=3.2, color=INK, zorder=3)
                _hidden.append((c["titles"][j], round(Z * se * pts_per_unit, 2)))
    for i, (name, _) in enumerate(rows):
        axs[0].text(-0.06, 2 - i, name, transform=axs[0].get_yaxis_transform(), ha="right",
                    va="center", fontsize=8, color=INK)
    fig.savefig("figs/fig1_calibration.pdf")
    print("fig1 hidden intervals (half-width pt):", _hidden)
    plt.close(fig)


def fig_channels():
    ch = D["channels"]
    rows = [("Oracle: counterparty identity", ch["oracle"]),
            ("Normalized skew, residual mid", ch["norm"]),
            ("Raw skew, residual mid", ch["raw"]),
            ("Normalized skew, full-book mid", ch["full"]),
            ("Trade record only, signed flow", ch["tape"])]
    fig, ax = plt.subplots(figsize=(5.0, 1.9))
    ax.axvline(0, color=MUTED, lw=0.6, zorder=0)
    n = len(rows)
    for i, (name, (m, se)) in enumerate(rows):
        y = n - 1 - i
        ax.errorbar(m, y, xerr=(Z * se) if se else None, fmt="o", ms=3.4, color=INK,
                    elinewidth=0.9, capsize=2)
        txt = "1 (exact)" if se == 0 else f"{m:.2f}".replace("-", "\u2212")
        ax.text(m + (Z * se if se else 0) + 0.3, y, txt, va="center", ha="left", fontsize=7.5, color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-11.5, 3.2)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(r"Out-of-sample $R^2$ for the maker's inventory (mean and 95% interval)")
    fig.savefig("figs/fig2_channels.pdf")
    plt.close(fig)


def fig_smoothing():
    s = D["smoothing"]
    M = s["M"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.4),
                               gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.42})
    for (mu, se), col, mk, ms in ((s["jit"], JITTER, "^", 3.6), (s["qua"], QUANT, "s", 3.0)):
        a.fill_between(M, [m - Z * e for m, e in zip(mu, se)], [m + Z * e for m, e in zip(mu, se)],
                       color=col, alpha=0.15, lw=0)
        a.plot(M, mu, color=col, marker=mk, ms=ms)
    a.plot(M, s["base"][0], color=INK, marker="o", ms=2.8, lw=1.0)
    a.axhline(0, color=MUTED, lw=0.6, zorder=0)
    a.set_xscale("log")
    a.set_xticks(M)
    a.set_xticklabels([str(m) for m in M])
    a.minorticks_off()
    a.set_ylim(-19, 3.4)
    a.set_xlim(0.8, 420)
    a.set_xlabel(r"Attacker's averaging window $M$ (seconds)")
    a.set_ylabel(r"Out-of-sample $R^2$")
    a.text(1.1, 2.35, "Undefended", color=INK, fontsize=8, va="center")
    a.text(1.1, -0.95, "Quantized", color=QUANT, fontsize=8, va="center")
    a.text(1.45, -13.4, "Jittered", color=JITTER, fontsize=8, va="center")
    panel_label(a, "(a) Averaging window, 8 seeds")
    ins = a.inset_axes([0.60, 0.20, 0.36, 0.30])
    for (mu, se), col, mk in ((s["base"], INK, "o"), (s["qua"], QUANT, "s")):
        ins.fill_between(M, [m - Z * e for m, e in zip(mu, se)], [m + Z * e for m, e in zip(mu, se)],
                         color=col, alpha=0.15, lw=0)
        ins.plot(M, mu, color=col, marker=mk, ms=2.2, lw=0.9)
    ins.set_xscale("log")
    ins.set_xticks([1, 10, 100])
    ins.set_xticklabels(["1", "10", "100"])
    ins.minorticks_off()
    ins.set_ylim(0.6, 1.05)
    ins.tick_params(labelsize=6.5, length=2)
    ins.set_title("enlarged", fontsize=6.5, color=MUTED, pad=2)

    x = [0, 1, 2]
    (jc, jcs), (qc, qcs) = s["jit_climb"], s["qua_climb"]
    b.axhline(0, color=MUTED, lw=0.6, zorder=0)
    b.errorbar([v - 0.09 for v in x], jc, yerr=[Z * e for e in jcs], fmt="^", ms=4, color=JITTER,
               elinewidth=0.9, capsize=2)
    b.errorbar([v + 0.09 for v in x], qc, yerr=[Z * e for e in qcs], fmt="s", ms=3.5, color=QUANT,
               elinewidth=0.9, capsize=2)
    b.set_xticks(x)
    b.set_xticklabels([f"{v:.2f}" for v in s["levels"]])
    b.set_xlim(-0.5, 2.5)
    b.set_ylim(-0.8, 11.8)
    b.set_xlabel(r"Informed arrival rate / $\lambda$")
    b.set_ylabel(r"Excess rise in $R^2$")
    b.text(-0.05, jc[0] + Z * jcs[0] + 0.5, "Jittered", color=JITTER, fontsize=8, ha="center")
    b.text(0.12, 0.9, "Quantized", color=QUANT, fontsize=8, va="center")
    panel_label(b, "(b) Three informed-trading rates, 24 seeds")
    fig.savefig("figs/fig3_smoothing.pdf")
    plt.close(fig)


def fig_width():
    w = D["width"]
    seeds = w["seeds"]
    arms = ["Q-0.75", "Q-1.00", "Q-1.25", "Q-1.50"]
    xs = [0, 1, 2, 3]
    xflat = 4.4
    ymin = -12.0
    rng = random.Random(1)  # per-seed x jitter
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.5),
                               gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.38})
    a.axhline(0, color=MUTED, lw=0.6, zorder=0)
    for x, arm in list(zip(xs, arms)) + [(xflat, "FLAT")]:
        vals = [seeds[arm][k] for k in range(24)]
        shown = [v for v in vals if v >= ymin]
        a.scatter([x + rng.uniform(-0.17, 0.17) for _ in shown], shown, s=5, color=SEED, lw=0, zorder=1)
        below = len(vals) - len(shown)
        if below:
            a.annotate(f"{below} seeds\nbelow {ymin:g}".replace("-", "\u2212"), xy=(x + 0.22, ymin),
                       xytext=(x + 0.3, ymin + 1.6), fontsize=7, color=MUTED, ha="left", va="center",
                       arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, shrinkA=0, shrinkB=0,
                                       mutation_scale=6))
    a.errorbar(xs, w["r2"], yerr=[Z * e for e in w["r2se"]], fmt="s", ms=4, color=QUANT,
               elinewidth=1.0, capsize=2.5, zorder=3)
    fm, fs = w["flat"]
    a.errorbar([xflat], [fm], yerr=[Z * fs], fmt="o", ms=4, color=INK, elinewidth=1.0, capsize=2.5, zorder=3)
    a.plot(xs, w["cap"], color=INK, lw=0.9, ls=(0, (1, 1.5)), zorder=2)
    a.text(3.2, w["cap"][-1], "analytic cap", fontsize=7.5, color=INK, va="center", ha="left")
    a.set_xticks(xs + [xflat])
    a.set_xticklabels([f"{v:.2f}" for v in w["w"]] + ["Zero\nskew"])
    a.set_xlim(-0.5, xflat + 0.95)
    a.set_ylim(ymin, 1.6)
    a.set_xlabel(r"Bucket width $w$ / sd($q$)")
    a.set_ylabel(r"Out-of-sample $R^2$")
    panel_label(a, r"(a) Attacker's out-of-sample $R^2$")

    b.axhline(0, color=MUTED, lw=0.6, zorder=0)
    gx = list(range(len(w["gain_w"])))
    b.errorbar(gx, w["gain"], yerr=[Z * e for e in w["gain_se"]], fmt="s", ms=4, color=QUANT,
               elinewidth=1.0, capsize=2.5)
    b.set_xticks(gx)
    b.set_xticklabels([f"{v:.2f}" for v in w["gain_w"]])
    b.set_xlim(-0.5, len(gx) - 0.5)
    b.set_ylim(-0.004, 0.072)
    b.set_xlabel(r"Bucket width $w$ / sd($q$)")
    b.set_ylabel(r"$R_{\mathrm{direct}}$ minus zero skew")
    panel_label(b, "(b) Inventory control relative to zero skew")
    fig.savefig("figs/fig4_width.pdf")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    fig_calibration()
    fig_channels()
    fig_smoothing()
    fig_width()
    print("wrote figs/fig1_calibration.pdf, fig2_channels.pdf, fig3_smoothing.pdf, fig4_width.pdf")
