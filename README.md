# Hiding a Market Maker's Inventory

Code and results for the paper *Hiding a Market Maker's Inventory: Quantized and Jittered Quote Skew against an Explicit Attacker* (Luca Nicolae-Jozwiakowski, 2026).

The paper builds an explicit attacker that recovers an Avellaneda–Stoikov market maker's inventory from its quotes. It then tests two defenses, quantized skew and jittered skew, in an agent-based limit order book calibrated to Binance.US BTC/USD.

## Contents

**Data collection and reconstruction**
- `binance_orderbook_collector.py`: records the public Binance.US WebSocket streams.
- `reconstruct_book.py`: rebuilds the order book from a depth snapshot and the incremental depth stream.
- `data_manifest_sha256.txt`: SHA-256 fingerprints of the raw data files. The raw data are not redistributed.

**Simulator**
- `sim_run.py`: runs the simulation.
- `matching_engine.py`: the matching engine.
- `fundamental_value.py`: the fundamental value process.
- `noise_traders.py`, `informed_traders.py`, `clipped_placement.py`: the other traders.
- `market_maker.py`: the market maker.
- `trade_sizes.bin`: the empirical trade-size sample the simulator resamples.

**Experiments**

Each experiment has a script and a `*_results.txt` file holding its printed output. The pre-registered scripts state their decision rule in the header and were committed before they were run.

| Paper section | Results files |
|---|---|
| Empirical targets (Table 1) | `july_citability_results.txt`, `spread_depth_results.txt`, `volatility_clean_results.txt`, `trade_size_results.txt`, `trade_sign_acf_results.txt` |
| Fill-decay constant k | `measure_k_option1_results.txt` |
| Calibration (Figure 1) | `clipped_confirm_results.txt`, `cluster_b_calibration_results.txt`, `phase8_toxicity_results.txt` |
| Leakage (Figure 2, Table 3) | `sniffer_skew_results.txt`, `sniffer_fill_only_results.txt` |
| Defenses and averaging (Figure 3) | `phase7_blind_results.txt`, `phase7_nobs_results.txt`, `phase8_toxicity_results.txt`, `phase8_gamma_results.txt`, `phase8_reference_results.txt`, `quant_repro_results.txt` |
| Width sweep (Figure 4, Table 4) | `phase7_width_results.txt`, `sd_ytest_results.txt` |
| Damage (Table 5) | `damage_ab_results.txt`, `damage_qs12_results.txt` |
| Extraction | `extraction_diag_results.txt`, `onehour_confirm_results.txt` |

## Reproducing the figures

```
python make_figs.py
```

`fig_data.py` reads every plotted value from the results files in this folder. Each value is checked against the exact source line, so the script fails if a results file has changed. `make_figs.py` then draws the four figures into `figs/`. They require matplotlib. The typeface is TeX Gyre Pagella, and matplotlib falls back to its default font if Pagella is not installed.

## Notes

`NOTES.md`, cited in some scripts and results files, is the author's notes file and is not included.

AI use: The code was written with AI coding tools (Claude Code) from the author's specifications; the research questions, experimental design and decision rules are the author's.
