# diag_frontier.py: trade-rate frontier including marketable limit orders

import statistics
from diag_coherence import run

print("frontier search; can any setting hit both the Phase 3 trade rate and "
      "a live anchor?")
print("Phase 3 target: 0.0451 trades/s")
print("%-6s %-7s %-6s %-8s %8s %8s %9s %10s %9s %8s %9s %7s"
      % ("mult", "p_mkt", "life", "disp", "trade/s", "lim mkt%", "uncond|e|",
         "cond|e|", "resid2sd", "age med", "full2sd", "rest"))
print("-" * 116)
CFG = ((3, 0.20, 180.0, 0.0020), (5, 0.08, 180.0, 0.0020),
       (8, 0.02, 180.0, 0.0020), (5, 0.02, 180.0, 0.0020),
       (3, 0.12, 90.0, 0.0020), (2, 0.25, 180.0, 0.0009))
for mult, pm, life, disp in CFG:
    x = run(mult, pm, life, disp)
    flag = "  <== ON TARGET" if abs(x["trate"] - 0.0451) < 0.010 else ""
    print("%-6d %-7.3f %-6.0f %-8.5f %8.4f %7.1f%% %9.2f %10.2f %8.1f%% %7.0fs %8.2f%% %7d%s"
          % (mult, pm, life, disp, x["trate"], x["lim_mkt"], x["uncond"],
             x["cond"], x["r2"], x["age"], x["tw"], x["rest"], flag))
