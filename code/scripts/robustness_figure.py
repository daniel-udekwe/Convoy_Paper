"""Gap RMSE vs V2V delay, from results/robustness.json."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convoy import plotstyle as ps  # noqa: E402
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
res = json.loads((ROOT / "results" / "robustness.json").read_text())

# gap_comp and gap_robust coincide without blackouts, so the delay plot shows
# gap_robust only; blackout results are in robustness_blackout.png
C = {"baseline": ps.GREY, "ff": ps.MAGENTA, "gap_naive": ps.GREEN,
     "gap_robust": ps.BLUE}
LBL = {"baseline": "Baseline (PI spacing)", "ff": "PI + feedforward",
       "gap_naive": "Gap NMPC, stale plan",
       "gap_robust": "Gap NMPC, age-compensated (proposed)"}
MK = {"baseline": "s", "ff": "^", "gap_naive": "D", "gap_robust": "o"}
ps.apply()

delays = [0.0, 0.1, 0.2, 0.3]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, cond, title in zip(axs, ["delay", "delay+loss20"],
                           ["No packet loss", "20% packet loss (3 seeds, range shaded)"]):
    for ctrl in C:
        m, lo, hi = [], [], []
        for d in delays:
            v = [r["gap_rmse"] for r in res
                 if r["ctrl"] == ctrl and r["cond"] == cond and r["delay"] == d]
            m.append(np.mean(v)); lo.append(min(v)); hi.append(max(v))
        x = np.array(delays) * 1000
        ax.plot(x, m, color=C[ctrl], marker=MK[ctrl], ms=7, lw=2, label=LBL[ctrl])
        if cond != "delay":
            ax.fill_between(x, lo, hi, color=C[ctrl], alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("V2V delay (ms)")
    ax.set_title(title, loc="left")
axs[0].set_ylabel("Follower gap RMSE (m, log scale)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=9)
fig.suptitle("GNSS-like path: gap error vs communication delay", x=0.01,
             ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 0.97))
fig.savefig(FIG / "robustness_delay.png")
print("saved", FIG / "robustness_delay.png")
