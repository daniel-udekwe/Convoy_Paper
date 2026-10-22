"""RL (SAC) vs gap NMPC across evaluation conditions, from results/rl_eval.json."""
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
res = json.loads((ROOT / "results" / "rl_eval.json").read_text())
COND = ["s1_nominal", "s2_nominal", "s2_delay300_loss20", "s2_blackout",
        "s2_realistic_ekf", "s2_rigid_bar", "s2_rigid_realistic"]
NAMES = ["Ideal path", "GNSS-like path", "+ 300 ms delay, 20% loss",
         "+ 2 s blackout", "+ EKF, delay, loss", "+ rigid bar",
         "+ rigid bar, EKF, delay, loss"]
STY = {"nmpc": ("Gap NMPC", ps.BLUE, "o"),
       "sac_seed0": ("SAC seed 0", ps.MAGENTA, "^"),
       "sac_seed1": ("SAC seed 1", ps.GREEN, "v")}
METRICS = [("gap_rmse", "Gap RMSE (m)"), ("lat_rmse", "Lateral RMSE (m)"),
           ("jerk_rms", "Jerk RMS (m/s³)")]
ps.apply()

y = np.arange(len(COND))[::-1]
fig, axs = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
for ax, (key, title) in zip(axs, METRICS):
    for ctrl, (lbl, col, mk) in STY.items():
        v = [next(r[key] for r in res if r["ctrl"] == ctrl and r["cond"] == c) for c in COND]
        ax.scatter(v, y, color=col, marker=mk, s=55 if ctrl == "nmpc" else 45,
                   label=lbl, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_title(title, loc="left")
    ax.set_xlim(left=0)
    ax.grid(axis="y")
axs[0].set_yticks(y, NAMES)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9)
fig.suptitle("Follower controller: model-free RL vs gap NMPC (test paths unseen in training)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 0.96))
out = ROOT / "results" / "figures" / "rl_vs_nmpc.png"
fig.savefig(out)
print("saved", out)
