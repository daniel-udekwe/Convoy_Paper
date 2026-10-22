"""Time series through a 2 s V2V blackout (t = 6-8 s) on the GNSS-like path."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convoy import plotstyle as ps  # noqa: E402
from convoy.convoy import simulate          # noqa: E402
from convoy.params import ConvoyConfig      # noqa: E402
from convoy.paths import gnss_like_path     # noqa: E402

GAP = dict(gap_nmpc=True, follower_preview=True, leader_preview=True,
           leader_plan=True, plan_compensation=True)
RUNS = {"ff": ("PI + feedforward", ps.MAGENTA, dict(feedforward=True, pose_history=30)),
        "gap_comp": ("Gap NMPC, age-compensated", ps.GREEN, dict(GAP)),
        "gap_robust": ("+ planned-path extension + pose history (proposed)", ps.BLUE,
                       dict(GAP, plan_path_ext=True, pose_history=30))}
BLACK = (6.0, 8.0)


def run(key):
    K = ConvoyConfig(**RUNS[key][2], v2v_delay=0.1, v2v_blackout=BLACK)
    return key, simulate(gnss_like_path(), K=K)


if __name__ == "__main__":
    with ProcessPoolExecutor(2) as ex:
        res = dict(ex.map(run, RUNS))
    ps.apply()
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for key, (lbl, col, _) in RUNS.items():
        r = res[key]
        axs[0].plot(r["t"], r["eyF"], color=col, lw=1.8, label=lbl)
        axs[1].plot(r["t"], r["ed"], color=col, lw=1.8)
    for ax in axs:
        ax.axvspan(*BLACK, color=ps.LIGHT, alpha=0.25, lw=0)
        ax.set_xlim(0, 20)
    axs[0].set_ylabel("Follower lateral (m)")
    axs[1].set_ylabel("Gap error (m)")
    axs[1].set_xlabel("t (s)")
    axs[0].set_title("2 s V2V blackout (shaded) during the first braking event, "
                     "100 ms delay", loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = ROOT / "results" / "figures" / "robustness_blackout.png"
    fig.savefig(out)
    print("saved", out)
