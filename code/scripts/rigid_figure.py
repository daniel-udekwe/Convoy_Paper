"""Bar tension and leader path error over time with a rigid drawbar."""
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

ROBUST = dict(gap_nmpc=True, follower_preview=True, leader_preview=True,
              leader_plan=True, plan_compensation=True, plan_path_ext=True,
              pose_history=30)
RUNS = {"ff": ("PI + feedforward (CoM gap)", ps.MAGENTA, dict(feedforward=True)),
        "gap_hitch": ("Gap NMPC, hitch length (proposed)", ps.BLUE,
                      dict(ROBUST, hitch_aware=True))}


def run(key):
    return key, simulate(gnss_like_path(),
                         K=ConvoyConfig(**RUNS[key][2], link_rigid=True))


if __name__ == "__main__":
    with ProcessPoolExecutor(2) as ex:
        res = dict(ex.map(run, RUNS))
    ps.apply()
    fig, axs = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    for key, (lbl, col, _) in RUNS.items():
        r = res[key]
        axs[0].plot(r["t"], r["f_link"], color=col, lw=1.6, label=lbl)
        axs[1].plot(r["t"], r["eyL"], color=col, lw=1.6)
    axs[0].axhline(0, color=ps.GREY, lw=0.8)
    axs[0].set_ylabel("Bar tension (N)\n+ pull, − push")
    axs[1].set_ylabel("Leader lateral error (m)")
    axs[1].set_xlabel("t (s)")
    axs[0].set_title("Rigid drawbar on the GNSS-like path", loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = ROOT / "results" / "figures" / "rigid_bar.png"
    fig.savefig(out)
    print("saved", out)
