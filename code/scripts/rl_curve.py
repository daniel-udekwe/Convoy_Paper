"""SAC training curves (validation every 20k steps), with the robust gap NMPC
on the same 8 validation paths as a reference line.

Output: results/figures/rl_training_curve.png, results/rl/nmpc_val.json
"""
import json
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
from convoy.convoy import simulate                          # noqa: E402
from convoy.params import ConvoyConfig                      # noqa: E402
from convoy.paths import gnss_like_path, random_waypoints   # noqa: E402

RL = ROOT / "results" / "rl"
NMPC = dict(gap_nmpc=True, follower_preview=True, leader_preview=True,
            leader_plan=True, plan_compensation=True, plan_path_ext=True,
            pose_history=30)


def nmpc_val(seed):
    """Same path construction as scripts/rl_record.py for the validation seeds."""
    rng = np.random.default_rng(seed)
    path = gnss_like_path(seed=seed, wp=random_waypoints(seed), vcap=rng.uniform(4.5, 6.0))
    r = simulate(path, K=ConvoyConfig(**NMPC))
    return float(np.sqrt(np.mean(r["ed"] ** 2))), float(np.sqrt(np.mean(r["eyF"] ** 2)))


def load_curve(seed):
    rows = (RL / f"sac_seed{seed}_curve.csv").read_text().strip().splitlines()[1:]
    a = np.array([[float(x) for x in r.split(",")] for r in rows])
    return dict(steps=a[:, 0], ret=a[:, 2], gap=a[:, 3], lat=a[:, 4], fails=a[:, 5])


if __name__ == "__main__":
    cache = RL / "nmpc_val.json"
    if cache.exists():
        ref = json.loads(cache.read_text())
    else:
        with ProcessPoolExecutor(2) as ex:
            vals = list(ex.map(nmpc_val, range(1000, 1008)))
        ref = dict(gap=float(np.mean([v[0] for v in vals])),
                   lat=float(np.mean([v[1] for v in vals])), per_path=vals)
        cache.write_text(json.dumps(ref, indent=1))
    print(f"NMPC on validation paths: gap {ref['gap']:.4f} m, lateral {ref['lat']:.4f} m")

    ps.apply()
    sty = {0: ("SAC seed 0", ps.MAGENTA, "^"), 1: ("SAC seed 1", ps.GREEN, "v")}
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    for seed, (lbl, col, mk) in sty.items():
        c = load_curve(seed)
        k = c["steps"] / 1000
        axs[0].plot(k, c["ret"], color=col, marker=mk, ms=4, lw=1.6, label=lbl)
        axs[1].plot(k, c["gap"], color=col, marker=mk, ms=4, lw=1.6, label=lbl)
        axs[2].plot(k, c["lat"], color=col, marker=mk, ms=4, lw=1.6, label=lbl)
        # the checkpoint kept for evaluation (lowest validation gap + lateral)
        b = int(np.argmin(c["gap"] + c["lat"] + 10 * c["fails"]))
        for ax, key in zip(axs[1:], ("gap", "lat")):
            ax.scatter(k[b], c[key][b], s=110, facecolor="none", edgecolor=col,
                       linewidth=1.6, zorder=4)
    for ax, key in zip(axs[1:], ("gap", "lat")):
        ax.axhline(ref[key], color=ps.BLUE, lw=1.8, ls="--", label="Gap NMPC (no training)")
    axs[0].set_title("Validation return", loc="left")
    axs[1].set_title("Validation gap RMSE (m)", loc="left")
    axs[2].set_title("Validation lateral RMSE (m)", loc="left")
    axs[1].set_ylim(0, 0.15)
    axs[2].set_ylim(0, 0.08)
    axs[0].set_ylim(280, 302)
    for ax in axs:
        ax.set_xlabel("Training steps (thousands)")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("SAC training on 60 random paths, validated on 8 held-out paths "
                 "(circles: checkpoint kept)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out = ROOT / "results" / "figures" / "rl_training_curve.png"
    fig.savefig(out)
    print("saved", out)
