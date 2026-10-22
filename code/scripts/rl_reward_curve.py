"""SAC learning curve: reward collected in every training episode.

Reads the per-episode Monitor logs written by rl_train.py (return, length)
and plots, against training steps, the episode return (raw dots + rolling
mean) and the reward per step (return / length; episodes run on paths of
different lengths, so this is the like-for-like measure). The robust gap
NMPC is scored with the same reward function on the same 60 training paths
as a reference.
Usage: python scripts/rl_reward_curve.py [run_dir]   (default results/rl/rerun)
Output: results/figures/rl_reward_curve.png, <run_dir>/nmpc_train_reward.json
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
from convoy import plotstyle as ps                          # noqa: E402
from convoy.convoy import simulate                          # noqa: E402
from convoy.params import ConvoyConfig, MPCConfig           # noqa: E402
from convoy.paths import gnss_like_path, random_waypoints   # noqa: E402

NMPC = dict(gap_nmpc=True, follower_preview=True, leader_preview=True,
            leader_plan=True, plan_compensation=True, plan_path_ext=True,
            pose_history=30)


def step_rewards(ey, epsi, ed, a, d):
    """Exactly the training reward of convoy/rl.py, applied to a logged run."""
    du_a = np.diff(np.concatenate([[0.0], a])) / 0.5
    du_d = np.diff(np.concatenate([[0.0], d])) / 0.05
    cost = (0.5 * (ey / 0.1) ** 2 + 0.2 * (epsi / 0.05) ** 2 + 0.5 * (ed / 0.2) ** 2
            + 0.05 * (du_a ** 2 + du_d ** 2))
    return 1.0 - 0.1 * np.minimum(cost, 10.0)


def nmpc_train(seed):
    """Same path construction as scripts/rl_record.py for the training seeds."""
    rng = np.random.default_rng(seed)
    path = gnss_like_path(seed=seed, wp=random_waypoints(seed), vcap=rng.uniform(4.5, 6.0))
    r = simulate(path, K=ConvoyConfig(**NMPC))
    rw = step_rewards(r["eyF"], r["epsiF"], r["ed"], r["aF"], r["dF"])
    return float(rw.sum()), int(rw.size)


def load_monitor(f):
    lines = f.read_text().strip().splitlines()[2:]      # JSON header + column names
    a = np.array([[float(x) for x in ln.split(",")] for ln in lines])
    ret, length = a[:, 0], a[:, 1]
    return dict(ret=ret, length=length, steps=np.cumsum(length))


def rolling(x, w):
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w - 1, x[0]), x])
    return np.convolve(pad, k, "valid")


if __name__ == "__main__":
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "rl" / "rerun"
    cache = run / "nmpc_train_reward.json"
    if cache.exists():
        ref = json.loads(cache.read_text())
    else:
        with ProcessPoolExecutor(2) as ex:
            v = list(ex.map(nmpc_train, range(60)))
        ref = dict(ret=float(np.mean([x[0] for x in v])),
                   per_step=float(np.sum([x[0] for x in v]) / np.sum([x[1] for x in v])),
                   per_path=v)
        cache.write_text(json.dumps(ref, indent=1))
    print(f"NMPC on the training paths: mean episode return {ref['ret']:.1f}, "
          f"reward per step {ref['per_step']:.4f}")

    ps.apply()
    sty = {0: ("SAC seed 0", ps.MAGENTA), 1: ("SAC seed 1", ps.GREEN)}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
    for seed, (lbl, col) in sty.items():
        m = load_monitor(run / f"sac_seed{seed}.monitor.csv")
        k = m["steps"] / 1000
        per = m["ret"] / m["length"]
        axs[0].scatter(k, m["ret"], s=4, color=col, alpha=0.18, lw=0)
        axs[0].plot(k, rolling(m["ret"], 25), color=col, lw=2, label=f"{lbl} (25-episode mean)")
        axs[1].scatter(k, per, s=4, color=col, alpha=0.18, lw=0)
        axs[1].plot(k, rolling(per, 25), color=col, lw=2, label=f"{lbl} (25-episode mean)")
        print(f"{lbl}: {len(per)} episodes; last-100 mean return {m['ret'][-100:].mean():.1f}, "
              f"reward/step {per[-100:].mean():.4f}")
    axs[0].axhline(ref["ret"], color=ps.BLUE, ls="--", lw=1.8, label="Gap NMPC, same paths")
    axs[1].axhline(ref["per_step"], color=ps.BLUE, ls="--", lw=1.8, label="Gap NMPC, same paths")
    axs[0].set_title("Reward collected per training episode")
    axs[0].set_ylabel("Episode return")
    axs[1].set_title("Reward per step (episodes differ in length)")
    axs[1].set_ylabel("Mean reward per step (max 1)")
    axs[1].set_ylim(0.5, 1.02)
    for ax in axs:
        ax.set_xlabel("Training steps (thousands)")
        ax.set_xlim(0, 300)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = ROOT / "results" / "figures" / "rl_reward_curve.png"
    fig.savefig(out)
    print("saved", out)
