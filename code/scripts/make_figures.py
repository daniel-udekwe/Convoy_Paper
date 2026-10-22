"""Figures from results/*.npz (run run_experiments.py first)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convoy import plotstyle as ps  # noqa: E402
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# reference palette (dataviz skill, light mode); baseline in neutral grey
C = {"baseline": ps.GREY, "ff": ps.MAGENTA, "gap_nmpc": ps.BLUE,
     "leader": ps.BLUE, "follower": ps.MAGENTA, "ref": ps.LIGHT}
LBL = {"baseline": "Baseline (PI spacing, no preview)",
       "ff": "PI + speed feedforward",
       "ff_preview": "PI + feedforward + preview",
       "ff_preview_plan": "PI + feedforward + preview w/ leader plan",
       "gap_nmpc_nopreview": "Gap-aware NMPC, no preview",
       "gap_nmpc": "Proposed: gap-aware NMPC + preview"}
NICE = {"scenario1_ideal": "Scenario 1 (ideal path)",
        "scenario2_gnss_like": "Scenario 2 (GNSS-like path)"}
ORDER = ["baseline", "ff", "ff_preview", "ff_preview_plan",
         "gap_nmpc_nopreview", "gap_nmpc"]
ps.apply()


def load(scen, var):
    return dict(np.load(RES / f"{scen}__{var}.npz"))


def trajectory(scen):
    r = load(scen, "baseline")
    fig, ax = plt.subplots(figsize=(6, 5))
    p = r["path"]
    ax.plot(p[:, 1], p[:, 2], color=C["ref"], lw=4, label="Reference", zorder=1)
    ax.plot(r["xL"], r["yL"], color=C["leader"], label="Leader")
    ax.plot(r["xF"], r["yF"], color=C["follower"], ls="--", label="Follower")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{NICE[scen]}: baseline trajectories", loc="left")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(FIG / f"{scen}_trajectory.png"); plt.close(fig)


def follower_errors(scen):
    fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    keys = [("eyF", "Lateral (m)"), ("epsiF", "Yaw (rad)"),
            ("evF", "Speed vs leader (m/s)"), ("ed", "Gap error (m)")]
    for var in ("baseline", "ff", "gap_nmpc"):
        r = load(scen, var)
        for ax, (k, name) in zip(axs, keys):
            ax.plot(r["t"], r[k], color=C[var], label=LBL[var])
            ax.set_ylabel(name)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("t (s)")
    axs[0].set_title(f"{NICE[scen]}: follower errors", loc="left")
    fig.tight_layout(); fig.savefig(FIG / f"{scen}_follower_errors.png"); plt.close(fig)


def gap_summary():
    """Gap RMSE per variant, one panel per scenario, proposed highlighted."""
    scens = [("scenario1_ideal", "Scenario 1 (ideal)"),
             ("scenario2_gnss_like", "Scenario 2 (GNSS-like)")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, (s, name) in zip(axs, scens):
        vals = [np.sqrt(np.mean(load(s, v)["ed"] ** 2)) for v in ORDER]
        cols = [C["gap_nmpc"] if v == "gap_nmpc" else ps.LIGHT for v in ORDER]
        ax.barh(y, vals, 0.62, color=cols)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.015, yi, f"{v:.3f}", va="center", fontsize=8)
        ax.set_title(f"{name}: gap RMSE (m)", loc="left")
        ax.set_xlim(0, max(vals) * 1.18); ax.grid(axis="y")
    axs[0].set_yticks(y, [LBL[v] for v in ORDER])
    fig.tight_layout(); fig.savefig(FIG / "gap_error_summary.png"); plt.close(fig)


if __name__ == "__main__":
    for s in ["scenario1_ideal", "scenario2_gnss_like"]:
        trajectory(s); follower_errors(s)
    gap_summary()
    print("figures in", FIG)
