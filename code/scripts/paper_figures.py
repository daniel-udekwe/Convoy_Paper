"""Trajectory-tracking figures for the manuscript (from results/*.npz).

Outputs in results/figures/:
  scenarios.png          path geometry, curvature and reference speed, both scenarios
  trajectories.png       leader/follower trajectories with zoomed insets
  tracking_errors.png    follower lateral, heading, speed and gap errors vs time
  inputs_speed.png       speeds, commanded acceleration and steering vs time
Run scripts/run_experiments.py first.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convoy import plotstyle as ps                    # noqa: E402
from convoy.params import MPCConfig                   # noqa: E402
from convoy.paths import gnss_like_path, ideal_path   # noqa: E402

RES, FIG = ROOT / "results", ROOT / "results" / "figures"
SCEN = [("scenario1_ideal", "Scenario 1 (ideal path)", ideal_path),
        ("scenario2_gnss_like", "Scenario 2 (GNSS-like path)", gnss_like_path)]
VAR = [("baseline", "Baseline (PI spacing)", ps.GREY, "-"),
       ("ff", "PI + speed feedforward", ps.MAGENTA, "-"),
       ("gap_nmpc", "Gap-aware NMPC (proposed)", ps.BLUE, "-")]
C = MPCConfig()


def load(s, v):
    return dict(np.load(RES / f"{s}__{v}.npz"))


def scenarios():
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2),
                            gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    for row, (key, name, fn) in enumerate(SCEN):
        p = fn().arr
        s, x, y, rho, v = p[:, 0], p[:, 1], p[:, 2], p[:, 4], p[:, 5]
        ax = axs[row, 0]
        ax.plot(x, y, color=ps.INK, lw=1.6)
        ax.plot(x[0], y[0], "o", color=ps.BLUE, ms=6, label="start")
        ax.plot(x[-1], y[-1], "s", color=ps.MAGENTA, ms=6, label="end")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("$X$ (m)"); ax.set_ylabel("$Y$ (m)")
        ax.set_title(f"{name}: geometry")
        if row == 0:
            ax.legend(loc="upper right")
        ax = axs[row, 1]
        ax.plot(s, rho, color=ps.INK, lw=1.2)
        ax.axhline(0, color=ps.GREY, lw=0.8)
        ax.set_xlabel("Arc length $s$ (m)"); ax.set_ylabel(r"Curvature $\rho$ (1/m)")
        ax.set_title("Signed curvature")
        ax = axs[row, 2]
        ax.plot(s, v, color=ps.INK, lw=1.4)
        ax.set_xlabel("Arc length $s$ (m)"); ax.set_ylabel("Reference speed (m/s)")
        ax.set_ylim(0, 7.2)
        ax.set_title("Leader reference speed")
    fig.tight_layout()
    fig.savefig(FIG / "scenarios.png"); plt.close(fig)


def _draw(ax, key, thick):
    base = load(key, "baseline")
    p = base["path"]
    ax.plot(p[:, 1], p[:, 2], color=ps.LIGHT, lw=thick, solid_capstyle="round",
            label="Reference path", zorder=1)
    ax.plot(base["xL"], base["yL"], color=ps.INK, lw=1.2, label="Leader", zorder=3)
    for v, lbl, col, ls in VAR:
        r = load(key, v)
        ax.plot(r["xF"], r["yF"], color=col, lw=1.5, ls=ls, label=f"Follower: {lbl}",
                zorder=4 if v == "gap_nmpc" else 2)


def trajectories():
    """Top: full trajectories. Bottom: zooms on the start-up transient (largest
    baseline deviation in the first 10 s) and on the corner with the largest
    baseline deviation afterwards."""
    fig = plt.figure(figsize=(11, 9.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1])
    tops = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4])]
    zooms = [fig.add_subplot(gs[1, k]) for k in range(4)]
    for c, (key, name, _) in enumerate(SCEN):
        base = load(key, "baseline")
        ax = tops[c]
        _draw(ax, key, 7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("$X$ (m)"); ax.set_ylabel("$Y$ (m)")
        ax.set_title(name)
        early = base["t"] <= 10.0
        picks = [("start-up", int(np.argmax(np.abs(base["eyF"]) * early)), 1.0),
                 ("corner", int(np.argmax(np.abs(base["eyF"]) * ~early)), 0.45)]
        for k, (tag, i, w) in enumerate(picks):
            cx, cy = base["xF"][i], base["yF"][i]
            ax.add_patch(plt.Rectangle((cx - w, cy - w), 2 * w, 2 * w, fill=False,
                                       ec=ps.MUTED, lw=1.0, zorder=6))
            ax.annotate(f"({chr(97 + 2 * c + k)})", (cx + w, cy + w), xytext=(3, 2),
                        textcoords="offset points", fontsize=9, color=ps.MUTED)
            z = zooms[2 * c + k]
            _draw(z, key, 16)
            z.set_xlim(cx - w, cx + w); z.set_ylim(cy - w, cy + w)
            z.set_aspect("equal")
            z.set_xlabel("$X$ (m)")
            if k == 0 and c == 0:
                z.set_ylabel("$Y$ (m)")
            z.set_title(f"({chr(97 + 2 * c + k)}) S{c + 1} {tag}, t = {base['t'][i]:.1f} s",
                        fontsize=9.5)
            z.tick_params(labelsize=8)
    h, l = tops[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "trajectories.png"); plt.close(fig)


def tracking_errors():
    rows = [("eyF", "Lateral error $e_y$ (m)"), ("epsiF", r"Heading error $e_\psi$ (rad)"),
            ("evF", "Speed error $v_L - v_F$ (m/s)"), ("ed", "Gap error $e_d$ (m)")]
    fig, axs = plt.subplots(4, 2, figsize=(11, 9), sharex="col")
    for c, (key, name, _) in enumerate(SCEN):
        for v, lbl, col, ls in VAR:
            r = load(key, v)
            for rr, (k, yl) in enumerate(rows):
                axs[rr, c].plot(r["t"], r[k], color=col, lw=1.3 if v != "gap_nmpc" else 1.7,
                                ls=ls, label=lbl, zorder=3 if v == "gap_nmpc" else 2)
                if c == 0:
                    axs[rr, c].set_ylabel(yl)
        axs[0, c].set_title(name)
        axs[-1, c].set_xlabel("Time $t$ (s)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "tracking_errors.png"); plt.close(fig)


def inputs_speed():
    fig, axs = plt.subplots(3, 2, figsize=(11, 7.6), sharex="col")
    for c, (key, name, _) in enumerate(SCEN):
        base = load(key, "baseline")
        prop = load(key, "gap_nmpc")
        axs[0, c].plot(prop["t"], prop["vL"] + prop["evL"], color=ps.LIGHT, lw=4,
                       label="Leader reference speed")
        axs[0, c].plot(prop["t"], prop["vL"], color=ps.INK, lw=1.2, label="Leader speed")
        for v, lbl, col, ls in VAR:
            r = load(key, v)
            axs[0, c].plot(r["t"], r["vF"], color=col, lw=1.3, ls=ls,
                           label=f"Follower: {lbl}")
            axs[1, c].plot(r["t"], r["aF"], color=col, lw=1.2, ls=ls)
            axs[2, c].plot(r["t"], r["dF"], color=col, lw=1.2, ls=ls)
        for ax, lim in ((axs[1, c], (C.a_min, C.a_max)), (axs[2, c], (-C.d_max, C.d_max))):
            for b in lim:
                ax.axhline(b, color=ps.MUTED, lw=0.9, ls="--")
        axs[0, c].set_title(name)
        axs[-1, c].set_xlabel("Time $t$ (s)")
    axs[0, 0].set_ylabel("Speed (m/s)")
    axs[1, 0].set_ylabel("Follower $a_c$ (m/s$^2$)")
    axs[2, 0].set_ylabel(r"Follower $\delta$ (rad)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "inputs_speed.png"); plt.close(fig)


# inset placement [x0, y0, w, h] in axes fraction, chosen to sit in empty space
INSET = {"scenario1_ideal": [0.56, 0.05, 0.42, 0.42],
         "scenario2_gnss_like": [0.03, 0.52, 0.42, 0.42]}

if __name__ == "__main__":
    ps.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    scenarios(); trajectories(); tracking_errors(); inputs_speed()
    print("saved scenarios, trajectories, tracking_errors, inputs_speed")
