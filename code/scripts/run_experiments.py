"""Run both scenarios for the baseline and the first improvements.

Variants: see VARIANTS below (baseline -> proposed gap_nmpc, with ablations).

Outputs go to results/: metrics.md, metrics.json, *.npz, figures/*.png
Usage: python scripts/run_experiments.py [--kp 2.0 --ki 1.0]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convoy.convoy import simulate                      # noqa: E402
from convoy.metrics import fmt, table                   # noqa: E402
from convoy.params import ConvoyConfig                  # noqa: E402
from convoy.paths import gnss_like_path, ideal_path     # noqa: E402

SCENARIOS = {"scenario1_ideal": ideal_path, "scenario2_gnss_like": gnss_like_path}
VARIANTS = {
    # Franci et al. (2026) as described
    "baseline": dict(),
    # C1 quick win: v_ref = v_leader + PI
    "ff": dict(feedforward=True),
    # + curvature preview, clamped at the buffer's end
    "ff_preview": dict(feedforward=True, follower_preview=True, leader_preview=True),
    # + preview continued with the leader's shared plan (C2 full)
    "ff_preview_plan": dict(feedforward=True, follower_preview=True,
                            leader_preview=True, leader_plan=True),
    # spacing inside the NMPC, no curvature preview (C1 full, ablation)
    "gap_nmpc_nopreview": dict(gap_nmpc=True, leader_preview=True),
    # proposed: C1 full + C2 full
    "gap_nmpc": dict(gap_nmpc=True, follower_preview=True, leader_preview=True,
                     leader_plan=True),
}


def job(args):
    scen, var, kp, ki = args
    K = ConvoyConfig(Kp=kp, Ki=ki, **VARIANTS[var])
    r = simulate(SCENARIOS[scen](), K=K)
    np.savez_compressed(ROOT / "results" / f"{scen}__{var}.npz", **r)
    return scen, var, table(r), float(r["solve_ms"].mean()), float(r["solve_ms"].max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=float, default=ConvoyConfig.Kp)
    ap.add_argument("--ki", type=float, default=ConvoyConfig.Ki)
    a = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    jobs = [(s, v, a.kp, a.ki) for s in SCENARIOS for v in VARIANTS]
    with ProcessPoolExecutor(2) as ex:
        out = list(ex.map(job, jobs))
    md = [f"# Results (Kp={a.kp}, Ki={a.ki})\n"]
    js = {}
    for scen, var, rows, ms_mean, ms_max in out:
        md.append("```\n" + fmt(rows, f"{scen} / {var}   NMPC solve: "
                                      f"{ms_mean:.1f} ms mean, {ms_max:.1f} ms max")
                  + "\n```\n")
        js[f"{scen}/{var}"] = dict(rows=rows, solve_ms_mean=ms_mean, solve_ms_max=ms_max)
        print(md[-1])
    (ROOT / "results" / "metrics.md").write_text("\n".join(md))
    (ROOT / "results" / "metrics.json").write_text(json.dumps(js, indent=1))
