"""Mechanical coupling study (contribution C3).

A spring-damper link (damping ratio 0.7) joins a hitch 1.2 m behind the
leader's CoM to one 1.2 m ahead of the follower's. Stiffness sweeps from
no link (k = 0: the link length is still measured, showing what a rigid bar
would have to absorb) through soft to near-rigid.

Controllers regulate either the CoM gap (as in the paper and so far) or the
hitch-to-hitch length the link actually sees (hitch_aware).
GNSS-like path, ideal V2V, exact states, to isolate the coupling.
Output: results/coupling.json, printed table.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convoy.convoy import simulate          # noqa: E402
from convoy.params import ConvoyConfig      # noqa: E402
from convoy.paths import gnss_like_path     # noqa: E402

ROBUST = dict(gap_nmpc=True, follower_preview=True, leader_preview=True,
              leader_plan=True, plan_compensation=True, plan_path_ext=True,
              pose_history=30)
CTRL = {"ff": dict(feedforward=True),
        "ff_hitch": dict(feedforward=True, hitch_aware=True),
        "gap": dict(ROBUST),
        "gap_hitch": dict(ROBUST, hitch_aware=True)}
STIFF = [0.0, 2e3, 2e4, 2e5]


def job(a):
    ctrl, k = a
    r = simulate(gnss_like_path(), K=ConvoyConfig(**CTRL[ctrl], link_k=k))
    rms = lambda x: float(np.sqrt(np.mean(r[x] ** 2)))   # noqa: E731
    pk = lambda x: float(np.max(np.abs(r[x])))           # noqa: E731
    return dict(ctrl=ctrl, k=k,
                link_rmse=rms("e_link"), link_peak=pk("e_link"),
                force_rms=rms("f_link"), force_peak=pk("f_link"),
                gap_rmse=rms("ed"), lat_rmse=rms("eyF"), lat_peak=pk("eyF"),
                lead_lat_peak=pk("eyL"),
                accel_sat=float(np.mean((r["aF"] <= -1.99) | (r["aF"] >= 0.99))))


if __name__ == "__main__":
    jobs = [(c, k) for k in STIFF for c in CTRL]
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(job, jobs))
    (ROOT / "results" / "coupling.json").write_text(json.dumps(res, indent=1))
    print(f"{'ctrl':10}{'k N/m':>9}{'link err rms':>14}{'peak':>8}{'F rms N':>10}"
          f"{'F peak N':>10}{'lat rms':>9}{'lat pk':>8}{'lead pk':>9}{'a sat':>7}")
    for r in res:
        print(f"{r['ctrl']:10}{r['k']:9.0f}{r['link_rmse']:14.4f}{r['link_peak']:8.3f}"
              f"{r['force_rms']:10.0f}{r['force_peak']:10.0f}{r['lat_rmse']:9.4f}"
              f"{r['lat_peak']:8.3f}{r['lead_lat_peak']:9.3f}{r['accel_sat']:7.0%}")
