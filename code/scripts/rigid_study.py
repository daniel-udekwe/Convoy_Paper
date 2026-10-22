"""Rigid drawbar study (contribution C3, final check).

The hitch-to-hitch distance is held exactly at its rest length by a
constraint force (plant.step_pair_rigid). Four controllers on both
scenarios with exact states and an ideal link, then the proposed
controller with the full realistic stack (EKF on RTK GNSS, 100 ms V2V
delay, 10% loss).
Output: results/rigid.json, printed table.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convoy.convoy import simulate                    # noqa: E402
from convoy.params import ConvoyConfig                # noqa: E402
from convoy.paths import gnss_like_path, ideal_path   # noqa: E402

ROBUST = dict(gap_nmpc=True, follower_preview=True, leader_preview=True,
              leader_plan=True, plan_compensation=True, plan_path_ext=True,
              pose_history=30)
CTRL = {"ff": dict(feedforward=True),
        "ff_hitch": dict(feedforward=True, hitch_aware=True),
        "gap": dict(ROBUST),
        "gap_hitch": dict(ROBUST, hitch_aware=True)}
REAL = dict(sensing="ekf", gnss_sigma=0.02, v2v_delay=0.1, v2v_loss=0.1)
SCEN = {"ideal": ideal_path, "gnss_like": gnss_like_path}


def job(a):
    scen, ctrl, real, seed = a
    kw = dict(CTRL[ctrl], link_rigid=True)
    if real:
        kw.update(REAL, v2v_seed=seed, noise_seed=300 + seed)
    r = simulate(SCEN[scen](), K=ConvoyConfig(**kw))
    rms = lambda x: float(np.sqrt(np.mean(r[x] ** 2)))   # noqa: E731
    pk = lambda x: float(np.max(np.abs(r[x])))           # noqa: E731
    return dict(scen=scen, ctrl=ctrl, realistic=real, seed=seed,
                force_rms=rms("f_link"), force_peak=pk("f_link"),
                bar_err_peak=pk("e_link"), lat_rmse=rms("eyF"), lat_peak=pk("eyF"),
                lead_lat_rmse=rms("eyL"), lead_lat_peak=pk("eyL"),
                lead_speed_rmse=rms("evL"),
                accel_sat=float(np.mean((r["aF"] <= -1.99) | (r["aF"] >= 0.99))),
                steer_sat=float(np.mean(np.abs(r["dF"]) > 0.39)))


if __name__ == "__main__":
    jobs = [(s, c, False, 0) for s in SCEN for c in CTRL]
    jobs += [("gnss_like", c, True, k) for c in ("ff", "gap_hitch") for k in range(3)]
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(job, jobs))
    (ROOT / "results" / "rigid.json").write_text(json.dumps(res, indent=1))
    print(f"{'scen':10}{'ctrl':10}{'real':>5}{'seed':>5}{'F rms':>8}{'F pk':>8}"
          f"{'bar pk':>9}{'lat rms':>9}{'lat pk':>8}{'lead rms':>9}{'lead pk':>8}"
          f"{'lead v':>8}{'a sat':>7}{'d sat':>7}")
    for r in res:
        print(f"{r['scen']:10}{r['ctrl']:10}{str(r['realistic'])[0]:>5}{r['seed']:>5}"
              f"{r['force_rms']:8.0f}{r['force_peak']:8.0f}{r['bar_err_peak']:9.1e}"
              f"{r['lat_rmse']:9.4f}{r['lat_peak']:8.3f}{r['lead_lat_rmse']:9.4f}"
              f"{r['lead_lat_peak']:8.3f}{r['lead_speed_rmse']:8.3f}"
              f"{r['accel_sat']:7.0%}{r['steer_sat']:7.0%}")
