"""Evaluate trained SAC followers against the robust gap NMPC.

Both get the same information (shared plan, planned-path extension, pose
history, age compensation) and run through convoy.simulate under identical
conditions on the two test scenarios, which RL never saw in training.
Output: results/rl_eval.json, printed table.
Usage: python scripts/rl_eval.py [seed ...]   (default: seeds with a best model)
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

INFO = dict(leader_preview=True, leader_plan=True, plan_compensation=True,
            plan_path_ext=True, pose_history=30)
NMPC = dict(INFO, gap_nmpc=True, follower_preview=True)
COND = {
    "s1_nominal": ("ideal", dict()),
    "s2_nominal": ("gnss", dict()),
    "s2_delay300_loss20": ("gnss", dict(v2v_delay=0.3, v2v_loss=0.2, v2v_seed=0)),
    "s2_blackout": ("gnss", dict(v2v_delay=0.1, v2v_blackout=(6.0, 8.0))),
    "s2_realistic_ekf": ("gnss", dict(v2v_delay=0.1, v2v_loss=0.1, v2v_seed=0,
                                      sensing="ekf", gnss_sigma=0.02, noise_seed=200)),
    "s2_rigid_bar": ("gnss", dict(link_rigid=True)),
    "s2_rigid_realistic": ("gnss", dict(link_rigid=True, v2v_delay=0.1, v2v_loss=0.1,
                                        v2v_seed=0, sensing="ekf", gnss_sigma=0.02,
                                        noise_seed=300)),
}
SCEN = {"ideal": ideal_path, "gnss": gnss_like_path}


def job(a):
    ctrl, cond = a
    scen, extra = COND[cond]
    policy = None
    if ctrl == "nmpc":
        K = ConvoyConfig(**NMPC, **extra)
    else:
        import torch
        from stable_baselines3 import SAC
        torch.set_num_threads(1)
        model = SAC.load(ROOT / "results" / "rl" / f"{ctrl}_best", device="cpu")
        policy = lambda o: model.predict(o, deterministic=True)[0]   # noqa: E731
        K = ConvoyConfig(**INFO, **extra)
    r = simulate(SCEN[scen](), K=K, follower_policy=policy)
    rms = lambda x: float(np.sqrt(np.mean(r[x] ** 2)))   # noqa: E731
    pk = lambda x: float(np.max(np.abs(r[x])))           # noqa: E731
    fms = r["solve_ms"][1::2]                            # follower's controller
    return dict(ctrl=ctrl, cond=cond, gap_rmse=rms("ed"), gap_peak=pk("ed"),
                lat_rmse=rms("eyF"), lat_peak=pk("eyF"),
                jerk_rms=float(np.sqrt(np.mean((np.diff(r["aF"]) / 0.1) ** 2))),
                accel_sat=float(np.mean((r["aF"] <= -1.99) | (r["aF"] >= 0.99))),
                steer_sat=float(np.mean(np.abs(r["dF"]) > 0.39)),
                force_peak=pk("f_link"), lead_lat_peak=pk("eyL"),
                ctrl_ms_mean=float(np.mean(fms)), ctrl_ms_max=float(np.max(fms)),
                finished=bool(r["t"][-1] > 30.0))


if __name__ == "__main__":
    seeds = sys.argv[1:] or [p.stem.split("_")[1][4:] for p in
                             sorted((ROOT / "results" / "rl").glob("sac_seed*_best.zip"))]
    ctrls = ["nmpc"] + [f"sac_seed{s}" for s in seeds]
    jobs = [(c, k) for k in COND for c in ctrls]
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(job, jobs))
    (ROOT / "results" / "rl_eval.json").write_text(json.dumps(res, indent=1))
    print(f"{'condition':22}{'ctrl':12}{'gap rms':>9}{'gap pk':>8}{'lat rms':>9}"
          f"{'lat pk':>8}{'jerk':>7}{'a sat':>7}{'F pk N':>8}{'lead pk':>8}{'ms':>7}")
    for r in res:
        print(f"{r['cond']:22}{r['ctrl']:12}{r['gap_rmse']:9.3f}{r['gap_peak']:8.2f}"
              f"{r['lat_rmse']:9.3f}{r['lat_peak']:8.2f}{r['jerk_rms']:7.2f}"
              f"{r['accel_sat']:7.0%}{r['force_peak']:8.0f}{r['lead_lat_peak']:8.2f}"
              f"{r['ctrl_ms_mean']:7.1f}")
