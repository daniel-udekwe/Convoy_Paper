"""State estimation study (contribution C5): truth vs raw sensors vs EKF.

Robust gap-aware NMPC on the GNSS-like path, realistic V2V link (100 ms
delay, 10% loss). GNSS grades: RTK (2 cm) and standard/SBAS (0.3 m).
Output: results/estimation.json, printed table.
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
              pose_history=30, v2v_delay=0.1, v2v_loss=0.1)


def job(a):
    sens, sg, seed = a
    K = ConvoyConfig(**ROBUST, sensing=sens, gnss_sigma=sg, v2v_seed=seed,
                     noise_seed=200 + seed)
    r = simulate(gnss_like_path(), K=K)
    rms = lambda k: float(np.sqrt(np.mean(r[k] ** 2)))   # noqa: E731
    pk = lambda k: float(np.max(np.abs(r[k])))           # noqa: E731
    return dict(sensing=sens, gnss_sigma=sg, seed=seed,
                gap_rmse=rms("ed"), gap_peak=pk("ed"),
                lat_rmse=rms("eyF"), lat_peak=pk("eyF"),
                jerk_rms=float(np.sqrt(np.mean((np.diff(r["aF"]) / 0.1) ** 2))),
                est_pos_rmse=rms("est_posF"), est_psi_rmse=rms("est_psiF"),
                est_vy_rmse=rms("est_vyF"))


if __name__ == "__main__":
    jobs = [("truth", 0.0, s) for s in range(3)]
    jobs += [(k, sg, s) for sg in (0.02, 0.3) for k in ("raw", "ekf") for s in range(3)]
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(job, jobs))
    (ROOT / "results" / "estimation.json").write_text(json.dumps(res, indent=1))
    keys = ["gap_rmse", "gap_peak", "lat_rmse", "lat_peak", "jerk_rms",
            "est_pos_rmse", "est_psi_rmse", "est_vy_rmse"]
    print(f"{'sensing':8}{'GNSS':>6}" + "".join(f"{k:>14}" for k in keys))
    for sens, sg in [("truth", 0.0), ("raw", 0.02), ("ekf", 0.02), ("raw", 0.3), ("ekf", 0.3)]:
        v = [r for r in res if r["sensing"] == sens and r["gnss_sigma"] == sg]
        row = [np.max([x[k] for x in v]) if "peak" in k else np.mean([x[k] for x in v])
               for k in keys]
        print(f"{sens:8}{sg:6.2f}" + "".join(f"{x:14.3f}" for x in row))
