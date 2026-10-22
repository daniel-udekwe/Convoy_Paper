"""Robustness of the convoy controllers to V2V delay, loss, blackout and
GNSS noise (contribution C5), on the GNSS-like scenario.

Controllers
  baseline   : Eq. 40 PI (needs V2V only to fill the path buffer)
  ff         : PI + leader-speed feedforward (speed taken from V2V)
  gap_naive  : proposed gap-aware NMPC, uses the latest plan as if fresh
  gap_comp   : proposed, with each plan shifted forward by its age
  gap_robust : gap_comp + planned-path extension + 3 s pose history per message

Output: results/robustness.json, printed summary.
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

GAP = dict(gap_nmpc=True, follower_preview=True, leader_preview=True, leader_plan=True)
CTRL = {"baseline": dict(), "ff": dict(feedforward=True),
        "gap_naive": dict(GAP), "gap_comp": dict(GAP, plan_compensation=True),
        "gap_robust": dict(GAP, plan_compensation=True, plan_path_ext=True,
                           pose_history=30)}


def conditions():
    out = []
    for d in (0.0, 0.1, 0.2, 0.3):
        out.append(dict(cond="delay", delay=d, loss=0.0, seed=0))
        for seed in range(3):
            out.append(dict(cond="delay+loss20", delay=d, loss=0.2, seed=seed))
    # 2 s blackout across the first braking event (leader slows near t = 7 s)
    out.append(dict(cond="blackout", delay=0.1, loss=0.0, seed=0, blackout=(6.0, 8.0)))
    for seed in range(3):
        out.append(dict(cond="realistic", delay=0.1, loss=0.1, seed=seed,
                        pos_noise=0.02, heading_noise=0.005))
    return out


def job(args):
    ctrl, c = args
    K = ConvoyConfig(**CTRL[ctrl], v2v_delay=c["delay"], v2v_loss=c["loss"],
                     v2v_seed=c["seed"], noise_seed=100 + c["seed"],
                     v2v_blackout=c.get("blackout"),
                     pos_noise=c.get("pos_noise", 0.0),
                     heading_noise=c.get("heading_noise", 0.0))
    r = simulate(gnss_like_path(), K=K)
    ed, ey, a = r["ed"], r["eyF"], r["aF"]
    return dict(ctrl=ctrl, **{k: v for k, v in c.items() if k != "blackout"},
                gap_rmse=float(np.sqrt(np.mean(ed ** 2))),
                gap_peak=float(np.max(np.abs(ed))),
                lat_rmse=float(np.sqrt(np.mean(ey ** 2))),
                lat_peak=float(np.max(np.abs(ey))),
                jerk_rms=float(np.sqrt(np.mean((np.diff(a) / 0.1) ** 2))),
                accel_sat=float(np.mean((a <= -1.99) | (a >= 0.99))),
                max_age_s=float(r["age"].max() * 0.1))


if __name__ == "__main__":
    # optional: python robustness.py gap_robust  -> run only those, merge in
    only = sys.argv[1:] or list(CTRL)
    out_f = ROOT / "results" / "robustness.json"
    prev = [r for r in json.loads(out_f.read_text()) if r["ctrl"] not in only] \
        if out_f.exists() and sys.argv[1:] else []
    jobs = [(k, c) for c in conditions() for k in only]
    with ProcessPoolExecutor(2) as ex:
        res = []
        for i, r in enumerate(ex.map(job, jobs)):
            res.append(r)
            print(f"[{i + 1}/{len(jobs)}] {r['ctrl']:10s} {r['cond']:13s} "
                  f"d={r['delay']:.1f} loss={r['loss']:.1f} s={r['seed']} "
                  f"gap {r['gap_rmse']:.3f} ({r['gap_peak']:.2f}) "
                  f"lat {r['lat_rmse']:.3f}", flush=True)
    out_f.write_text(json.dumps(prev + res, indent=1))
