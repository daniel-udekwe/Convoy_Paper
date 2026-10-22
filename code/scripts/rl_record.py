"""Record the leader NMPC on random paths for RL training / validation.

Training: seeds 0-59; validation (checkpoint selection): seeds 1000-1007.
The two test scenarios are never used here.
Output: results/rl/leaders_{train,val}.pkl (pickled list of dicts).
"""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convoy.paths import gnss_like_path, random_waypoints   # noqa: E402
from convoy.rl import record_leader                          # noqa: E402


def job(seed):
    rng = np.random.default_rng(seed)
    path = gnss_like_path(seed=seed, wp=random_waypoints(seed),
                          vcap=rng.uniform(4.5, 6.0))
    return record_leader(path)


if __name__ == "__main__":
    out = ROOT / "results" / "rl"
    out.mkdir(parents=True, exist_ok=True)
    for name, seeds in [("val", range(1000, 1008)), ("train", range(60))]:
        with ProcessPoolExecutor(2) as ex:
            recs = list(ex.map(job, seeds))
        (out / f"leaders_{name}.pkl").write_bytes(pickle.dumps(recs))
        steps = sum(len(r["x"]) for r in recs)
        print(f"{name}: {len(recs)} recordings, {steps} steps", flush=True)
