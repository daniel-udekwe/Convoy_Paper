"""Grid-search PI gains for the baseline (Eq. 40) on the ideal path.

The paper does not report Kp, Ki. To give the baseline its best case we pick
the gains that minimise gap RMSE without oscillation, then reuse them for
every variant so comparisons are fair.
"""
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convoy.convoy import simulate          # noqa: E402
from convoy.params import ConvoyConfig      # noqa: E402
from convoy.paths import ideal_path         # noqa: E402


def run(gains):
    Kp, Ki, ff = gains
    r = simulate(ideal_path(), K=ConvoyConfig(Kp=Kp, Ki=Ki, feedforward=ff))
    ed = r["ed"]
    return dict(Kp=Kp, Ki=Ki, ff=ff, rmse=float(np.sqrt(np.mean(ed ** 2))),
                peak=float(np.max(np.abs(ed))),
                accel_rms=float(np.sqrt(np.mean(r["aF"] ** 2))))


if __name__ == "__main__":
    grid = list(itertools.product([0.5, 1.0, 2.0, 4.0], [0.3, 1.0, 3.0],
                                  [False, True]))
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(run, grid))
    res.sort(key=lambda d: (d["ff"], d["rmse"]))
    for d in res:
        print(d)
    out = Path(__file__).resolve().parents[1] / "results" / "pi_sweep.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
