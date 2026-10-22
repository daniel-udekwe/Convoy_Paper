"""Error metrics in the format of Tables 3-5 (MSE, RMSE, MAE) plus peak."""
import numpy as np

ROWS = [("Follower", "Lateral (m)", "eyF"), ("Follower", "Yaw (rad)", "epsiF"),
        ("Follower", "Speed (m/s)", "evF"), ("Follower", "Distance (m)", "ed"),
        ("Leader", "Lateral (m)", "eyL"), ("Leader", "Yaw (rad)", "epsiL"),
        ("Leader", "Speed (m/s)", "evL")]


def table(res):
    out = []
    for veh, name, key in ROWS:
        e = res[key]
        out.append(dict(vehicle=veh, error=name, MSE=np.mean(e ** 2),
                        RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)),
                        peak=np.max(np.abs(e))))
    return out


def fmt(rows, title=""):
    s = [title, f"{'':9}{'Error':14}{'MSE':>9}{'RMSE':>9}{'MAE':>9}{'Peak':>9}"]
    for r in rows:
        s.append(f"{r['vehicle']:9}{r['error']:14}{r['MSE']:9.4f}"
                 f"{r['RMSE']:9.4f}{r['MAE']:9.4f}{r['peak']:9.4f}")
    return "\n".join(s)
