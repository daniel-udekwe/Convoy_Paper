"""Train the SAC follower on recorded random paths.

Usage: python scripts/rl_train.py --seed 0 --steps 300000
Every 20k steps the deterministic policy is evaluated on the validation
paths; the checkpoint with the lowest validation cost (gap RMSE + lateral
RMSE, failures heavily penalised) is kept as the "best" model.
Outputs (in --out, default results/rl): sac_seed{N}_best.zip, _final.zip,
_curve.csv (validation) and sac_seed{N}.monitor.csv (every training episode:
return, length, wall time).
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from convoy.rl import make_env_class   # noqa: E402

RL = ROOT / "results" / "rl"
OUT = RL


def evaluate(model, env_cls, recs):
    """Deterministic rollouts on each validation recording."""
    gaps, lats, fails, rets = [], [], 0, []
    for i, rec in enumerate(recs):
        env = env_cls([rec], seed=10_000 + i)
        obs, _ = env.reset()
        done, ret, g, l = False, 0.0, [], []
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(act)
            ret += r
            ey, _, ed = env.err
            g.append(ed); l.append(ey)
            done = term or trunc
        fails += int(term)
        rets.append(ret)
        gaps.append(np.sqrt(np.mean(np.square(g))))
        lats.append(np.sqrt(np.mean(np.square(l))))
    cost = np.mean(gaps) + np.mean(lats) + 10.0 * fails
    return dict(ret=float(np.mean(rets)), gap=float(np.mean(gaps)),
                lat=float(np.mean(lats)), fails=fails, cost=float(cost))


class EvalCallback(BaseCallback):
    def __init__(self, env_cls, val, tag, every=20_000):
        super().__init__()
        self.env_cls, self.val, self.tag, self.every = env_cls, val, tag, every
        self.best = np.inf
        self.t0 = time.time()
        self.curve = OUT / f"{tag}_curve.csv"
        self.curve.write_text("steps,wall_s,val_return,val_gap_rmse,val_lat_rmse,val_fails\n")

    def _on_step(self):
        if self.num_timesteps % self.every:
            return True
        m = evaluate(self.model, self.env_cls, self.val)
        wall = time.time() - self.t0
        with self.curve.open("a") as f:
            f.write(f"{self.num_timesteps},{wall:.0f},{m['ret']:.1f},{m['gap']:.4f},"
                    f"{m['lat']:.4f},{m['fails']}\n")
        tag = ""
        if m["cost"] < self.best:
            self.best = m["cost"]
            self.model.save(OUT / f"{self.tag}_best")
            tag = "  <- best"
        print(f"[{self.tag}] {self.num_timesteps:7d} steps {wall / 60:5.1f} min  "
              f"val return {m['ret']:7.1f}  gap {m['gap']:.3f}  lat {m['lat']:.3f}  "
              f"fails {m['fails']}{tag}", flush=True)
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--out", default=str(RL))
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    train = pickle.loads((RL / "leaders_train.pkl").read_bytes())
    val = pickle.loads((RL / "leaders_val.pkl").read_bytes())
    Env = make_env_class()
    tag = f"sac_seed{a.seed}"
    env = Monitor(Env(train, seed=a.seed), filename=str(OUT / tag))  # episode log
    model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=300_000,
                batch_size=256, gamma=0.98, tau=0.005, train_freq=1,
                gradient_steps=1, learning_starts=5_000,
                policy_kwargs=dict(net_arch=[256, 256]), seed=a.seed, verbose=0)
    model.learn(total_timesteps=a.steps, callback=EvalCallback(Env, val, tag))
    model.save(OUT / f"{tag}_final")
    print(f"[{tag}] done", flush=True)
