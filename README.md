# Making rigid couplings feasible: gap-aware NMPC for mechanically connected automated convoys

Simulation code, result data, figures and trained RL models.

```
convoy_paper_package/
└── code/           simulation, controllers, experiments, results (Python)
│   ├── convoy/       vehicle model, NMPC, convoy simulation, EKF, coupling, RL environment
│   ├── scripts/      experiments and figure scripts
│   └── results/      result data (.json, .npz, logs), figures, RL models and training logs
```

## Reproducing the results

```bash
cd code
pip install -r requirements.txt
```

Run from `code/`. Times are for a 2-core machine.

| Step | Command | Time | Produces |
| --- | --- | --- | --- |
| PI gain sweep | `python scripts/tune_pi.py` | 5 min | gain selection (results/pi_sweep.json) |
| Baseline, ablation, tracking | `python scripts/run_experiments.py --kp 1.0 --ki 0.3` | 3 min | results/metrics.json, results/*.npz |
| Tracking figures | `python scripts/paper_figures.py` | 1 min | scenarios, trajectories, tracking_errors, inputs_speed |
| Ablation figure | `python scripts/make_figures.py` | <1 min | gap_error_summary |
| Communication robustness | `python scripts/robustness.py` | 25 min | results/robustness.json |
| Robustness figures | `python scripts/robustness_figure.py`, `python scripts/robustness_blackout.py` | 3 min | robustness_delay, robustness_blackout |
| State estimation | `python scripts/estimation_study.py` | 5 min | results/estimation.json |
| Spring–damper coupling | `python scripts/coupling_study.py`, `python scripts/coupling_figure.py` | 6 min | results/coupling.json, coupling_force |
| Rigid drawbar | `python scripts/rigid_study.py`, `python scripts/rigid_figure.py` | 8 min | results/rigid.json, rigid_bar |
| RL leader recordings | `python scripts/rl_record.py` | 8 min | results/rl/leaders_{train,val}.pkl |
| RL training (per seed) | `python scripts/rl_train.py --seed 0 --out results/rl/rerun` | 2 h | best model, validation curve, episode log |
| RL evaluation | `python scripts/rl_eval.py` | 15 min | results/rl_eval.json |
| RL figures | `python scripts/rl_figure.py`, `python scripts/rl_curve.py`, `python scripts/rl_reward_curve.py results/rl/rerun` | 15 min | rl_vs_nmpc, rl_training_curve, rl_reward_curve |

All figures are written to `code/results/figures/`. The RL steps can be skipped:
the trained models (`results/rl/sac_seed{0,1}_best.zip`), recordings and
training logs are included. The logged re-run in `results/rl/rerun/` reproduced
the original training bit for bit (identical validation logs and model
weights); only its episode logs are included, as its models duplicate the
originals.

## Notes

- Scenario 2 is a synthetic stand-in for a recorded GNSS path.
- Parameters not reported in the baseline paper (Franci et al., 2026) are marked
  in `code/convoy/params.py` and in Table 2 of the manuscript.
