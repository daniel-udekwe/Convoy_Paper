# convoy-nmpc

Python reproduction and extension of Franci, Pugi, Gulino & Alessandrini (2026),
*Design of a mixed PI–NMPC control strategy for automated vehicles: application
to mechanically connected convoys*, J. Intelligent Transportation Systems,
doi:10.1080/15472450.2026.2669494.

Status: baseline reproduced (phase 1); contributions C1 (spacing inside the
NMPC) and C2 (arc-length curvature preview) implemented and ablated (phase 2);
robustness to V2V delay, loss, blackout and GNSS noise tested, with EKF state
estimation (C5, phase 3); spring-damper and rigid-drawbar coupling forces
modelled (C3, phase 4); model-free RL (SAC) baseline trained and compared
(phase 5).

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_experiments.py      # ~2.5 min on 2 cores -> results/metrics.md
python scripts/make_figures.py         # -> results/figures/*.png
python scripts/robustness.py           # ~20 min: 100 runs -> results/robustness.json
python scripts/robustness_figure.py    # -> results/figures/robustness_delay.png
python scripts/robustness_blackout.py  # -> results/figures/robustness_blackout.png
python scripts/estimation_study.py     # ~4 min -> results/estimation.json
python scripts/coupling_study.py       # ~5 min -> results/coupling.json
python scripts/coupling_figure.py      # -> results/figures/coupling_force.png
python scripts/rigid_study.py          # ~6 min -> results/rigid.json
python scripts/rigid_figure.py         # -> results/figures/rigid_bar.png
# RL baseline (needs: pip install torch stable-baselines3 gymnasium)
python scripts/rl_record.py            # ~8 min: leader on 68 random paths
python scripts/rl_train.py --seed 0    # ~2 h per seed on one CPU core
python scripts/rl_eval.py              # ~15 min -> results/rl_eval.json
python scripts/rl_figure.py            # -> results/figures/rl_vs_nmpc.png
python scripts/rl_curve.py             # -> results/figures/rl_training_curve.png
python scripts/rl_reward_curve.py      # -> results/figures/rl_reward_curve.png (needs the episode logs from rl_train.py)
python scripts/paper_figures.py        # -> scenarios, trajectories, tracking_errors, inputs_speed (.png)
```

## What is implemented

| File | Contents |
| --- | --- |
| `convoy/params.py` | Vehicle (Table 2), NMPC and PI settings; unreported values marked ASSUMED |
| `convoy/plant.py` | Dynamic bicycle model, linear tyres, first-order acceleration lag (Eqs. 1–5); spring-damper link or rigid pin-jointed drawbar between hitch points (C3) |
| `convoy/nmpc.py` | CasADi/IPOPT NMPC on the nonlinear path-error model (Eqs. 6–36); optional `gap_mode` adds arc-length progress and a spacing cost (C1) |
| `convoy/paths.py` | Ideal path (straights + arcs, cf. Fig. 7a), GNSS-like path (cf. Fig. 7b), path/buffer projection |
| `convoy/convoy.py` | Two-vehicle convoy: leader NMPC, FIFO buffer, follower NMPC with PI spacing (Fig. 8) or gap-aware NMPC; chord-to-arc gap conversion; leader plan sharing |
| `convoy/estimation.py` | Sensor suite (GNSS, odometry, biased gyro, accelerometer), raw dead-reckoning and an EKF with mismatched model |
| `convoy/rl.py` | RL follower: shared observation builder, leader recorder, Gymnasium training environment |
| `convoy/metrics.py` | MSE / RMSE / MAE / peak, as in Tables 3–5 |
| `scripts/` | Experiments, figures, PI gain sweep |

## Results

Every variant uses the same vehicle, NMPC weights and PI gains (Kp = 1.0,
Ki = 0.3); each changes one thing. Follower gap error, RMSE (peak), in metres:

| Variant | Scenario 1 (ideal) | Scenario 2 (GNSS-like) |
| --- | --- | --- |
| Baseline (paper's Eq. 40, no preview) | 0.931 (2.61) | 1.216 (3.53) |
| PI + leader-speed feedforward | 0.087 (0.26) | 0.280 (0.92) |
| PI + feedforward + curvature preview | 0.090 (0.25) | 0.292 (0.78) |
| PI + feedforward + preview with leader plan | 0.090 (0.25) | 0.292 (0.78) |
| Gap-aware NMPC, no preview (C1) | 0.016 (0.05) | 0.023 (0.06) |
| **Proposed: gap-aware NMPC + preview (C1 + C2)** | **0.016 (0.05)** | **0.024 (0.06)** |

Follower lateral error, RMSE (peak), in metres:

| Variant | Scenario 1 | Scenario 2 |
| --- | --- | --- |
| Baseline | 0.042 (0.24) | 0.074 (0.45) |
| PI + feedforward + preview | 0.010 (0.04) | 0.055 (0.32) |
| Gap-aware NMPC, no preview | 0.010 (0.05) | 0.016 (0.08) |
| **Proposed** | **0.006 (0.03)** | **0.008 (0.03)** |

Follower actuation on scenario 2: jerk RMS 1.42 (baseline), 2.07 (PI +
feedforward), **0.82 m/s³ (proposed)**; acceleration at its limit 16%, 30%,
**3%** of the time. NMPC solves average about 30 ms (IPOPT), inside the
100 ms period. Full tables: `results/metrics.md`; figures: `results/figures/`.

### Findings

1. **Eq. 40 alone cannot hold the gap during speed changes.** With no
   feedforward, the integrator must carry the cruise speed, so every leader
   acceleration builds a gap error of roughly a/Ki. Across a 24-point gain
   sweep the baseline never beat about 0.7 m RMSE; stiffer gains oscillate
   against the actuator limits. The paper's ±0.05 m could not be reproduced
   under these assumptions, which suggests unreported details. Worth asking
   the authors.
2. **Feedforward is a cheap 4–11× gain** (v_ref = v_leader + PI), but the
   PI cascade still reacts only after an error appears.
3. **Spacing inside the NMPC is the main result: 50–60× lower gap error than
   the baseline** and 5–12× lower than feedforward PI. The follower's NMPC
   gets an arc-length progress state and a cost on where it should be at each
   horizon step, computed from the leader's shared predicted progress minus
   the arc lag that keeps the chord (the physical link length) at d0. It
   anticipates leader speed changes instead of reacting to them, and it is
   smoother: half the jerk and a tenth of the saturation of feedforward PI.
4. **Preview halves lateral error, but only once speed is smooth** (scenario 2:
   0.016 → 0.008 m with the gap-aware NMPC). With the PI cascade, speed
   oscillations dominate and preview barely helps.
5. **Continuing the preview with the leader's plan past the buffer end adds
   nothing measurable** here; the 6 m lag covers most of the 2 s horizon.
   Report as a negative result or drop.
6. **The buffer must store course angle, not yaw.** With sideslip the leader's
   yaw is not its direction of travel; using yaw as the path tangent caused
   bang-bang follower steering in fast corners (ideal-path lateral RMSE
   0.059 → 0.017 m after the fix).
7. **Paper note:** Eq. 15 takes the absolute value of curvature, which loses
   the turn direction; this code uses signed curvature.

## Robustness to communication and sensing (C5)

The gap-aware NMPC relies on the leader's plan arriving over V2V, so it was
tested on the GNSS-like path under message delay (0–300 ms), 20% random loss
(3 seeds), a 2 s blackout during the first braking event, and a combined
"realistic" case (100 ms delay, 10% loss, 2 cm GNSS position and 0.005 rad
heading noise on both vehicles, 3 seeds). 100 runs; logged errors are always
computed from true states. Follower gap RMSE (peak), in metres:

| Condition | Baseline | PI + feedforward | Gap NMPC, stale plan | Gap NMPC, age-compensated | **Robust (proposed)** |
| --- | --- | --- | --- | --- | --- |
| Ideal link | 1.216 (3.53) | 0.280 (0.92) | 0.024 (0.06) | 0.024 (0.06) | **0.024 (0.06)** |
| 100 ms delay | 1.216 (3.53) | 0.387 (1.31) | 0.527 (0.62) | 0.026 (0.07) | **0.026 (0.07)** |
| 300 ms delay | 1.216 (3.53) | 0.600 (2.06) | 1.580 (1.80) | 0.049 (0.20) | **0.049 (0.20)** |
| 300 ms + 20% loss | 1.214 (3.53) | 0.632 (2.21) | 1.729 (2.31) | 0.049 (0.20) | **0.049 (0.20)** |
| 2 s blackout | 1.224 (3.53) | 0.362 (1.32) | 1.675 (5.47) | 0.399 (2.06) | **0.028 (0.09)** |
| Realistic (delay + loss + noise) | 1.206 (3.63) | 0.408 (1.40) | 0.601 (1.02) | 0.027 (0.08) | **0.027 (0.08)** |

Follower lateral RMSE (peak) for the robust controller stays at 0.008–0.009 m
(≤ 0.04 m) in every delay/loss case and through the blackout, and 0.015 m
(0.05 m) with GNSS noise. In the blackout, PI + feedforward peaks at 1.17 m
lateral and the age-compensated NMPC at 0.80 m.

"Robust" = age-compensated plan + two additions for message loss:

- **Planned-path extension:** the follower tracks the buffer extended with the
  leader's planned path, integrated from its last received pose with the
  shared curvature plan (about 2 s, 10 m ahead). Without it, the buffer stops
  growing during an outage and the follower drives off its end.
- **Pose history:** each message repeats the leader's last 3 s of poses
  (about 150 numbers), so the first packets after an outage refill the hole
  instead of the buffer bridging it with a straight chord across a curve.

### Robustness findings

8. **Age compensation is essential.** Using the latest plan as if fresh turns
   100 ms of delay into a near-constant 0.5 m gap offset (about speed ×
   delay) and makes the gap-aware NMPC *worse than the baseline* beyond
   200 ms; with 20% loss it also chatters (jerk RMS 10.5 m/s³ vs 0.8).
   Shifting each received plan forward by its age keeps gap RMSE at
   0.024–0.049 m up to 300 ms with 20% loss.
9. **Feedforward PI degrades steadily with delay** (0.28 → 0.60 m at 300 ms),
   because the leader speed it uses is stale; the baseline (Eq. 40) is
   delay-immune but already poor.
10. **Outages need path redundancy, not just spacing logic.** The 2 s
    blackout breaks every controller without the planned-path extension and
    pose history; with both, the robust NMPC is unaffected (0.028 m gap,
    0.009 m lateral RMSE).
11. **Negative result:** anchoring the current spacing offset on the local
    link-sensor measurement (instead of the plan) did not help and was worse
    under delay (0.074 vs 0.049 m at 300 ms + 20% loss), so it was dropped.
12. **2 cm GNSS noise roughly doubles lateral RMSE (0.008 → 0.015 m) and
    jerk (0.9 → 1.8 m/s³)** but leaves gap error unchanged. An EKF fusing
    GNSS with odometry and IMU should recover most of that.

## State estimation (C5, continued)

Until here the controllers saw exact velocities and yaw rate. Each vehicle now
has a realistic sensor suite and **no heading sensor**: GNSS position, wheel
odometry (0.03 m/s), a gyro with a constant random bias (0.003 rad/s, 1σ)
plus 0.005 rad/s noise, and an accelerometer (0.05 m/s²). Three ways of
using them, all with the robust controller and the realistic link (100 ms
delay, 10% loss), 3 seeds each:

- **truth**: exact states (reference)
- **raw**: sensors used directly; heading from integrating the biased gyro
- **EKF**: fusion on [X, Y, ψ, vx, vy, r, a, gyro bias] with the bicycle model,
  its tyre stiffness, mass and inertia deliberately mismatched by 10–20%

| Sensing | GNSS 1σ | Gap RMSE (peak), m | Lateral RMSE (peak), m | Jerk RMS, m/s³ | Heading est. RMSE, rad |
| --- | --- | --- | --- | --- | --- |
| Truth | – | 0.026 (0.07) | 0.008 (0.03) | 0.89 | 0 |
| Raw | 2 cm (RTK) | 0.069 (0.33) | 0.179 (0.49) | 2.42 | 0.071 |
| **EKF** | 2 cm (RTK) | **0.033 (0.09)** | **0.020 (0.06)** | **1.26** | **0.008** |
| Raw | 0.3 m | 0.436 (1.20) | 0.302 (0.92) | 14.36 | 0.081 |
| **EKF** | 0.3 m | **0.062 (0.18)** | **0.097 (0.29)** | **1.38** | **0.010** |

13. **With RTK GNSS the EKF keeps the convoy close to perfect-state
    performance** (gap 0.033 vs 0.026 m, lateral 0.020 vs 0.008 m). Raw
    sensors fail mainly through heading: integrating a biased gyro drifts,
    and lateral error is 9× higher than with the EKF.
14. **With standard 0.3 m GNSS, raw sensors are unusable** (jerk 14 m/s³);
    the EKF holds 0.06 m gap and 0.10 m lateral error. The remaining lateral
    error is mostly leader-path error: the follower faithfully replicates the
    leader's *estimated* path. Both vehicles' GNSS errors are independent
    here; in practice they are largely common-mode, so a relative or
    moving-baseline RTK solution would do better.

## Mechanical coupling (C3)

The link is a spring-damper (damping ratio 0.7) between a hitch 1.2 m behind
the leader's CoM and one 1.2 m ahead of the follower's (rest length 3.6 m,
so the 6 m CoM gap leaves it unstressed on a straight). Link forces act on
both vehicles' longitudinal, lateral and yaw dynamics; the controllers do not
model the link. GNSS-like path, ideal link, exact states, to isolate the
effect. Each controller regulates either the CoM gap (as in the paper and
above) or the hitch-to-hitch length the link actually sees ("hitch-aware").

| Controller | Hitch-length error, no link: RMSE (peak), m | Peak link force, k = 2 / 20 / 200 kN/m (N) | Leader lateral peak at 200 kN/m, m |
| --- | --- | --- | --- |
| PI + feedforward, CoM gap | 0.284 (0.95) | 1041 / 2311 / 2441 | 0.362 |
| PI + feedforward, hitch length | 0.230 (0.75) | 427 / 2040 / 2324 | 0.266 |
| Gap NMPC, CoM gap | 0.052 (0.15) | 139 / 212 / 216 | 0.033 |
| **Gap NMPC, hitch length (proposed)** | **0.022 (0.06)** | **78 / 106 / 114** | **0.027** |

For scale, the cart weighs about 4.4 kN.

15. **Perfect path replication does not keep the link length constant.** On
    curves the hitch points sit off the CoM path, so regulating the CoM gap
    leaves up to 0.15 m of hitch-length variation even for the gap NMPC —
    exactly what a rigid bar would have to absorb. Regulating the hitch
    length instead (using the recorded course to place the follower's hitch)
    halves it.
16. **The proposed controller keeps link forces 13–22× lower than PI**: peak
    about 110 N (2.5% of the cart's weight) even with a near-rigid 200 kN/m
    link, against 2.3–2.4 kN (over half the cart's weight) for feedforward
    PI. Forces level off with stiffness because they are set by how well the
    follower tracks, not by the link.
17. **This qualifies the paper's conclusion.** Franci et al. conclude that a
    rigid link is incompatible with the spacing errors and recommend an
    elastic one. That holds for PI-based spacing: with PI, a stiff link
    drags the leader 0.36 m off its path. With predictive, hitch-aware
    spacing, the link carries little load at any stiffness, so a stiff link
    looks feasible — confirmed below with a truly rigid bar.

### Rigid drawbar

The link is replaced by a rigid bar with pin joints: the hitch-to-hitch
distance is held exactly at 3.6 m by a constraint force solved at every
integration step (Lagrange multiplier with Baumgarte stabilisation; bar
length drift stays below 4×10⁻⁸ m). Checked against a hand calculation: a
leader accelerating at 0.5 m/s² towing a coasting follower gives exactly
m·a/2 = 112.5 N of tension.

| Controller | Scenario | Bar force RMS / peak (N) | Leader lateral RMSE (peak), m | Leader speed error RMSE, m/s |
| --- | --- | --- | --- | --- |
| PI + feedforward, CoM gap | Ideal | 525 / 1514 | 0.091 (0.31) | 0.61 |
| **Proposed** | Ideal | **16 / 72** | **0.005 (0.03)** | **0.02** |
| PI + feedforward, CoM gap | GNSS-like | 652 / 2054 | 0.119 (0.39) | 0.76 |
| PI + feedforward, hitch length | GNSS-like | 329 / 1833 | 0.063 (0.33) | 0.46 |
| Gap NMPC, CoM gap | GNSS-like | 53 / 208 | 0.011 (0.03) | 0.05 |
| **Proposed** | GNSS-like | **28 / 111** | **0.010 (0.03)** | **0.03** |
| PI + feedforward, realistic stack × 3 seeds | GNSS-like | 509–523 / 2114–2265 | 0.076–0.082 (≤ 0.37) | 0.61 |
| **Proposed, realistic stack × 3 seeds** | GNSS-like | **33–36 / 126–209** | **0.017–0.020 (≤ 0.06)** | **0.03** |

"Realistic stack" = EKF on RTK GNSS, 100 ms V2V delay, 10% loss.

18. **The rigid bar confirms finding 16.** Peak bar force with the proposed
    controller is 111 N on the GNSS-like path (114 N with the 200 kN/m
    spring) and stays at 126–209 N with EKF sensing and a lossy, delayed
    link: about 2.5–5% of the cart's weight, 10–21× below PI in every case.
19. **With PI spacing, the follower fights the leader through the bar.**
    Bar force swings between about +2 kN and −2 kN, the leader is pushed
    0.39 m off its path, and its speed error rises to 0.76 m/s RMS (the run
    takes 5 s longer). With the proposed controller the leader tracks as if
    it were driving alone.

## Model-free RL baseline (SAC)

A Soft Actor-Critic policy replaces only the follower's NMPC. It gets the
same information as the robust gap NMPC: its own path errors and speeds, the
gap error, the leader's speed and acceleration from the age-compensated
shared plan, curvature 0–10 m ahead along the buffer extended with the
leader's planned path, and its previous command. Actions are acceleration
and steering squashed into the same actuator limits. The observation is
built by one function used by both the training environment and
`simulate()`, so trained policies run through every evaluation condition
unchanged.

- **Training:** 300k steps per seed (2 seeds, about 2 h each on one CPU
  core), replaying leader recordings on 60 random feasible paths (minimum
  radius 7.5 m, speed caps 4.5–6 m/s, 2 cm path noise), with an ideal V2V
  link, exact states and no physical link. Reward penalises lateral, heading
  and gap error and command changes; episodes end at 2 m lateral or 4 m gap
  error. The checkpoint with the lowest validation error on 8 further random
  paths is kept.
- **Testing:** the two scenarios, which RL never saw, under all the
  conditions above. Neither delay, loss, blackout, sensor noise nor the bar
  appeared in training.

| Condition | Gap RMSE, m: NMPC / SAC s0 / SAC s1 | Lateral RMSE, m | Jerk RMS, m/s³ |
| --- | --- | --- | --- |
| Ideal path | 0.016 / 0.022 / 0.039 | 0.006 / 0.016 / 0.016 | 0.44 / 0.75 / 0.75 |
| GNSS-like path | 0.024 / 0.053 / 0.062 | 0.008 / 0.013 / 0.024 | 0.82 / 1.98 / 2.06 |
| + 300 ms delay, 20% loss | 0.049 / 0.075 / 0.098 | 0.008 / 0.016 / 0.022 | 0.95 / 2.03 / 2.45 |
| + 2 s blackout | 0.028 / 0.056 / 0.070 | 0.009 / 0.015 / 0.024 | 0.90 / 2.29 / 2.78 |
| + EKF, 100 ms delay, 10% loss | 0.035 / 0.057 / 0.081 | 0.021 / 0.021 / 0.028 | 1.32 / 2.52 / 3.22 |
| + rigid bar | 0.047 / 0.047 / 0.047 | 0.013 / 0.018 / 0.021 | 1.08 / 2.04 / 2.49 |
| + rigid bar, EKF, delay, loss | 0.047 / 0.046 / 0.047 | 0.021 / 0.024 / 0.032 | 1.45 / 2.74 / 3.19 |

Peak bar force with the rigid bar: 208 / 239 / 546 N nominal and
221 / 421 / 681 N with EKF, delay and loss. Here the NMPC regulates the CoM
gap, like the RL policy; the hitch-aware NMPC brings it to 111 / 126–209 N.
Decision time per step: NMPC 27–29 ms (IPOPT), SAC 1.2–1.3 ms.

20. **SAC is a strong, robust baseline.** It never failed, and it degraded
    gracefully under delay, loss, blackout, EKF sensing and a rigid bar,
    none of which it was trained on. Most of that robustness comes from the
    information channels it shares with the NMPC (age-compensated plan,
    planned-path extension, pose history), not from learning.
21. **The NMPC is still better on every accuracy and comfort metric:** gap
    error 1.4–2.6× lower (equal with the rigid bar, which fixes the
    geometry), lateral 1.0–3.0×, jerk 1.7–3.1×, and peak bar force
    1.1–3.1× lower. RL's advantage is computation: about 20× faster
    decisions.
22. **RL results vary by seed.** Seed 1 is consistently worse than seed 0,
    with 1.6–2.3× the bar force, and its validation error drifted upward
    late in training. Any RL claim needs several seeds and checkpoint
    selection on held-out paths; the NMPC needs neither.
23. **RL loses accuracy off its training distribution.** On the 8 held-out
    validation paths (same generator as training), seed 0's best gap RMSE is
    0.028 m, within 10% of the NMPC's 0.0255 m on those same paths; on the
    unseen test path it is 0.053 m against the NMPC's 0.024 m. Lateral error
    stays about 2x the NMPC's throughout training (0.020 vs 0.0093 m).
    Seed 1's validation gap error rose from 0.04 to 0.09 m over its last
    60k steps, so its final model is worse than its 140k-step checkpoint
    (`results/figures/rl_training_curve.png`; the first seed-0 point, at
    0.82 m, is off the axis).
24. **Learning curves** (`results/figures/rl_reward_curve.png`, from the
    per-episode training logs in `results/rl/rerun/`). Both seeds reach about
    0.95 reward per step within about 30k steps and plateau at 0.970 / 0.959
    (last 100 episodes); the NMPC scored with the same reward on the same 60
    training paths gets 0.993. Seed 1 collapses transiently near 70k and 130k
    steps and recovers. The logged re-run reproduces the original training
    bit for bit (identical validation logs and best-model weights), so these
    curves describe exactly the evaluated models.

## Assumptions and differences from the paper

- PI gains, NMPC weights, horizon (20 × 0.1 s) and acceleration lag (0.3 s)
  are not reported; values are in `params.py`.
- Gap = straight-line CoM-to-CoM distance, d0 = 6 m. The coupling is not
  modelled yet (contribution C3).
- Phase 1 (follower drives straight to the leader's start) is implemented by
  pre-filling the buffer with that straight segment.
- Both vehicles start at 1 m/s because the model has 1/vx terms (guarded at
  1 m/s; contribution C4 will remove this).
- **Scenario 2 is synthetic.** No GNSS log was available, so it is a spline
  through hand-placed waypoints, sampled at 25 Hz with 2 cm noise, with heading
  and curvature re-estimated from the noisy samples. Its minimum radius is
  7.5 m and its speed is capped at 6 m/s, leaving 1 m/s of catch-up headroom
  below the 7 m/s vehicle limit. Swap in a real log via `Track.from_arrays`.
- The gap-aware follower uses the leader's NMPC plan (predicted speed,
  progress and curvature over the horizon) sent over V2V. Delay is fixed per
  run and loss is independent per message (Bernoulli); real links also have
  jitter and bursty loss, which the blackout case approximates.
- The link sensor (straight-line gap or hitch length) is assumed local,
  noise-free and delay-free. In the robustness study the NMPC sees noisy
  positions with exact velocities; the estimation study replaces this with
  the full sensor suite and EKF.
- The coupling is a spring-damper or rigid bar between pin joints at the
  hitch points, with no play, bending stiffness or joint friction, and tyres
  stay linear even under bar loads; the hitch-aware controller places the follower's hitch
  using the recorded course (ignoring sideslip, about 0.02 rad here).

## Next steps

1. A force-aware NMPC that predicts and bounds bar force explicitly, and
   nonlinear tyres under bar loads.
2. Model valid from standstill (C4), for stop-and-go.
3. N-vehicle convoys and string stability with physical coupling (C6).
4. acados for real-time timing; jitter and bursty loss on the V2V link.
   Residual RL (a learned correction on top of the NMPC) as a follow-up,
   since plain SAC trails the NMPC but is 20x cheaper to run.
5. A real GNSS log in place of the synthetic Scenario 2.
