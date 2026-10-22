"""Vehicle and controller parameters.

Vehicle values follow Table 2 of Franci et al. (2026). Parameters the paper
does not report (yaw-rate time constant tau, NMPC weights, horizon, PI gains)
are our choices and are marked ASSUMED; tune them against Table 5.
"""
from dataclasses import dataclass, field


@dataclass
class Vehicle:
    m: float = 450.0        # kg            (Table 2)
    lf: float = 0.8         # m             (Table 2)
    lr: float = 0.8         # m             (Table 2)
    Iz: float = 270.0       # kg m^2        (Table 2)
    Cf: float = 10000.0     # N/rad per tyre (Table 2)
    Cr: float = 9000.0      # N/rad per tyre (Table 2)
    vmax: float = 7.0       # m/s           (Table 2)
    tau: float = 0.3        # s, accel lag  ASSUMED (paper does not give it)
    v_floor: float = 1.0    # m/s, guard on 1/vx terms (see contribution C4)


@dataclass
class MPCConfig:
    Np: int = 20            # prediction steps          ASSUMED
    T: float = 0.1          # control period, s         ASSUMED
    n_sub: int = 4          # RK4 sub-steps per period
    a_min: float = -2.0     # m/s^2                      (Sec. 4.3)
    a_max: float = 1.0      # m/s^2                      (Sec. 4.3)
    d_max: float = 0.4      # rad                        (Sec. 4.3)
    # cost weights                                       ASSUMED
    q_ey: float = 20.0
    q_epsi: float = 10.0
    q_v: float = 30.0
    r_a: float = 0.05
    r_d: float = 0.5
    s_a: float = 0.5
    s_d: float = 30.0
    # gap-aware follower (contribution C1, full)              ASSUMED
    q_gap: float = 20.0     # arc-length spacing error
    q_v_gap: float = 2.0    # weak tie to the leader's predicted speed


@dataclass
class ConvoyConfig:
    d0: float = 6.0         # m, reference gap           (Sec. 4.3)
    Kp: float = 1.0         # PI proportional gain       ASSUMED; best worst-case over both scenarios
    Ki: float = 0.3         # PI integral gain           ASSUMED; stiffer gains oscillate in scenario 2
    feedforward: bool = False   # False = Eq. 40 exactly; True = v_L + PI
    follower_preview: bool = False  # False = nearest-point, constant curvature
    leader_preview: bool = False
    gap_nmpc: bool = False      # True = spacing inside the follower NMPC (C1, full);
                                # leader shares its predicted speed/curvature plan
    leader_plan: bool = False   # follower preview uses the leader's shared plan
                                # beyond the end of the buffer (C2, full)
    # ---- communication and sensing (contribution C5). Defaults = ideal.
    v2v_delay: float = 0.0      # s, leader -> follower message latency
    v2v_loss: float = 0.0       # probability each message is lost
    v2v_blackout: tuple = None  # (t_start, t_end) s: all messages lost
    v2v_seed: int = 0
    plan_compensation: bool = False  # shift received plans forward by their age
    plan_path_ext: bool = False  # extend the buffer with the leader's planned path
    gap_anchor: bool = False    # current spacing offset from the link sensor
    pose_history: int = 0       # poses repeated in each message (30 = 3 s)
    pos_noise: float = 0.0      # m, GNSS position noise (1 sigma), both vehicles
    heading_noise: float = 0.0  # rad, heading noise (1 sigma)
    noise_seed: int = 1
    sensing: str = "truth"      # 'truth' | 'raw' | 'ekf'  (convoy/estimation.py)
    gnss_sigma: float = 0.02    # m, GNSS position noise for 'raw'/'ekf'
    # ---- mechanical coupling (contribution C3). k = 0: no physical link.
    link_k: float = 0.0         # N/m, link stiffness
    link_zeta: float = 0.7      # damping ratio (c = 2 zeta sqrt(k m/2))
    hitch_L: float = 1.2        # m, leader hitch behind its CoM
    hitch_F: float = 1.2        # m, follower hitch ahead of its CoM
    hitch_aware: bool = False   # regulate hitch-to-hitch length, not CoM gap
    link_rigid: bool = False    # rigid drawbar with pin joints (constraint force)
    dt_plant: float = 0.005  # plant integration step, s
