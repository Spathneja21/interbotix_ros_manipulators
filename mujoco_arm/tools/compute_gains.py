#!/usr/bin/env python3
"""Size the velocity-actuator gains from the arm's actual joint-space inertia.

    python3 tools/compute_gains.py

Prints the gain table that is pasted into arm_sim.py. Re-run this if the model
changes (new damping, the mobile base, a payload) -- the numbers are derived,
not tuned by hand.

Method
------
For each arm joint take the WORST-CASE (largest) diagonal mass-matrix entry
M_ii over the reachable workspace, then place a critically damped second-order
closed loop at OMEGA_N:

    kp = M_max * w^2
    kv = 2 * zeta * M_max * w          (zeta = 1)

Using M_max rather than the median is the conservative choice: at lighter
configurations the loop becomes overdamped (zeta = sqrt(M_max/M) >= 1) instead
of ringing.

For the plain <velocity> servo there is no position term, so its kv is sized
for a first-order velocity response with time constant 1/OMEGA_N:

    kv = M_max * w

Note this deliberately departs from plan.md's original suggestion of seeding
from the Gazebo effort-controller gains. Those are hand-tuned numbers for a
different controller and they disagree with the physics here -- Gazebo weights
the shoulder 5x the waist, while the measured inertias say the two are nearly
equal. The measurement wins; see plan.md phase 2.
"""
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(os.path.dirname(HERE), "models", "scene.xml")

ARM_JOINTS = ["waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate"]
EFFORT = [8, 18, 13, 5, 1]          # URDF <limit effort=...>, N m

# Target closed-loop bandwidth. 40 rad/s ~= 6.4 Hz: fast enough that gravity
# droop on the shoulder stays under a degree, slow enough to stay far from the
# 0.002 s timestep (w*dt = 0.08).
OMEGA_N = 40.0
ZETA = 1.0
N_POSES = 2000


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    lo, hi = m.jnt_range[:5, 0], m.jnt_range[:5, 1]

    inertia = np.zeros((N_POSES, 5))
    gravity = np.zeros((N_POSES, 5))
    M = np.zeros((m.nv, m.nv))
    for k in range(N_POSES):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[:5] = rng.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, M, d.qM)
        inertia[k] = np.diag(M)[:5]
        gravity[k] = np.abs(d.qfrc_bias[:5])

    mmax = inertia.max(axis=0)
    gmax = gravity.max(axis=0)

    kp = mmax * OMEGA_N**2
    kv_int = 2 * ZETA * mmax * OMEGA_N
    kv_vel = mmax * OMEGA_N

    print(f"measured over {N_POSES} random poses, omega_n = {OMEGA_N} rad/s\n")
    print(f"{'joint':<14}{'M_max':>8}{'tau_g':>8}{'kp':>9}{'kv':>8}{'kv(vel)':>9}"
          f"{'droop':>9}")
    print(f"{'':14}{'kg m^2':>8}{'N m':>8}{'N m/rad':>9}{'N m s':>8}{'N m s':>9}"
          f"{'deg':>9}")
    print("-" * 65)
    for i, j in enumerate(ARM_JOINTS):
        # steady-state position error of the intvelocity servo holding gravity
        droop = np.degrees(gmax[i] / kp[i])
        print(f"{j:<14}{mmax[i]:8.5f}{gmax[i]:8.3f}{kp[i]:9.2f}{kv_int[i]:8.3f}"
              f"{kv_vel[i]:9.3f}{droop:9.2f}")

    print("\n# paste into arm_sim.py")
    print("GAINS = {")
    for i, j in enumerate(ARM_JOINTS):
        print(f'    "{j}": dict(kp={kp[i]:.2f}, kv={kv_int[i]:.3f}, '
              f'kv_vel={kv_vel[i]:.3f}),')
    print("}")

    print("\nsanity checks")
    for i, j in enumerate(ARM_JOINTS):
        sat = gmax[i] / EFFORT[i]
        # position error at which the servo saturates its effort limit
        esat = np.degrees(EFFORT[i] / kp[i])
        print(f"  {j:<14} gravity uses {100*sat:4.1f}% of the {EFFORT[i]} N m "
              f"limit; saturates at {esat:5.1f} deg of setpoint error")

    print("\n  a plain <velocity> servo cannot hold a static load: at zero")
    print("  commanded velocity its torque is kv*(0 - qd), so it settles into a")
    print("  CONSTANT SINK RATE tau_g/kv, it does not merely droop:")
    for i, j in enumerate(ARM_JOINTS):
        if gmax[i] > 1e-6:
            print(f"    {j:<14} {gmax[i]/kv_vel[i]:6.3f} rad/s "
                  f"({np.degrees(gmax[i]/kv_vel[i]):6.1f} deg/s)")


if __name__ == "__main__":
    main()
