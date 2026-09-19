#!/usr/bin/env python3
"""Run the Cartesian reach controller against the MuJoCo model, with dynamics.

    python3 test_reach_sim.py 0.35 0 0.20

Same control law as reach_and_log_hw.py -- both import cartesian_reach -- so
this is the dress rehearsal. Run it before pointing the real arm at a new
target.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "mujoco_arm"))
from velocity_controller import VelocityController      # noqa: E402
from cartesian_reach import CartesianReach, Z_OFFSET    # noqa: E402

TIMEOUT = 15.0


def main():
    target = np.array([float(v) for v in sys.argv[1:4]]) if len(sys.argv) > 3 \
        else np.array([0.35, 0.0, 0.20])

    reach = CartesianReach()
    ctrl = VelocityController(mode="intvelocity", rate_hz=50.0)
    ctrl.reset("home")

    start = reach.fk(ctrl.q)
    print(f"target {target}  start {start.round(4)}  "
          f"distance {np.linalg.norm(target - start):.4f} m")

    ticks = int(TIMEOUT * ctrl.rate_hz)
    hist = []
    rows = []
    reached_at = None
    for k in range(ticks):
        q = ctrl.q
        qd, err, dist = reach.step(q, target)
        sigma = reach.manipulability(q)
        hist.append((ctrl.sim.t, dist, sigma))
        # same CSV schema as reach_and_log_hw.py; in sim there is no independent
        # TF measurement, so ee_* and fk_* are both the model's FK
        fk = reach.fk(q)
        rows.append([ctrl.sim.t, *q, *fk, *fk, *qd, *ctrl.qd, dist, sigma])
        if dist < reach.tol and reached_at is None:
            reached_at = ctrl.sim.t
            break
        ctrl.set_joint_velocity(qd, warn=False)
        ctrl.run(1.0 / ctrl.rate_hz)

    ctrl.stop()
    ctrl.run(0.5)

    final = reach.fk(ctrl.q)
    dist = np.linalg.norm(target - final)
    sigmas = [h[2] for h in hist]

    print(f"final  {final.round(4)}")
    print(f"error  {(target - final).round(5)}  |e| = {dist*1000:.2f} mm")
    print(f"time   {reached_at if reached_at else '>timeout'} s")
    print(f"min sigma along the path: {min(sigmas):.4f} "
          f"({'ok' if min(sigmas) > 0.02 else 'SINGULAR - do not run on hardware'})")
    out = sys.argv[4] if len(sys.argv) > 4 else "reach_sim.csv"
    from arm_sim import ARM_JOINTS
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + ARM_JOINTS + ["ee_x", "ee_y", "ee_z"]
                   + ["fk_x", "fk_y", "fk_z"]
                   + [f"qd_cmd_{j}" for j in ARM_JOINTS]
                   + [f"qd_act_{j}" for j in ARM_JOINTS] + ["dist", "sigma"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} samples)")
    print(f"joints (deg): {np.degrees(ctrl.q).round(1)}")
    print(f"peak |qd| during motion: {np.abs(ctrl.trace.qd).max():.3f} rad/s")
    print(f"\nNOTE: this model is the standalone wx200. It does NOT contain the "
          f"LoCoBot's\ncamera tower, plate or battery, so it cannot check for "
          f"collisions with them.\nZ_OFFSET {Z_OFFSET*1000:+.3f} mm is applied "
          f"to report positions in the robot's frame.")
    return 0 if dist < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
