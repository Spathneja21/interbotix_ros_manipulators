#!/usr/bin/env python3
"""Phase 3 checkpoint / milestone M3: does every joint actually track a
commanded velocity, and does it stop when told?

    python3 experiments/verify_tracking.py

For each arm joint, at each of several commanded speeds:
  - run until the joint has covered a fixed angle (capped at 2 s)
  - measure steady-state velocity over the last 40% of the motion
  - command zero, then separate two things that are easily conflated:
      stopping distance -- how far it travels while decelerating, which
                           legitimately scales with speed
      creep             -- whether it keeps moving once it HAS stopped,
                           measured over a 1 s window opened 2 s after the stop
                           so the exponential settling tail is excluded

Two things are being checked. The obvious one is steady-state tracking. The
other is the low-speed floor: every joint carries frictionloss = 0.1 N m from
the URDF's <dynamics friction="0.1"/>, which on wrist_rotate is 10% of its
1 N m effort limit. If that produces a velocity deadband it will show up as the
error blowing up at the smallest commanded speeds, and we would rather know now
than mistake it for a gain problem in phase 4.

Runs on the floorless bench so a joint swinging into the ground cannot be
mistaken for a tracking failure.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_sim import ARM_JOINTS                          # noqa: E402
from velocity_controller import VelocityController      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SPEEDS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

# Direction to drive each joint from home so it moves into free space rather
# than into the base. See plan.md phase 2: the upper arm reaches the base at
# shoulder 100.6 deg, inside the joint limit.
DIRECTION = {"waist": +1, "shoulder": -1, "elbow": -1,
             "wrist_angle": -1, "wrist_rotate": +1}

TRAVEL = 0.8        # radians of motion to aim for
T_MAX = 2.0
T_MIN = 0.6
SETTLE_TRANSIENT = 2.0   # let the deceleration tail decay
SETTLE_WINDOW = 1.0      # then measure creep over this window


def trial(joint, speed):
    v = speed * DIRECTION[joint]
    duration = float(np.clip(TRAVEL / speed, T_MIN, T_MAX))

    ctrl = VelocityController(mode="intvelocity", bench=True)
    ctrl.reset("home")
    ctrl.set_joint(joint, v, warn=False)
    ctrl.run(duration)

    t = ctrl.trace.t
    qd = ctrl.trace.qd[:, ARM_JOINTS.index(joint)]
    steady = qd[t > t[0] + 0.6 * duration]
    achieved = float(steady.mean())

    # Two different things get conflated if you just measure "how far did it
    # move after the stop". Coming down from 2 rad/s the joint has to
    # decelerate, and that costs real distance -- stopping distance, not drift.
    # Drift is whether it keeps creeping once it HAS stopped.
    #
    # The settling tail is exponential but not instant: wrist_rotate is the
    # lightest joint and carries the lowest kp, so from 2 rad/s its residual
    # velocity decays 5e-3 -> 2e-3 -> 5e-4 -> 2e-5 rad/s over the following
    # seconds. Sampling drift half a second after the stop therefore measures
    # the tail of that transient, not creep -- so wait SETTLE_TRANSIENT before
    # opening the measurement window.
    q_at_stop = ctrl.q.copy()
    ctrl.stop()
    ctrl.run(SETTLE_TRANSIENT)
    q_settled = ctrl.q.copy()
    ctrl.run(SETTLE_WINDOW)

    stop_dist = float(np.degrees(np.abs(ctrl.q - q_at_stop)).max())
    drift = float(np.degrees(np.abs(ctrl.q - q_settled)).max())
    ncon = int(ctrl.sim.data.ncon)

    return dict(joint=joint, speed=speed, cmd=v, achieved=achieved,
                err_pct=100 * (abs(achieved) - speed) / speed,
                drift=drift, stop_dist=stop_dist, ncon=ncon,
                resid=float(np.abs(ctrl.qd).max()))


def main():
    results = [trial(j, s) for j in ARM_JOINTS for s in SPEEDS]

    print(f"{'joint':<14}{'cmd':>8}{'achieved':>10}{'err':>9}"
          f"{'stopping dist':>15}{'creep':>10}{'residual qd':>14}")
    print(f"{'':14}{'rad/s':>8}{'rad/s':>10}{'%':>9}{'deg':>15}"
          f"{'deg/s':>10}{'rad/s':>14}")
    print("-" * 80)
    for j in ARM_JOINTS:
        for r in [x for x in results if x["joint"] == j]:
            flag = "  <-- deadband" if abs(r["err_pct"]) > 5 else ""
            print(f"{r['joint']:<14}{abs(r['cmd']):8.3f}{abs(r['achieved']):10.4f}"
                  f"{r['err_pct']:+9.2f}{r['stop_dist']:15.3f}"
                  f"{r['drift']:10.4f}{r['resid']:14.2e}{flag}")
        print()

    contacts = [r for r in results if r["ncon"]]
    if contacts:
        raise RuntimeError(f"{len(contacts)} trial(s) made contact: "
                           f"{[(r['joint'], r['speed']) for r in contacts]}")

    # -- verdicts ------------------------------------------------------------
    fast = [r for r in results if r["speed"] >= 0.1]
    slow = [r for r in results if r["speed"] < 0.1]
    worst_fast = max(fast, key=lambda r: abs(r["err_pct"]))
    worst_drift = max(results, key=lambda r: r["drift"])

    print("=" * 74)
    ok = True
    v = abs(worst_fast["err_pct"]) < 5.0
    ok &= v
    print(f"[{'PASS' if v else 'FAIL'}] tracking within 5% at >=0.1 rad/s "
          f"(worst: {worst_fast['joint']} @ {worst_fast['speed']} rad/s, "
          f"{worst_fast['err_pct']:+.2f}%)")

    v = worst_drift["drift"] < 0.05
    ok &= v
    print(f"[{'PASS' if v else 'FAIL'}] stays stopped: <0.05 deg of creep over 1 s, "
          f"measured 2 s after the stop\n       (worst: {worst_drift['joint']} @ "
          f"{worst_drift['speed']} rad/s, {worst_drift['drift']:.4f} deg)")

    worst_stop = max(results, key=lambda r: r["stop_dist"])
    print(f"       stopping distance is a separate matter and scales with speed: "
          f"worst is\n       {worst_stop['joint']} @ {worst_stop['speed']} rad/s "
          f"-> {worst_stop['stop_dist']:.2f} deg to come to rest")

    v = all(r["resid"] < 1e-2 for r in results)
    ok &= v
    print(f"[{'PASS' if v else 'FAIL'}] residual velocity after stop < 1e-2 rad/s")

    worst_slow = max(slow, key=lambda r: abs(r["err_pct"]))
    print(f"\nlow-speed behaviour (the frictionloss question): worst error below "
          f"0.1 rad/s is\n  {worst_slow['joint']} @ {worst_slow['speed']} rad/s "
          f"-> {worst_slow['err_pct']:+.2f}%")

    # -- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for i, j in enumerate(ARM_JOINTS):
        rs = [r for r in results if r["joint"] == j]
        s = [r["speed"] for r in rs]
        ax[0].plot(s, [abs(r["achieved"]) for r in rs], "o-", label=j)
        ax[1].plot(s, [r["err_pct"] for r in rs], "o-", label=j)
    lim = [min(SPEEDS), max(SPEEDS)]
    ax[0].plot(lim, lim, "k--", lw=1, label="ideal")
    ax[0].set_xscale("log"), ax[0].set_yscale("log")
    ax[0].set_xlabel("commanded speed (rad/s)")
    ax[0].set_ylabel("achieved speed (rad/s)")
    ax[0].set_title("(a) steady-state tracking", loc="left", fontweight="bold")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].axhspan(-5, 5, color="green", alpha=0.08)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("commanded speed (rad/s)")
    ax[1].set_ylabel("error (%)")
    ax[1].set_title("(b) tracking error (green band = ±5%)", loc="left",
                    fontweight="bold")
    for a in ax:
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    fig.suptitle("WX200 joint-velocity tracking, intvelocity actuators",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "verify_tracking.png")
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
