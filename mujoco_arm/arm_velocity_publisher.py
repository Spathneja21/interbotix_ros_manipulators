#!/usr/bin/env python3
"""Command a constant joint velocity to the simulated WX200 for a fixed
duration, then stop. The arm-side counterpart of

    rosrun uan_base_control velocity_publisher.py -x 0.1 -t 3

Examples:
    python3 arm_velocity_publisher.py --joint elbow --vel -0.3 -t 2
    python3 arm_velocity_publisher.py --qd 0,0.2,-0.1,0,0 -t 2
    python3 arm_velocity_publisher.py --joint waist --vel 0.5 -t 2 --view
    python3 arm_velocity_publisher.py --joint elbow --vel -0.3 -t 2 \
            --csv run.csv --plot run.png

Velocities are rad/s. --qd takes all five joints in the wx200.yaml order:
waist, shoulder, elbow, wrist_angle, wrist_rotate. Anything beyond the URDF
limit of +-pi rad/s is clamped, with a warning, exactly like the base tool.

Unlike the real arm there is no watchdog here yet (phase 5), but the command is
still re-issued every control tick so that the timing behaves the same way.
"""
import argparse
import os
import sys

import numpy as np

from arm_sim import ARM_JOINTS
from velocity_controller import VelocityController

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--joint", choices=ARM_JOINTS,
                   help="move a single joint by name")
    g.add_argument("--qd", help="all five joint velocities, comma separated")
    p.add_argument("--vel", type=float, default=0.0,
                   help="velocity in rad/s, used with --joint")
    p.add_argument("-t", "--time", type=float, default=2.0,
                   help="how long to move, in seconds (default 2)")
    p.add_argument("-r", "--rate", type=float, default=50.0,
                   help="control rate in Hz (default 50)")
    p.add_argument("--settle", type=float, default=0.5,
                   help="seconds to keep simulating after the stop (default 0.5)")
    p.add_argument("--mode", default="intvelocity",
                   choices=["intvelocity", "velocity"],
                   help="actuator type (default intvelocity)")
    p.add_argument("--start", default="home", choices=["home", "sleep"],
                   help="starting keyframe (default home)")
    p.add_argument("--csv", help="write the trace to this CSV")
    p.add_argument("--plot", help="write a commanded-vs-actual plot to this PNG")
    p.add_argument("--view", action="store_true",
                   help="show the interactive viewer while running")
    return p.parse_args(argv[1:])


def build_command(args):
    if args.joint:
        qd = np.zeros(len(ARM_JOINTS))
        qd[ARM_JOINTS.index(args.joint)] = args.vel
        return qd
    parts = [s for s in args.qd.replace(" ", "").split(",") if s != ""]
    if len(parts) != len(ARM_JOINTS):
        sys.exit(f"--qd needs {len(ARM_JOINTS)} comma-separated values "
                 f"({', '.join(ARM_JOINTS)}), got {len(parts)}")
    try:
        return np.array([float(s) for s in parts])
    except ValueError as e:
        sys.exit(f"--qd: {e}")


def run_with_viewer(ctrl, qd, args):
    """Same motion, but stepped in wall-clock time with the viewer open."""
    import time

    import mujoco.viewer

    total = args.time + args.settle
    with mujoco.viewer.launch_passive(ctrl.sim.model, ctrl.sim.data) as v:
        start = time.time()
        while v.is_running():
            elapsed = ctrl.sim.t
            if elapsed >= total:
                break
            ctrl.set_joint_velocity(qd if elapsed < args.time else np.zeros(5),
                                    warn=False)
            ctrl.run(1.0 / ctrl.rate_hz)
            v.sync()
            # keep sim time roughly in step with wall time
            lag = ctrl.sim.t - (time.time() - start)
            if lag > 0:
                time.sleep(lag)


def main():
    args = parse_args(sys.argv)
    qd = build_command(args)

    ctrl = VelocityController(mode=args.mode, rate_hz=args.rate)
    ctrl.reset(args.start)
    # clamp + warn happens here, before anything moves
    ctrl.set_joint_velocity(qd)
    qd = ctrl.cmd.copy()

    named = ", ".join(f"{j}={v:+.3f}" for j, v in zip(ARM_JOINTS, qd) if v != 0)
    print(f"commanding {named or 'all zero'} rad/s for {args.time:.2f} s "
          f"at {args.rate:.0f} Hz ({args.mode}, from {args.start})")

    q0 = ctrl.q.copy()
    ee0 = ctrl.ee_pos.copy()

    if args.view:
        run_with_viewer(ctrl, qd, args)
    else:
        ctrl.set_joint_velocity(qd, warn=False)
        ctrl.run_and_stop(args.time, settle=args.settle)

    print("stopped")

    dq = np.degrees(ctrl.q - q0)
    print("\njoint          commanded    expected     actual      error")
    print("               rad/s        deg          deg         %")
    for i, j in enumerate(ARM_JOINTS):
        want = np.degrees(qd[i] * args.time)
        err = (100 * (dq[i] - want) / want) if abs(want) > 1e-9 else 0.0
        print(f"  {j:<13}{qd[i]:+9.3f}{want:+13.2f}{dq[i]:+11.2f}"
              f"{err:+10.1f}")

    print(f"\nEE moved {100*np.linalg.norm(ctrl.ee_pos - ee0):.2f} cm: "
          f"{np.round(ee0, 4)} -> {np.round(ctrl.ee_pos, 4)}")
    print(f"residual |qd| after stop: {np.abs(ctrl.qd).max():.2e} rad/s")

    if args.csv:
        print(f"wrote {ctrl.trace.to_csv(args.csv)}")
    if args.plot:
        from plots.plot_velocity_tracking import plot_trace
        print(f"wrote {plot_trace(ctrl.trace, args.plot)}")


if __name__ == "__main__":
    main()
