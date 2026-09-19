#!/usr/bin/env python3
"""Reach an (x, y, z) target on the REAL arm using joint velocity commands, log
the whole trajectory to CSV, and report the final position error.

    python3 reach_and_log_hw.py 0.35 0 0.20                     # dry-run FIRST
    python3 reach_and_log_hw.py 0.35 0 0.20 --go
    python3 reach_and_log_hw.py 0.35 0 0.20 --go --csv reach.csv

Target is in metres in the robot's locobot/arm_base_link frame. Nothing moves
without --go.

How the error is measured
-------------------------
Two independent readings of where the end effector ended up:

  fk_*  our MuJoCo model's forward kinematics from the measured joint angles.
        This is what the controller servos on, so comparing the target against
        it only tells you the servo converged -- it cannot reveal a model error.

  ee_*  the ROBOT's own TF (locobot/arm_base_link -> locobot/ee_gripper_link),
        computed by robot_state_publisher from the robot's URDF. Independent of
        our model, so this is the number the error report uses.

Neither is an external measurement of where the gripper physically is. The arm
has no EE position sensor -- both are forward kinematics from joint encoders.
So the reported error is servo + model error, NOT absolute positioning
accuracy. Encoder-based FK cannot see kinematic calibration error, link flex,
gear backlash or a bent finger. Getting that would need something external
(the RealSense looking at an AR tag on the gripper, for instance).

CSV columns match scripts/reach_and_log.py (time, <joints>, ee_x, ee_y, ee_z)
so plot_reach_trajectory.py works on it unchanged, plus the velocity, model-FK
and distance columns.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import rospy
import tf2_ros

from arm_velocity_hw import HardwareArm
from cartesian_reach import CartesianReach, Z_OFFSET
from ros_check import require_master

BASE_FRAME = "locobot/arm_base_link"
EE_FRAME = "locobot/ee_gripper_link"

DEFAULT_TIMEOUT = 20.0
SIGMA_MIN = 0.02          # abort if the Jacobian gets this close to singular
STALL_EPS = 0.0005        # improvement smaller than this does not count (m)
STALL_TIME = 2.5          # no improvement for this long -> stop, do not wait out
                          # the timeout


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.add_argument("z", type=float)
    p.add_argument("--go", action="store_true",
                   help="actually move. Without it, plans and reports only.")
    p.add_argument("--csv", default="reach_trajectory.csv")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("-r", "--rate", type=float, default=50.0)
    p.add_argument("--v-max", type=float, default=0.06,
                   help="cap on EE speed, m/s (default 0.06)")
    p.add_argument("--qd-max", type=float, default=0.4,
                   help="cap on any joint velocity, rad/s (default 0.4)")
    p.add_argument("--tol", type=float, default=0.005,
                   help="stop when this close to the target, m (default 0.005)")
    return p.parse_args(argv[1:])


def main():
    args = parse_args(sys.argv)
    target = np.array([args.x, args.y, args.z])

    require_master()
    rospy.init_node("uan_reach_and_log_hw", anonymous=True)

    arm = HardwareArm(rate_hz=args.rate, max_speed=args.qd_max,
                      dry_run=not args.go)
    reach = CartesianReach(v_max=args.v_max, qd_max=args.qd_max, tol=args.tol,
                           lower=arm.lower, upper=arm.upper)

    tf_buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buf)
    rospy.sleep(1.5)

    def tf_ee():
        try:
            t = tf_buf.lookup_transform(BASE_FRAME, EE_FRAME, rospy.Time(0),
                                        rospy.Duration(0.2)).transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception:                       # noqa: BLE001
            return np.array([np.nan] * 3)

    q0 = np.array(arm.positions())
    fk0, ee0 = reach.fk(q0), tf_ee()

    print(f"\ntarget        {target.round(4)}  ({BASE_FRAME})")
    print(f"start  fk     {fk0.round(4)}")
    print(f"start  tf     {ee0.round(4)}   (model vs robot TF differ by "
          f"{1000*np.linalg.norm(fk0-ee0):.2f} mm)")
    print(f"distance      {np.linalg.norm(target - fk0):.4f} m")

    ok, q_pred, d_pred = reach.reachable(target, q_seed=q0)
    print(f"offline check {'REACHABLE' if ok else 'NOT REACHABLE'} "
          f"(predicted residual {d_pred*1000:.1f} mm, "
          f"joints {np.degrees(q_pred).round(1)} deg)")
    if not ok:
        sys.exit("target not reachable from here -- refusing to move")
    sigma0 = reach.manipulability(q_pred)
    print(f"min sigma at goal {sigma0:.4f}")
    if sigma0 < SIGMA_MIN:
        sys.exit("goal pose is near-singular -- refusing to move")

    if not args.go:
        print("\n[dry-run] pass --go to actually move")
        return

    rows = []
    t0 = time.time()
    status = "timeout"
    q_servo = ee_servo = None
    best, best_t = np.inf, 0.0
    try:
        arm.set_velocity_mode()
        rate = rospy.Rate(args.rate)
        while not rospy.is_shutdown():
            el = time.time() - t0
            if el > args.timeout:
                break
            q = np.array(arm.positions())
            qd, err, dist = reach.step(q, target)
            sigma = reach.manipulability(q)

            rows.append([el, *q, *tf_ee(), *reach.fk(q), *qd,
                         *arm.velocities(), dist, sigma])

            if dist < best - STALL_EPS:
                best, best_t = dist, el
            if dist < args.tol:
                status = "reached"
                break
            # A proportional servo against a velocity-mode Dynamixel settles at
            # a non-zero residual: near the target the commanded velocities fall
            # below the joints' friction deadband and nothing moves. Detect that
            # instead of burning the whole timeout on it.
            if el - best_t > STALL_TIME:
                status = f"stalled at {best*1000:.1f} mm"
                break
            if sigma < SIGMA_MIN:
                status = "aborted: near-singular"
                rospy.logerr("sigma %.4f below %.4f -- stopping", sigma, SIGMA_MIN)
                break
            if arm.state_age() > 0.5:
                status = "aborted: stale joint_states"
                break

            arm.publish(arm.brake_near_limits(qd))
            rate.sleep()

        # Measure BEFORE shutdown. Restoring position mode torques the motors
        # off and on, and the arm sags several centimetres on the way -- so a
        # reading taken afterwards reports the sag, not what the servo achieved.
        arm.stop()
        rospy.sleep(0.3)
        q_servo = np.array(arm.positions())
        ee_servo = tf_ee()
    finally:
        arm.shutdown()

    rospy.sleep(0.8)
    q1 = np.array(arm.positions())
    fk1, ee1 = reach.fk(q1), tf_ee()
    if q_servo is None:
        q_servo, ee_servo = q1, ee1

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + list(arm.joints)
                   + ["ee_x", "ee_y", "ee_z"]
                   + ["fk_x", "fk_y", "fk_z"]
                   + [f"qd_cmd_{j}" for j in arm.joints]
                   + [f"qd_act_{j}" for j in arm.joints]
                   + ["dist", "sigma"])
        w.writerows(rows)

    e_servo = target - ee_servo
    e_rest = target - ee1
    sag = ee_servo - ee1

    print(f"\n{'='*72}\nresult: {status} after {time.time()-t0:.2f} s, "
          f"{len(rows)} samples -> {args.csv}\n{'='*72}")
    print(f"{'':22}{'x':>10}{'y':>10}{'z':>10}{'|error|':>12}")
    print(f"target                {target[0]:10.4f}{target[1]:10.4f}"
          f"{target[2]:10.4f}")
    print(f"end of servo (tf)     {ee_servo[0]:10.4f}{ee_servo[1]:10.4f}"
          f"{ee_servo[2]:10.4f}{1000*np.linalg.norm(e_servo):9.2f} mm"
          f"   <-- what the controller achieved")
    print(f"after mode restore    {ee1[0]:10.4f}{ee1[1]:10.4f}{ee1[2]:10.4f}"
          f"{1000*np.linalg.norm(e_rest):9.2f} mm   <-- where it ends up")
    print(f"\nservo error vector:   {np.round(e_servo, 5)} m")
    print(f"sag on mode restore:  {np.round(sag, 5)} m "
          f"({1000*np.linalg.norm(sag):.1f} mm)")
    if np.linalg.norm(sag) > 0.005:
        print("  ^ switching back to position mode torques the motors off and on;\n"
              "    the arm drops before the position loop catches it. Not servo error.")
    print(f"\nmodel vs robot TF:    {1000*np.linalg.norm(fk1-ee1):.2f} mm "
          f"(expected ~0 after the {abs(Z_OFFSET)*1000:.3f} mm Z_OFFSET correction)")
    print(f"joints at end of servo (deg): {np.degrees(q_servo-q0).round(2)}")
    print(f"joints after restore   (deg): {np.degrees(q1-q0).round(2)}")
    print("\nBoth readings are forward kinematics from joint encoders, so this "
          "is\nservo+model error, not absolute positioning accuracy.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
