#!/usr/bin/env python3
"""Return the real arm to its rest (sleep) pose.

    python3 go_to_sleep.py                # show the plan, move nothing
    python3 go_to_sleep.py --go
    python3 go_to_sleep.py --go --via-home
    python3 go_to_sleep.py --go --home     # go to the straight-up home pose instead

The sleep angles are read from the robot itself via /locobot/get_robot_info, so
this uses the INTEGRATED locobot_wx200 configuration, not the standalone arm's.
The two differ:

    locobot_wx200 (this robot) : [0, -1.30, 1.55, 0.70, 0]
    wx200 standalone           : [0, -1.88, 1.50, 0.80, 0]

Uses POSITION mode, deliberately. Everything else in this folder commands
velocities, but a joint-space target is the one case where position mode is
plainly better: 0.088 deg resolution and no velocity deadband, versus the
0.229 rev/min velocity quantum that makes velocity mode stick-slip below about
0.024 rad/s (see plot_jerk_analysis.py). The arm also holds the pose afterwards
instead of sagging.
"""
import argparse
import sys
import time

import numpy as np
import rospy
from interbotix_xs_msgs.msg import JointGroupCommand
from interbotix_xs_msgs.srv import (OperatingModes, OperatingModesRequest,
                                    RobotInfo, RobotInfoRequest,
                                    TorqueEnable, TorqueEnableRequest)
from sensor_msgs.msg import JointState
from ros_check import require_master

NS = "/locobot"
GROUP = "arm"

# interbotix_xslocobot_control/config/modes_all.yaml
POSITION_MODE = dict(mode="position", profile_type="time",
                     profile_velocity=2000, profile_acceleration=300)
MOVE_TIME = 2.0          # profile_velocity is 2000 ms
SETTLE = 1.0
TOL = 0.05               # rad, arrival tolerance


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="actually move")
    ap.add_argument("--via-home", action="store_true",
                    help="stop at the straight-up home pose on the way; safer "
                         "if the arm is extended or near an obstacle")
    ap.add_argument("--home", action="store_true",
                    help="go to home (all joints zero) instead of sleep")
    ap.add_argument("--torque-off", action="store_true",
                    help="release the motors once resting. The sleep pose is "
                         "mechanically supported, but the arm WILL settle.")
    args = ap.parse_args()

    require_master()
    rospy.init_node("uan_go_to_sleep", anonymous=True)

    latest = {}
    rospy.Subscriber(f"{NS}/joint_states", JointState,
                     lambda m: latest.update(zip(m.name, m.position)),
                     queue_size=1)
    rospy.wait_for_service(f"{NS}/get_robot_info", timeout=10)
    info = rospy.ServiceProxy(f"{NS}/get_robot_info", RobotInfo)(
        RobotInfoRequest(cmd_type="group", name=GROUP))
    joints = list(info.joint_names)
    sleep_q = np.array(info.joint_sleep_positions)

    deadline = time.time() + 5
    while time.time() < deadline and not all(j in latest for j in joints):
        time.sleep(0.05)
    if not all(j in latest for j in joints):
        sys.exit("no /joint_states -- is xs_sdk up?")
    q0 = np.array([latest[j] for j in joints])

    goal = np.zeros(len(joints)) if args.home else sleep_q
    label = "home" if args.home else "sleep"

    print(f"\ncurrent mode  : {info.mode} ({info.profile_type} profile)")
    print(f"{'joint':<14}{'current':>10}{'target':>10}{'move':>10}   (deg)")
    print("-" * 48)
    for i, j in enumerate(joints):
        print(f"{j:<14}{np.degrees(q0[i]):10.2f}{np.degrees(goal[i]):10.2f}"
              f"{np.degrees(goal[i]-q0[i]):10.2f}")
    print(f"\ntarget is the {label} pose from the robot's own config "
          f"(locobot_wx200)")

    if not args.go:
        print("\n[dry-run] pass --go to actually move")
        return

    pub = rospy.Publisher(f"{NS}/commands/joint_group", JointGroupCommand,
                          queue_size=1)
    mode_srv = rospy.ServiceProxy(f"{NS}/set_operating_modes", OperatingModes)
    rospy.sleep(0.5)

    # Anything that crashed mid-velocity-command could have left the group in
    # velocity mode; a position command would then be read as a velocity.
    if info.mode != "position":
        print(f"arm is in {info.mode} mode -- switching to position first")
        mode_srv(OperatingModesRequest(cmd_type="group", name=GROUP,
                                       **POSITION_MODE))
        rospy.sleep(0.5)

    waypoints = []
    if args.via_home and not args.home:
        waypoints.append(("home", np.zeros(len(joints))))
    waypoints.append((label, goal))

    for name, wp in waypoints:
        print(f"moving to {name} ...")
        pub.publish(JointGroupCommand(name=GROUP, cmd=[float(v) for v in wp]))
        t_end = time.time() + MOVE_TIME + SETTLE
        while time.time() < t_end and not rospy.is_shutdown():
            time.sleep(0.05)
        q = np.array([latest[j] for j in joints])
        err = np.abs(q - wp)
        print(f"  arrived within {np.degrees(err).max():.2f} deg "
              f"({'ok' if err.max() < TOL else 'NOT within tolerance'})")

    q1 = np.array([latest[j] for j in joints])
    print(f"\n{'joint':<14}{'final':>10}{'target':>10}{'error':>10}   (deg)")
    print("-" * 48)
    for i, j in enumerate(joints):
        print(f"{j:<14}{np.degrees(q1[i]):10.2f}{np.degrees(goal[i]):10.2f}"
              f"{np.degrees(q1[i]-goal[i]):10.2f}")

    if args.torque_off:
        print("\nreleasing torque -- the arm will settle onto its rest stops")
        rospy.ServiceProxy(f"{NS}/torque_enable", TorqueEnable)(
            TorqueEnableRequest(cmd_type="group", name=GROUP, enable=False))
    else:
        print("\ntorque still on, arm holding the pose "
              "(--torque-off to release it)")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
