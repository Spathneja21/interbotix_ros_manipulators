#!/usr/bin/env python3
"""Print the real arm's current joint angles, mapped by name.

    python3 read_joints.py            # table with limits and headroom
    python3 read_joints.py --qd       # just the 5 numbers, comma separated
    python3 read_joints.py --watch    # live, updating

Worth having as its own tool rather than reading /locobot/joint_states directly:
that topic leads with left_wheel_joint and right_wheel_joint, so the arm joints
are at indices 2-6. Indexing it positionally silently gives you the wheels.
"""
import argparse
import math
import sys
import time

import rospy
from interbotix_xs_msgs.srv import RobotInfo, RobotInfoRequest
from sensor_msgs.msg import JointState
from ros_check import require_master

NS = "/locobot"
GROUP = "arm"

latest = {}


def on_state(msg):
    latest.update(zip(msg.name, msg.position))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qd", action="store_true",
                    help="print just the 5 values, comma separated "
                         "(paste straight into --qd)")
    ap.add_argument("--deg", action="store_true", help="degrees instead of radians")
    ap.add_argument("--watch", action="store_true", help="keep updating")
    args = ap.parse_args()

    require_master()
    rospy.init_node("uan_read_joints", anonymous=True, disable_signals=True)

    rospy.wait_for_service(f"{NS}/get_robot_info", timeout=10)
    info = rospy.ServiceProxy(f"{NS}/get_robot_info", RobotInfo)(
        RobotInfoRequest(cmd_type="group", name=GROUP))
    joints = list(info.joint_names)

    rospy.Subscriber(f"{NS}/joint_states", JointState, on_state, queue_size=1)
    deadline = time.time() + 5
    while time.time() < deadline and not all(j in latest for j in joints):
        time.sleep(0.05)
    if not all(j in latest for j in joints):
        sys.exit("no /joint_states for the arm joints -- is xs_sdk up?")

    conv = math.degrees if args.deg else (lambda v: v)
    unit = "deg" if args.deg else "rad"

    def show():
        q = [latest[j] for j in joints]
        if args.qd:
            print(",".join(f"{v:.4f}" for v in q))
            return
        print(f"{'joint':<14}{'position':>10}{'lower':>10}{'upper':>10}"
              f"{'headroom -/+':>16}")
        print(f"{'':14}{unit:>10}{unit:>10}{unit:>10}{unit:>16}")
        print("-" * 60)
        for i, j in enumerate(joints):
            lo, hi = info.joint_lower_limits[i], info.joint_upper_limits[i]
            dn, up = q[i] - lo, hi - q[i]
            warn = "  <-- OUTSIDE" if (dn < 0 or up < 0) else ""
            print(f"{j:<14}{conv(q[i]):10.3f}{conv(lo):10.3f}{conv(hi):10.3f}"
                  f"{conv(dn):8.3f}{conv(up):8.3f}{warn}")
        g = latest.get("gripper")
        lf = latest.get("left_finger")
        if g is not None:
            print(f"\ngripper {conv(g):.3f} {unit}   "
                  f"fingers +-{lf:.4f} m" if lf is not None else "")

    if not args.watch:
        show()
        return
    try:
        while not rospy.is_shutdown():
            print("\033[2J\033[H", end="")
            show()
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
