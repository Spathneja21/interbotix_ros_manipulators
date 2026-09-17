#!/usr/bin/env python3
"""Record the real robot's joint states to CSV, for replay in MuJoCo.

    python3 record_joint_states.py run.csv            # Ctrl-C to stop
    python3 record_joint_states.py run.csv -t 15      # stop after 15 s

Moves nothing. Run it in one terminal while the arm is driven any other way --
reach_and_log_hw.py, arm_velocity_hw.py, go_to_sleep.py, teleop -- then replay
with ../mujoco_arm/replay_hardware.py.

Records EVERY joint in /locobot/joint_states (wheels, arm, gripper, fingers,
pan, tilt), one row per message, plus the end-effector position from the
robot's own TF in locobot/arm_base_link. Columns are named by joint, which is
also how the replay maps them onto the MuJoCo model, so the order of the topic
never matters.
"""
import argparse
import csv
import signal
import sys
import time

import rospy
import tf2_ros
from sensor_msgs.msg import JointState
from ros_check import require_master

BASE, EE = "locobot/arm_base_link", "locobot/ee_gripper_link"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("-t", "--time", type=float, default=0.0,
                    help="stop after this many seconds (default: until Ctrl-C)")
    args = ap.parse_args()

    require_master()
    rospy.init_node("uan_record_joint_states", anonymous=True, disable_signals=True)
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf)

    state = dict(names=None, rows=[], t0=None, stop=False)

    def on_msg(msg):
        if state["stop"]:
            return
        if state["names"] is None:
            state["names"] = list(msg.name)
            state["t0"] = msg.header.stamp.to_sec()
        pos = dict(zip(msg.name, msg.position))
        vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        try:
            p = buf.lookup_transform(BASE, EE, rospy.Time(0)).transform.translation
            ee = [p.x, p.y, p.z]
        except Exception:                      # noqa: BLE001
            ee = [float("nan")] * 3
        state["rows"].append([msg.header.stamp.to_sec() - state["t0"]]
                             + [pos.get(n, float("nan")) for n in state["names"]]
                             + [vel.get(n, float("nan")) for n in state["names"]]
                             + ee)

    rospy.Subscriber("/locobot/joint_states", JointState, on_msg, queue_size=50)
    signal.signal(signal.SIGINT, lambda *_: state.update(stop=True))

    print(f"recording /locobot/joint_states -> {args.csv}  (Ctrl-C to stop)")
    start = time.time()
    while not state["stop"]:
        if args.time and time.time() - start > args.time:
            break
        time.sleep(0.1)
        n = len(state["rows"])
        print(f"\r  {n} samples, {time.time()-start:5.1f} s", end="", flush=True)
    state["stop"] = True
    print()

    if not state["rows"]:
        sys.exit("no joint_states received -- is xs_sdk up?")
    names = state["names"]
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + names + [f"qd_act_{n}" for n in names]
                   + ["ee_x", "ee_y", "ee_z"])
        w.writerows(state["rows"])
    dur = state["rows"][-1][0]
    print(f"wrote {len(state['rows'])} samples over {dur:.2f} s "
          f"({len(state['rows'])/max(dur, 1e-9):.0f} Hz) to {args.csv}")


if __name__ == "__main__":
    main()
