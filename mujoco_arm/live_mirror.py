#!/usr/bin/env python3
"""Live mirror: the MuJoCo LoCoBot follows the REAL robot's joint angles.

    python3 live_mirror.py                     # viewer window follows the robot
    python3 live_mirror.py --record run.csv    # ...and record what it saw
    python3 live_mirror.py --no-view           # terminal only (e.g. over SSH)

Start it, then drive the robot however you like in another terminal --
arm_velocity_hw.py, reach_and_log_hw.py, go_to_sleep.py, teleop. It never
sends a command; it only reads /locobot/joint_states and writes those angles
into the model, joint by joint BY NAME.

The model is models/locobot/ (base, plate, camera tower, pan/tilt,
mobile_wx200 arm), whose kinematics match the robot's TF to 0.16 mm. The
viewer draws the end-effector trail, and the terminal shows the EE position in
locobot/arm_base_link and the arm's clearance to the robot body and floor.

A --record file has the same columns as hardware/record_joint_states.py, so it
can be replayed and analysed later with replay_hardware.py.
"""
import argparse
import csv
import os
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rospy
from sensor_msgs.msg import JointState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_hardware import Robot, add_trail            # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hardware"))
from ros_check import require_master           # noqa: E402

TOPIC = "/locobot/joint_states"
STALE = 0.5              # s without a message -> warn
TRAIL_MAX = 400          # trail points kept in the viewer


class Mirror:
    def __init__(self, record=None):
        self.lock = threading.Lock()
        self.latest = None           # (stamp, {name: pos}, {name: vel})
        self.recv_wall = 0.0
        self.count = 0
        self.rows, self.names, self.t0 = [], None, None
        self.record = record
        rospy.Subscriber(TOPIC, JointState, self.on_msg, queue_size=1)

    def on_msg(self, msg):
        pos = dict(zip(msg.name, msg.position))
        vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        stamp = msg.header.stamp.to_sec()
        with self.lock:
            self.latest = (stamp, pos, vel)
            self.recv_wall = time.time()
            self.count += 1
            if self.record:
                if self.names is None:
                    self.names, self.t0 = list(msg.name), stamp
                self.rows.append((stamp - self.t0, pos, vel))

    def snapshot(self):
        with self.lock:
            return self.latest, time.time() - self.recv_wall, self.count

    def save(self, robot, ee_log):
        if not self.record or not self.rows:
            return None
        names = self.names
        with open(self.record, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time"] + names + [f"qd_act_{n}" for n in names]
                       + ["ee_x", "ee_y", "ee_z"])
            for t, pos, vel in self.rows:
                # EE from the model -- matches the robot's TF to sub-mm
                robot.reset()
                robot.set_joints(pos)
                mujoco.mj_kinematics(robot.m, robot.d)
                ee = robot.ee_arm_base()
                w.writerow([f"{t:.4f}"] + [pos.get(n, float("nan")) for n in names]
                           + [vel.get(n, float("nan")) for n in names] + list(ee))
        return len(self.rows)


def status_line(robot, age, count, rate, show_clearance):
    ee = robot.ee_arm_base()
    s = (f"\r{count:7d} msgs {rate:5.1f} Hz | EE (arm_base) "
         f"x {ee[0]:+.3f} y {ee[1]:+.3f} z {ee[2]:+.3f} m")
    if show_clearance:
        c, who = robot.clearance()
        s += (f" | clearance {1000*c:6.1f} mm" +
              (f" ({who[0]}<->{who[1]})" if c < 0.05 else ""))
    if age > STALE:
        s += f" | NO DATA for {age:.1f} s"
    return s + "   "


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", help="also save the received joint states to CSV")
    ap.add_argument("--no-view", action="store_true", help="no window, terminal only")
    ap.add_argument("--no-trail", action="store_true")
    ap.add_argument("--no-clearance", action="store_true",
                    help="skip the clearance computation (about 0.5 ms per update)")
    ap.add_argument("-t", "--time", type=float, default=0.0,
                    help="stop after this many seconds (default: until closed)")
    args = ap.parse_args()

    require_master()
    rospy.init_node("uan_live_mirror", anonymous=True, disable_signals=True)
    robot = Robot()
    mirror = Mirror(args.record)

    print(f"mirroring {TOPIC} into MuJoCo (read-only, sends no commands)")
    wait_end = time.time() + 5
    while mirror.snapshot()[0] is None and time.time() < wait_end:
        time.sleep(0.05)
    if mirror.snapshot()[0] is None:
        sys.exit(f"no messages on {TOPIC} -- is the robot bringup running?")

    trail, ee_log = [], []
    last_count, last_rate_t, rate = 0, time.time(), 0.0
    start = time.time()

    def update():
        nonlocal last_count, last_rate_t, rate
        latest, age, count = mirror.snapshot()
        robot.reset()
        robot.set_joints(latest[1])
        mujoco.mj_forward(robot.m, robot.d)
        if not trail or np.linalg.norm(robot.ee_world() - trail[-1]) > 0.002:
            trail.append(robot.ee_world())
            del trail[:-TRAIL_MAX]
        now = time.time()
        if now - last_rate_t >= 1.0:
            rate = (count - last_count) / (now - last_rate_t)
            last_count, last_rate_t = count, now
        return age, count

    viewer = None
    try:
        if args.no_view:
            while not (args.time and time.time() - start > args.time):
                age, count = update()
                print(status_line(robot, age, count, rate, not args.no_clearance),
                      end="", flush=True)
                time.sleep(0.1)
        else:
            viewer = mujoco.viewer.launch_passive(robot.m, robot.d)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            last_print = 0.0
            while viewer.is_running():
                if args.time and time.time() - start > args.time:
                    break
                with viewer.lock():
                    age, count = update()
                    viewer.user_scn.ngeom = 0
                    if not args.no_trail and len(trail) > 1:
                        add_trail(viewer.user_scn, np.array(trail), (0.1, 0.5, 1, 1))
                viewer.sync()
                if time.time() - last_print > 0.2:
                    print(status_line(robot, age, count, rate, not args.no_clearance),
                          end="", flush=True)
                    last_print = time.time()
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        print()
        n = mirror.save(robot, ee_log)
        if n:
            print(f"wrote {n} samples to {args.record}")


if __name__ == "__main__":
    main()
