#!/usr/bin/env python3
"""Send joint VELOCITY commands to the REAL WX200 arm on the LoCoBot.

This is the hardware counterpart of mujoco_arm/arm_velocity_publisher.py and
takes the same arguments:

    python3 arm_velocity_hw.py --joint elbow --vel -0.2 -t 1.5
    python3 arm_velocity_hw.py --qd 0.2,0,-0.1,0,0 -t 1.5
    python3 arm_velocity_hw.py --joint waist --vel 0.2 -t 1 --dry-run

Deliberately standalone: it imports nothing from mujoco_arm/, so a change or a
broken import on the simulation side can never affect the hardware path.

WHY THERE IS SO MUCH SAFETY CODE HERE
-------------------------------------
In position mode the arm holds still if your program dies. In VELOCITY mode a
Dynamixel keeps turning at the last commanded speed until something stops it --
there is no built-in command timeout. A crashed script, a Ctrl-C at the wrong
moment or a dropped connection therefore means the arm drives itself into its
own hard stops. So this script:

  * caps speed well below the URDF limit (--max-speed, default 0.4 rad/s)
  * caps duration (--max-duration, default 5 s)
  * brakes each joint smoothly to zero as it approaches its position limit,
    using live /locobot/joint_states
  * stops if the joint_states stream goes stale (the robot stopped talking)
  * stops on ANY exit path -- normal end, exception, Ctrl-C, rospy shutdown
  * restores the arm to position mode afterwards, matching the configuration
    interbotix_xslocobot_control/config/modes_all.yaml booted it with

Known rough edge: switching operating mode torques the motors off and on again.
Expect a small settle/sag at the mode switch, especially with the arm extended.
Start from the sleep/rest pose where gravity torque is lowest.
"""
import argparse
import atexit
import sys
import threading
import time

import rospy
from interbotix_xs_msgs.msg import JointGroupCommand
from interbotix_xs_msgs.srv import OperatingModes, OperatingModesRequest
from interbotix_xs_msgs.srv import RobotInfo, RobotInfoRequest
from sensor_msgs.msg import JointState

ROBOT = "locobot"
GROUP = "arm"

# What interbotix_xslocobot_control/config/modes_all.yaml booted the arm with;
# verified live with `rosservice call /locobot/get_robot_info`.
RESTORE = dict(mode="position", profile_type="time",
               profile_velocity=2000, profile_acceleration=300)

DEFAULT_MAX_SPEED = 0.4       # rad/s, vs the 3.14 rad/s the URDF allows
DEFAULT_MAX_DURATION = 5.0    # s
LIMIT_MARGIN = 0.20           # rad; start braking this far from a hard stop
STALE_AFTER = 0.5             # s without joint_states -> stop


class HardwareArm:
    def __init__(self, robot=ROBOT, group=GROUP, rate_hz=50.0,
                 max_speed=DEFAULT_MAX_SPEED, dry_run=False):
        self.ns = f"/{robot}"
        self.group = group
        self.rate_hz = rate_hz
        self.max_speed = max_speed
        self.dry_run = dry_run

        self._lock = threading.Lock()
        self._pos = {}
        self._vel = {}
        self._stamp = None
        self._mode_changed = False
        self._stopped = False

        rospy.loginfo("waiting for %s services...", self.ns)
        rospy.wait_for_service(f"{self.ns}/get_robot_info", timeout=10)
        rospy.wait_for_service(f"{self.ns}/set_operating_modes", timeout=10)
        self._info_srv = rospy.ServiceProxy(f"{self.ns}/get_robot_info", RobotInfo)
        self._mode_srv = rospy.ServiceProxy(f"{self.ns}/set_operating_modes",
                                            OperatingModes)

        info = self._info_srv(RobotInfoRequest(cmd_type="group", name=group))
        self.joints = list(info.joint_names)
        self.lower = list(info.joint_lower_limits)
        self.upper = list(info.joint_upper_limits)
        self.vlimit = list(info.joint_velocity_limits)
        self.start_mode = info.mode
        rospy.loginfo("%s group %s: %s (currently in %s mode)",
                      robot, group, self.joints, self.start_mode)

        self._pub = rospy.Publisher(f"{self.ns}/commands/joint_group",
                                    JointGroupCommand, queue_size=1)
        self._sub = rospy.Subscriber(f"{self.ns}/joint_states", JointState,
                                     self._on_joint_state, queue_size=1)

        # Every exit path must stop the arm, not just the happy one.
        atexit.register(self.shutdown)
        rospy.on_shutdown(self.shutdown)

        self._wait_for_state()

    # -- state ---------------------------------------------------------------

    def _on_joint_state(self, msg):
        with self._lock:
            for n, p in zip(msg.name, msg.position):
                self._pos[n] = p
            for n, v in zip(msg.name, msg.velocity):
                self._vel[n] = v
            self._stamp = rospy.Time.now()

    def _wait_for_state(self, timeout=5.0):
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while rospy.Time.now() < deadline and not rospy.is_shutdown():
            with self._lock:
                if all(j in self._pos for j in self.joints):
                    return
            rospy.sleep(0.05)
        raise RuntimeError("no /joint_states for the arm joints -- is xs_sdk up?")

    def positions(self):
        """Current arm joint positions, mapped BY NAME.

        /locobot/joint_states leads with left_wheel_joint and right_wheel_joint,
        so positional indexing into it silently gives you the wheels.
        """
        with self._lock:
            return [self._pos[j] for j in self.joints]

    def velocities(self):
        """Measured arm joint velocities (rad/s), by name. Zeros if the driver
        is not publishing a velocity field."""
        with self._lock:
            return [self._vel.get(j, 0.0) for j in self.joints]

    def state_age(self):
        with self._lock:
            if self._stamp is None:
                return float("inf")
            return (rospy.Time.now() - self._stamp).to_sec()

    # -- safety --------------------------------------------------------------

    def brake_near_limits(self, qd):
        """Scale each joint's velocity toward zero as it nears a hard stop."""
        out = []
        q = self.positions()
        for i, v in enumerate(qd):
            room = (self.upper[i] - q[i]) if v > 0 else (q[i] - self.lower[i])
            if room < LIMIT_MARGIN:
                scale = max(0.0, room / LIMIT_MARGIN)
                if scale < 0.999:
                    rospy.logwarn_throttle(
                        0.5, "%s is %.3f rad from its limit, scaling %.3f -> %.3f",
                        self.joints[i], max(room, 0.0), v, v * scale)
                v *= scale
            out.append(v)
        return out

    def clamp(self, qd):
        out = []
        for i, v in enumerate(qd):
            cap = min(self.max_speed, self.vlimit[i])
            if abs(v) > cap:
                rospy.logwarn("%s %+.3f rad/s exceeds the cap %.3f, clamping",
                              self.joints[i], v, cap)
                v = cap if v > 0 else -cap
            out.append(v)
        return out

    # -- mode ----------------------------------------------------------------

    def set_velocity_mode(self):
        if self.dry_run:
            rospy.loginfo("[dry-run] would switch %s to velocity mode", self.group)
            return
        rospy.loginfo("switching %s to VELOCITY mode", self.group)
        self._mode_srv(OperatingModesRequest(
            cmd_type="group", name=self.group, mode="velocity",
            profile_type="time", profile_velocity=0, profile_acceleration=0))
        self._mode_changed = True
        # The mode switch torques off and on; command zero straight away so the
        # arm is not left with a stale velocity target.
        self.publish([0.0] * len(self.joints))
        time.sleep(0.2)

    def restore_mode(self):
        if not self._mode_changed or self.dry_run:
            return
        rospy.loginfo("restoring %s to %s mode", self.group, RESTORE["mode"])
        try:
            self._mode_srv(OperatingModesRequest(
                cmd_type="group", name=self.group, **RESTORE))
            self._mode_changed = False
        except Exception as e:            # noqa: BLE001 - never mask a stop
            rospy.logerr("FAILED to restore position mode: %s", e)

    # -- commanding ----------------------------------------------------------

    def publish(self, qd):
        if self.dry_run:
            return
        self._pub.publish(JointGroupCommand(name=self.group,
                                            cmd=[float(v) for v in qd]))

    def stop(self, n=5):
        """Command zero velocity, repeatedly -- one dropped message is enough
        to leave the arm turning.

        Uses time.sleep, not rospy.sleep, on purpose. This runs on the failure
        path, and rospy.sleep blocks or raises once rospy is shutting down or
        the master has gone away -- which is exactly when the stop matters
        most. (Learned the hard way: an earlier version hung here when roscore
        went down mid-command.)
        """
        for _ in range(n):
            try:
                self.publish([0.0] * len(self.joints))
            except Exception as e:        # noqa: BLE001 - keep trying to stop
                rospy.logerr("stop publish failed: %s", e)
            time.sleep(0.02)

    def shutdown(self):
        if self._stopped:
            return
        self._stopped = True
        try:
            self.stop()
        finally:
            self.restore_mode()

    def run(self, qd, duration):
        qd = self.clamp(qd)
        rate = rospy.Rate(self.rate_hz)
        end = rospy.Time.now() + rospy.Duration(duration)
        while not rospy.is_shutdown() and rospy.Time.now() < end:
            age = self.state_age()
            if age > STALE_AFTER:
                rospy.logerr("joint_states stale by %.2f s -- stopping", age)
                break
            cmd = self.brake_near_limits(qd)
            if self.dry_run:
                rospy.loginfo_throttle(
                    0.25, "[dry-run] would send %s",
                    " ".join(f"{n}={v:+.3f}" for n, v in zip(self.joints, cmd)))
            self.publish(cmd)
            rate.sleep()
        self.stop()


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--joint", help="move a single joint by name")
    g.add_argument("--qd", help="all five joint velocities, comma separated")
    p.add_argument("--vel", type=float, default=0.0, help="rad/s, with --joint")
    p.add_argument("-t", "--time", type=float, default=1.5,
                   help="how long to move, seconds (default 1.5)")
    p.add_argument("-r", "--rate", type=float, default=50.0,
                   help="publish rate in Hz (default 50)")
    p.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED,
                   help=f"per-joint speed cap (default {DEFAULT_MAX_SPEED})")
    p.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION,
                   help=f"hard duration cap (default {DEFAULT_MAX_DURATION})")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be sent; never changes mode or moves")
    return p.parse_args(argv[1:])


def main():
    args = parse_args(sys.argv)
    if args.time > args.max_duration:
        sys.exit(f"-t {args.time} exceeds --max-duration {args.max_duration}")

    rospy.init_node("uan_arm_velocity_hw", anonymous=True)
    arm = HardwareArm(rate_hz=args.rate, max_speed=args.max_speed,
                      dry_run=args.dry_run)

    if args.joint:
        if args.joint not in arm.joints:
            sys.exit(f"unknown joint {args.joint!r}, expected one of {arm.joints}")
        qd = [0.0] * len(arm.joints)
        qd[arm.joints.index(args.joint)] = args.vel
    else:
        parts = [s for s in args.qd.replace(" ", "").split(",") if s]
        if len(parts) != len(arm.joints):
            sys.exit(f"--qd needs {len(arm.joints)} values ({arm.joints})")
        qd = [float(s) for s in parts]

    before = arm.positions()
    named = ", ".join(f"{n}={v:+.3f}" for n, v in zip(arm.joints, qd) if v)
    rospy.loginfo("%scommanding %s rad/s for %.2f s",
                  "[dry-run] " if args.dry_run else "",
                  named or "all zero", args.time)
    rospy.loginfo("start pose: %s",
                  " ".join(f"{v:+.3f}" for v in before))

    try:
        arm.set_velocity_mode()
        arm.run(qd, args.time)
    finally:
        arm.shutdown()

    after = arm.positions()
    rospy.loginfo("end pose:   %s", " ".join(f"{v:+.3f}" for v in after))
    rospy.loginfo("moved:      %s",
                  " ".join(f"{b - a:+.3f}" for a, b in zip(before, after)))


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
