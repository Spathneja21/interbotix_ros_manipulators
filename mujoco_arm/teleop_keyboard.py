#!/usr/bin/env python3
"""Keyboard teleop for the simulated WX200, mirroring uan_base_control's
teleop_keyboard.py.

    1/2 : waist         -/+
    3/4 : shoulder      -/+
    5/6 : elbow         -/+
    7/8 : wrist_angle   -/+
    9/0 : wrist_rotate  -/+
    [ ] : smaller / larger velocity step
    x or space : stop all joints
    h : return to the home pose
    q or Ctrl-C : quit

As on the base, velocity is HELD until changed -- releasing the key does not
stop the arm, press x. The command is re-issued every control tick.

Needs a display for the viewer (we have :0) and a real terminal for the key
reads. Run it directly, not through a pipe.
"""
import os
import select
import sys
import termios
import time
import tty

import numpy as np

from arm_sim import ARM_JOINTS
from velocity_controller import VelocityController

STEP_DEFAULT = 0.1          # rad/s per keypress
STEP_MIN, STEP_MAX = 0.01, 1.0

# key -> (joint index, sign)
BINDINGS = {}
for _i, _j in enumerate(ARM_JOINTS):
    _minus, _plus = "13579"[_i], "24680"[_i]
    BINDINGS[_minus] = (_i, -1)
    BINDINGS[_plus] = (_i, +1)


def get_key(settings, timeout=0.0):
    tty.setraw(sys.stdin.fileno())
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if ready else ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


# wrist_angle and wrist_rotate both truncate to "wrist", so label explicitly
SHORT = {"waist": "waist", "shoulder": "shldr", "elbow": "elbow",
         "wrist_angle": "wr_ang", "wrist_rotate": "wr_rot"}


def status(cmd, step, ee):
    parts = " ".join(f"{SHORT[j]}={v:+.2f}" for j, v in zip(ARM_JOINTS, cmd))
    return (f"{parts}  | step {step:.2f}  | ee "
            f"({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f})   ")


def main():
    if not sys.stdin.isatty():
        sys.exit("teleop needs a real terminal (stdin is not a tty)")

    import mujoco.viewer

    settings = termios.tcgetattr(sys.stdin)
    ctrl = VelocityController(mode="intvelocity", rate_hz=50.0)
    ctrl.reset("home")
    step = STEP_DEFAULT

    print(__doc__)
    try:
        with mujoco.viewer.launch_passive(ctrl.sim.model, ctrl.sim.data) as v:
            wall0 = time.time()
            while v.is_running():
                key = get_key(settings)
                if key in BINDINGS:
                    i, sign = BINDINGS[key]
                    qd = ctrl.cmd.copy()
                    qd[i] = np.clip(qd[i] + sign * step,
                                    -ctrl.vel_limit, ctrl.vel_limit)
                    ctrl.set_joint_velocity(qd, warn=False)
                elif key in ("x", " "):
                    ctrl.stop()
                elif key == "[":
                    step = max(STEP_MIN, step / 2)
                elif key == "]":
                    step = min(STEP_MAX, step * 2)
                elif key == "h":
                    ctrl.reset("home", clear_trace=False)
                    wall0 = time.time() - ctrl.sim.t
                elif key in ("q", "\x03"):
                    break

                ctrl.run(1.0 / ctrl.rate_hz, log=False)
                v.sync()
                print("\r" + status(ctrl.cmd, step, ctrl.ee_pos), end="",
                      flush=True)

                lag = ctrl.sim.t - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
    finally:
        ctrl.stop()
        ctrl.run(0.2, log=False)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print("\nstopped")


if __name__ == "__main__":
    main()
