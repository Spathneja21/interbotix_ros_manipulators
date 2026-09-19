#!/usr/bin/env python3
"""Joint-space velocity control for the WX200, plus trace logging.

This sits between a command source (CLI, teleop, or the Cartesian layer coming
in phase 4) and ArmSim's actuators.

    from velocity_controller import VelocityController
    ctrl = VelocityController(mode="intvelocity")
    ctrl.set_joint_velocity([0, 0, -0.3, 0, 0])
    ctrl.run(2.0)
    ctrl.stop(); ctrl.run(0.5)
    ctrl.trace.to_csv("run.csv")

Control rate vs physics rate
----------------------------
The sim steps at 500 Hz (dt = 0.002) but commands are applied on a slower
control tick, 50 Hz by default. That mirrors how the real stack works -- the
base publishes cmd_vel at 20 Hz, an arm node would run at 50-100 Hz -- and it
means anything we build here keeps working when a real command source with real
latency is plugged in. It also gives phase 5's watchdog a natural place to live:
the control tick is where "have we heard from the commander recently?" gets
asked.
"""
import csv
import os

import numpy as np

from arm_sim import ArmSim, ARM_JOINTS, NARM


class Trace:
    """Time series of a run.

    The CSV schema deliberately extends scripts/reach_and_log.py's
    (time, <joints...>, ee_x, ee_y, ee_z) with the velocity columns, so the
    existing plotters stay usable in phase 6.
    """

    def __init__(self):
        self.rows = []

    def add(self, t, q, qd, cmd, ee, tau):
        self.rows.append((t, q.copy(), qd.copy(), cmd.copy(), ee.copy(),
                          tau.copy()))

    def __len__(self):
        return len(self.rows)

    @property
    def t(self):
        return np.array([r[0] for r in self.rows])

    def _col(self, i):
        return np.array([r[i] for r in self.rows])

    @property
    def q(self):
        return self._col(1)

    @property
    def qd(self):
        return self._col(2)

    @property
    def cmd(self):
        return self._col(3)

    @property
    def ee(self):
        return self._col(4)

    @property
    def tau(self):
        return self._col(5)

    def to_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time"] + ARM_JOINTS + ["ee_x", "ee_y", "ee_z"]
                       + [f"qd_cmd_{j}" for j in ARM_JOINTS]
                       + [f"qd_act_{j}" for j in ARM_JOINTS]
                       + [f"tau_{j}" for j in ARM_JOINTS])
            for t, q, qd, cmd, ee, tau in self.rows:
                w.writerow([f"{t:.4f}"] + [f"{v:.6f}" for v in q]
                           + [f"{v:.6f}" for v in ee]
                           + [f"{v:.6f}" for v in cmd]
                           + [f"{v:.6f}" for v in qd]
                           + [f"{v:.6f}" for v in tau])
        return path


class VelocityController:
    def __init__(self, mode="intvelocity", rate_hz=50.0, bench=False, sim=None):
        self.sim = sim or ArmSim(mode, bench=bench)
        self.rate_hz = float(rate_hz)
        self.steps_per_tick = max(1, int(round(1.0 / (self.rate_hz * self.sim.dt))))
        self.cmd = np.zeros(NARM)
        self.trace = Trace()

    # -- lifecycle -----------------------------------------------------------

    def reset(self, key="home", clear_trace=True):
        self.sim.reset(key)
        self.cmd = np.zeros(NARM)
        if clear_trace:
            self.trace = Trace()
        return self

    def set_qpos(self, q, key="home"):
        self.sim.set_qpos(q, key)
        return self

    # -- commands ------------------------------------------------------------

    @property
    def vel_limit(self):
        return self.sim.vel_limit

    def set_joint_velocity(self, qd, warn=True):
        """Command all five joint velocities (rad/s), in wx200.yaml order."""
        qd = np.asarray(qd, dtype=float).reshape(NARM)
        clamped = np.clip(qd, -self.vel_limit, self.vel_limit)
        if warn and not np.allclose(clamped, qd):
            for i in np.nonzero(~np.isclose(clamped, qd))[0]:
                print(f"warning: {ARM_JOINTS[i]} {qd[i]:+.3f} rad/s exceeds the "
                      f"URDF limit {self.vel_limit:.3f}, clamping to "
                      f"{clamped[i]:+.3f}")
        self.cmd = clamped
        return self

    def set_joint(self, name, v, warn=True):
        """Command one joint by name, leaving the others at their current command."""
        if name not in ARM_JOINTS:
            raise ValueError(f"unknown joint {name!r}, expected one of {ARM_JOINTS}")
        qd = self.cmd.copy()
        qd[ARM_JOINTS.index(name)] = v
        return self.set_joint_velocity(qd, warn=warn)

    def stop(self):
        self.cmd = np.zeros(NARM)
        return self

    # -- execution -----------------------------------------------------------

    def _sample(self):
        s = self.sim
        self.trace.add(s.t, s.q, s.qd, self.cmd, s.ee_pos, s.torque)

    def run(self, seconds, profile=None, log=True):
        """Hold (or generate) a command for `seconds` of simulated time.

        profile: optional callable t -> qd, evaluated once per control tick.
        Passing one is how a time-varying command (a ramp, a trajectory, the
        Cartesian layer in phase 4) drives the arm.
        """
        ticks = int(round(seconds * self.rate_hz))
        for _ in range(ticks):
            if profile is not None:
                self.set_joint_velocity(profile(self.sim.t), warn=False)
            # Re-issuing the command every tick mirrors the real stack, where
            # the command has to be republished or the watchdog stops the robot.
            self.sim.set_joint_velocity(self.cmd)
            if log:
                self._sample()
            self.sim.step(self.steps_per_tick)
        if log:
            self._sample()
        return self

    def run_and_stop(self, seconds, settle=0.5, profile=None, log=True):
        """Run, then command zero and keep stepping so the stop is captured."""
        self.run(seconds, profile=profile, log=log)
        self.stop()
        self.run(settle, log=log)
        return self

    # -- state passthrough ---------------------------------------------------

    @property
    def q(self):
        return self.sim.q

    @property
    def qd(self):
        return self.sim.qd

    @property
    def ee_pos(self):
        return self.sim.ee_pos


if __name__ == "__main__":
    c = VelocityController()
    c.set_joint("elbow", -0.3)
    c.run_and_stop(2.0)
    print(f"{len(c.trace)} samples, elbow moved "
          f"{np.degrees(c.trace.q[-1][2] - c.trace.q[0][2]):+.2f} deg, "
          f"final |qd| = {np.abs(c.qd).max():.2e} rad/s")
