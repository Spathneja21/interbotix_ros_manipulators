#!/usr/bin/env python3
"""Loading, stepping and state access for the WX200 MuJoCo model.

This is the layer everything else sits on. It deliberately does NOT decide
anything about control policy -- it just exposes "write a joint velocity
command, step, read the state back".

    from arm_sim import ArmSim
    sim = ArmSim(mode="intvelocity")
    sim.reset("home")
    sim.set_joint_velocity([0, 0.2, -0.1, 0, 0])
    sim.step(seconds=2.0)
    print(sim.q, sim.qd, sim.ee_pos)

Two actuator modes, see plan.md phase 2:
  intvelocity  integrates the command into a position setpoint, so zero command
               HOLDS the arm against gravity. The default.
  velocity     a pure velocity servo: zero command means zero torque, so the
               arm sinks under gravity at a constant rate. Kept for comparison.
"""
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

# interbotix_xsarm_control/config/wx200.yaml : joint_order (arm group)
ARM_JOINTS = ["waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate"]
NARM = len(ARM_JOINTS)

MODES = ("intvelocity", "velocity")
KEYFRAMES = {"home": 0, "sleep": 1}


class ArmSim:
    def __init__(self, mode="intvelocity", scene=None, bench=False):
        """bench=True loads the floorless testbench.

        Use it for any experiment where the arm is allowed to sink under
        gravity. With the floor present, "the servo could not hold the load"
        turns into "the gripper landed on the ground" and the measurement is
        quietly wrong -- which is exactly what happened the first time we ran
        the phase 2 comparison.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        prefix = "bench" if bench else "scene"
        path = scene or os.path.join(MODELS, f"{prefix}_{mode}.xml")
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)

        self.jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    for j in ARM_JOINTS]
        self.qadr = np.array([self.model.jnt_qposadr[i] for i in self.jid])
        self.vadr = np.array([self.model.jnt_dofadr[i] for i in self.jid])
        self.aid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                      f"{j}_v") for j in ARM_JOINTS]
        if any(a < 0 for a in self.aid):
            raise RuntimeError(f"{path} has no velocity actuators")
        self.ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                    "ee_gripper")

        # URDF <limit velocity="pi"> -- also the actuator ctrlrange
        self.vel_limit = self.model.actuator_ctrlrange[self.aid[0], 1]
        self.reset()

    # -- lifecycle -----------------------------------------------------------

    def reset(self, key="home"):
        """Reset to a named keyframe.

        Always use this rather than mj_resetData: qpos0 puts the fingers at 0,
        which is outside their [0.015, 0.037] limit.
        """
        mujoco.mj_resetDataKeyframe(self.model, self.data, KEYFRAMES[key])
        # An intvelocity actuator's state IS its position setpoint. Seed it from
        # the current joint angles, or the servo drives the arm to act=0 the
        # instant we start stepping.
        self.sync_setpoint()
        mujoco.mj_forward(self.model, self.data)
        return self

    def sync_setpoint(self):
        """Align the integrated setpoints with the measured joint angles."""
        if self.mode == "intvelocity":
            for k, a in enumerate(self.aid):
                self.data.act[self.model.actuator_actadr[a]] = self.q[k]

    def set_qpos(self, q, key="home"):
        """Place the arm at an arbitrary joint configuration."""
        self.reset(key)
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.vadr] = 0.0
        self.sync_setpoint()
        mujoco.mj_forward(self.model, self.data)
        return self

    # -- command -------------------------------------------------------------

    def set_joint_velocity(self, qd):
        """Command joint velocities (rad/s), in wx200.yaml order."""
        qd = np.clip(np.asarray(qd, float), -self.vel_limit, self.vel_limit)
        self.data.ctrl[self.aid] = qd
        return self

    def stop(self):
        return self.set_joint_velocity(np.zeros(NARM))

    # -- stepping ------------------------------------------------------------

    @property
    def dt(self):
        return self.model.opt.timestep

    def step(self, n=1, seconds=None):
        if seconds is not None:
            n = int(round(seconds / self.dt))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        return self

    # -- state ---------------------------------------------------------------

    @property
    def t(self):
        return self.data.time

    @property
    def q(self):
        return self.data.qpos[self.qadr].copy()

    @property
    def qd(self):
        return self.data.qvel[self.vadr].copy()

    @property
    def setpoint(self):
        """The intvelocity servo's integrated position setpoint (None for velocity)."""
        if self.mode != "intvelocity":
            return None
        return np.array([self.data.act[self.model.actuator_actadr[a]]
                         for a in self.aid])

    @property
    def torque(self):
        """Actuator torque actually applied to each arm joint (N m)."""
        return self.data.actuator_force[self.aid].copy()

    @property
    def ee_pos(self):
        """End-effector position in the base_link frame (== world here)."""
        return self.data.site_xpos[self.ee].copy()

    @property
    def ee_vel(self):
        """End-effector linear velocity in the base_link frame (m/s)."""
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee)
        return jacp @ self.data.qvel

    def gravity_torque(self):
        """Torque gravity+Coriolis demands at the current state (N m)."""
        return self.data.qfrc_bias[self.vadr].copy()


if __name__ == "__main__":
    for mode in MODES:
        s = ArmSim(mode).reset("home")
        print(f"{mode:<12} nu={s.model.nu} vel_limit={s.vel_limit:.4f} rad/s "
              f"ee={s.ee_pos.round(4)}")
