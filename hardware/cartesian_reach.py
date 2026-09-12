#!/usr/bin/env python3
"""Closed-loop Cartesian velocity servo: drive the end effector to an (x, y, z)
target by commanding JOINT VELOCITIES.

Pure computation -- no ROS, no hardware. Used by reach_and_log_hw.py on the real
arm and by the sim validation in test_reach_sim.py, so both run identical
control law.

    reach = CartesianReach()
    qd = reach.step(q_measured, target_xyz)

The 5-DOF problem
-----------------
The WX200 has five joints, so the Cartesian Jacobian is 6x5 and an arbitrary
6-D twist is NOT achievable. This solves the POSITION-only task: the 3x5
translational Jacobian, leaving orientation to fall where it may. That is the
right call for reaching a point, and it is what plan.md phase 4 specifies as
the first step.

Damped least squares (Levenberg-Marquardt) throughout:

    qd = J^T (J J^T + lambda^2 I)^-1 v

The undamped pseudo-inverse blows up near this arm's singularities (full
extension, and directly overhead), which are easy to wander into while
servoing. Damping trades a little tracking accuracy for not producing 50 rad/s
joint commands.

Frames
------
Kinematics come from the MuJoCo model in ../mujoco_arm, which is built from
wx200.urdf.xacro (the STANDALONE arm). The real LoCoBot runs mobile_wx200,
whose waist sits at z = 0.066175 instead of 0.072 -- a 5.825 mm base offset,
measured live against the robot's own TF and confirmed in the two xacro files.
Everything above the waist is identical (x and y agree to 1e-5 m).

So: positions from this module are reported in the ROBOT's arm_base_link frame,
with Z_OFFSET applied. The Jacobian needs no correction -- a constant
translation of the whole arm does not change it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "mujoco_arm"))
import mujoco                                        # noqa: E402
from arm_sim import ARM_JOINTS, NARM                 # noqa: E402

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "mujoco_arm", "models", "wx200.xml")

# standalone wx200 waist z (0.072) - mobile_wx200 waist z (0.066175)
Z_OFFSET = -0.005825


class CartesianReach:
    def __init__(self, gain=1.5, v_max=0.06, qd_max=0.4, damping=0.08,
                 tol=0.005, limit_margin=0.20, lower=None, upper=None):
        """
        gain     : proportional gain, EE velocity per metre of error (1/s)
        v_max    : cap on commanded EE speed (m/s) -- keep this small on hardware
        qd_max   : cap on any single joint velocity (rad/s)
        damping  : lambda in the damped least squares solve
        tol      : distance at which the target counts as reached (m)
        """
        self.model = mujoco.MjModel.from_xml_path(MODEL)
        self.data = mujoco.MjData(self.model)
        self.jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    for j in ARM_JOINTS]
        self.qadr = np.array([self.model.jnt_qposadr[i] for i in self.jid])
        self.vadr = np.array([self.model.jnt_dofadr[i] for i in self.jid])
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                      "ee_gripper")

        self.gain, self.v_max, self.qd_max = gain, v_max, qd_max
        self.damping, self.tol = damping, tol
        self.limit_margin = limit_margin
        self.lower = np.array(lower if lower is not None
                              else self.model.jnt_range[self.jid, 0])
        self.upper = np.array(upper if upper is not None
                              else self.model.jnt_range[self.jid, 1])

    # -- kinematics ----------------------------------------------------------

    def _sync(self, q):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[self.qadr] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def fk(self, q):
        """EE position for joint angles q, in the ROBOT's arm_base_link frame."""
        self._sync(q)
        p = self.data.site_xpos[self.site].copy()
        p[2] += Z_OFFSET
        return p

    def jacobian(self, q):
        """3x5 translational Jacobian at the EE, arm joints only."""
        self._sync(q)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site)
        return jacp[:, self.vadr]

    def manipulability(self, q):
        """Smallest singular value of J -- how close we are to a singularity."""
        return float(np.linalg.svd(self.jacobian(q), compute_uv=False).min())

    # -- control -------------------------------------------------------------

    def step(self, q, target):
        """One control tick. Returns (qd, error_vector, distance)."""
        q = np.asarray(q, float)
        ee = self.fk(q)
        err = np.asarray(target, float) - ee
        dist = float(np.linalg.norm(err))

        # proportional EE velocity, capped
        v = self.gain * err
        speed = np.linalg.norm(v)
        if speed > self.v_max:
            v *= self.v_max / speed

        J = self.jacobian(q)
        # damped least squares: qd = J^T (J J^T + lam^2 I)^-1 v
        lam2 = self.damping ** 2
        qd = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), v)

        qd = self.brake_near_limits(q, qd)

        # cap AFTER braking, preserving direction, so one hot joint does not
        # silently distort the Cartesian direction
        peak = np.abs(qd).max()
        if peak > self.qd_max:
            qd *= self.qd_max / peak

        return qd, err, dist

    def brake_near_limits(self, q, qd):
        out = np.array(qd, float)
        for i in range(NARM):
            room = (self.upper[i] - q[i]) if out[i] > 0 else (q[i] - self.lower[i])
            if room < self.limit_margin:
                out[i] *= max(0.0, room / self.limit_margin)
        return out

    def reachable(self, target, q_seed=None, iters=400):
        """Offline check: can the servo actually get there from q_seed?

        Runs the same control law against pure kinematics (no dynamics), which
        is a fast way to reject a target before pointing the real arm at it.
        """
        q = np.array(q_seed if q_seed is not None else np.zeros(NARM), float)
        dt = 0.02
        for _ in range(iters):
            qd, _, dist = self.step(q, target)
            if dist < self.tol:
                return True, q, dist
            q = np.clip(q + qd * dt, self.lower, self.upper)
        return False, q, dist
