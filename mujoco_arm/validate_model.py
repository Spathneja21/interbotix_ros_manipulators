#!/usr/bin/env python3
"""Checkpoint for phases 1-2: prove the generated WX200 model and its velocity
actuators are sane.

    python3 validate_model.py [--render]

Checks:
  1. model loads, expected joints present in the Interbotix order
  2. joint limits match the URDF
  3. damping / armature seeded
  4. the finger mimic equality constraint actually holds under simulation
  5. EE site position is consistent with the URDF origin chain
  6. self-collision hygiene, swept over 400 random poses
  7. gravity drop from 'sleep' is stable (no NaN, no explosion)
  8. both velocity actuator variants load, are limited from the URDF, and
     behave as phase 2 claims: intvelocity holds at zero command, velocity does
     not
"""
import argparse
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "models", "scene.xml")

ARM_JOINTS = ["waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate"]

# from interbotix_xsarm_descriptions/urdf/wx200.urdf.xacro <limit> tags
EXPECTED_LIMITS = {
    "waist": (-np.pi + 1e-5, np.pi - 1e-5),
    "shoulder": (np.radians(-108), np.radians(113)),
    "elbow": (np.radians(-108), np.radians(93)),
    "wrist_angle": (np.radians(-100), np.radians(123)),
    "wrist_rotate": (-np.pi + 1e-5, np.pi - 1e-5),
}
EXPECTED_EFFORT = {"waist": 8, "shoulder": 18, "elbow": 13,
                   "wrist_angle": 5, "wrist_rotate": 1}

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true",
                    help="also write PNGs of the home and sleep poses")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)

    print("\n1. structure")
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(m.njnt)]
    check("5 arm joints present, in wx200.yaml order",
          names[:5] == ARM_JOINTS, str(names[:5]))
    check("gripper + 2 finger joints present",
          names[5:] == ["gripper", "left_finger", "right_finger"], str(names[5:]))
    check("ee_gripper site exists",
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_gripper") >= 0)
    check("2 keyframes (home, sleep)", m.nkey == 2)

    print("\n2. joint limits vs URDF")
    for j, (lo, hi) in EXPECTED_LIMITS.items():
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        got = m.jnt_range[i]
        check(f"{j} range", np.allclose(got, [lo, hi], atol=1e-4),
              f"got [{got[0]:.4f}, {got[1]:.4f}] want [{lo:.4f}, {hi:.4f}]")

    print("\n3. damping / armature seeded")
    check("all arm joints have non-zero damping",
          all(m.dof_damping[m.jnt_dofadr[mujoco.mj_name2id(
              m, mujoco.mjtObj.mjOBJ_JOINT, j)]] > 0 for j in ARM_JOINTS),
          str(m.dof_damping[:5].round(3)))

    print("\n4. finger mimic equality")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    li = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")]
    ri = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "right_finger")]
    d.qpos[li] = 0.033
    for _ in range(500):
        mujoco.mj_step(m, d)
    check("right_finger tracks -left_finger",
          abs(d.qpos[ri] + d.qpos[li]) < 2e-3,
          f"left={d.qpos[li]:+.4f} right={d.qpos[ri]:+.4f} sum={d.qpos[li]+d.qpos[ri]:+.2e}")

    print("\n5. kinematics")
    mujoco.mj_resetDataKeyframe(m, d, 0)          # home = all arm joints at 0
    mujoco.mj_forward(m, d)
    ee = d.site("ee_gripper").xpos.copy()
    # At the zero pose the upper arm points up (with the 0.05 m elbow offset)
    # and the forearm runs horizontally forward, so the EE is out in front at
    # mid height -- NOT straight up. Expected position is just the sum of the
    # URDF joint origins, which is what these numbers are.
    origins = [(0, 0, 0.072),        # waist
               (0, 0, 0.03865),      # shoulder
               (0.05, 0, 0.2),       # elbow
               (0.2, 0, 0),          # wrist_angle
               (0.065, 0, 0),        # wrist_rotate
               (EE_OFFSET := 0.093575, 0, 0)]     # merged fixed chain to EE
    expect = np.array([sum(o[0] for o in origins), 0.0,
                       sum(o[2] for o in origins)])
    check("EE at home is in the xz plane (no y offset)",
          abs(ee[1]) < 1e-9, f"y={ee[1]:+.2e}")
    check("EE at home matches the URDF origin chain",
          np.allclose(ee, expect, atol=1e-4),
          f"got {ee.round(5)} want {expect.round(5)}")

    mujoco.mj_resetDataKeyframe(m, d, 1)          # sleep
    mujoco.mj_forward(m, d)
    ee_sleep = d.site("ee_gripper").xpos.copy()
    check("EE at sleep pose is low and forward",
          ee_sleep[2] < 0.15 and ee_sleep[0] > 0.05,
          f"pos={ee_sleep.round(4)}")

    print("\n6. self-collision hygiene")

    def contacts_now():
        return {tuple(sorted((
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom1) or "?",
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2) or "?")))
            for i in range(d.ncon)}

    for k, tag in ((0, "home"), (1, "sleep")):
        mujoco.mj_resetDataKeyframe(m, d, k)
        mujoco.mj_forward(m, d)
        c = contacts_now()
        check(f"no contacts in the {tag} pose", not c, str(sorted(c)))

    # A single pose proves little -- sweep the reachable space for adjacent-link
    # pairs that are always in contact and would need their own <exclude>.
    rng = np.random.default_rng(0)
    seen = {}
    lo, hi = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
    N = 400
    for _ in range(N):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[:5] = rng.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        for pair in contacts_now():
            seen[pair] = seen.get(pair, 0) + 1
    always = {p: n for p, n in seen.items() if n > 0.9 * N}
    check(f"no link pair collides in >90% of {N} random poses",
          not always, str(always) if always else "")
    if seen:
        top = sorted(seen.items(), key=lambda kv: -kv[1])[:5]
        print("       occasional contacts (expected, real geometry): " +
              ", ".join(f"{a}/{b}:{n}" for (a, b), n in top))

    print("\n7. stability: 3 s gravity drop from sleep, no actuators")
    mujoco.mj_resetDataKeyframe(m, d, 1)
    qmax = 0.0
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        qmax = max(qmax, float(np.abs(d.qvel).max()))
    finite = bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))
    check("no NaN/inf after 3 s", finite)
    check("velocities stayed bounded", qmax < 50.0, f"max|qvel| = {qmax:.2f} rad/s")
    inrange = all(
        m.jnt_range[i][0] - 0.05 <= d.qpos[m.jnt_qposadr[i]] <= m.jnt_range[i][1] + 0.05
        for i in range(m.njnt) if m.jnt_limited[i])
    check("all joints still inside their limits", inrange, str(d.qpos.round(3)))

    print("\n8. velocity actuators (phase 2)")
    from arm_sim import ArmSim, MODES              # noqa: E402
    for mode in MODES:
        s = ArmSim(mode, bench=True)
        check(f"{mode}: 5 actuators named <joint>_v", s.model.nu == 5,
              f"nu={s.model.nu}")
        check(f"{mode}: ctrlrange is the URDF velocity limit (+-pi)",
              np.allclose(s.model.actuator_ctrlrange[s.aid],
                          [[-np.pi, np.pi]] * 5, atol=1e-5))
        check(f"{mode}: effort limits enforced from the URDF",
              np.allclose(s.model.jnt_actfrcrange[s.jid][:, 1],
                          [8, 18, 13, 5, 1]))
    si = ArmSim("intvelocity", bench=True)
    check("intvelocity: actrange == joint position limits",
          np.allclose(si.model.actuator_actrange[si.aid],
                      si.model.jnt_range[si.jid], atol=1e-4))

    # The behavioural claim the whole phase-2 choice rests on.
    drift = {}
    for mode in MODES:
        s = ArmSim(mode, bench=True).set_qpos([0, 0, 0, 0, 0])
        q0 = s.q.copy()
        for _ in range(int(3.0 / s.dt)):
            s.stop()
            s.step()
        drift[mode] = np.degrees(np.abs(s.q - q0)).max()
        check(f"{mode}: no contact during the hold test", s.data.ncon == 0)
    check("intvelocity HOLDS at zero command (<1.5 deg drift in 3 s)",
          drift["intvelocity"] < 1.5, f"{drift['intvelocity']:.2f} deg")
    check("velocity SINKS at zero command (>10 deg drift in 3 s)",
          drift["velocity"] > 10.0, f"{drift['velocity']:.2f} deg")

    print("\n9. joint-velocity API (phase 3)")
    from velocity_controller import VelocityController      # noqa: E402
    c = VelocityController(mode="intvelocity", bench=True).reset("home")
    check("control tick decimates the physics rate",
          c.steps_per_tick == 10, f"{c.steps_per_tick} physics steps per tick "
          f"({c.rate_hz:.0f} Hz control, {1/c.sim.dt:.0f} Hz physics)")

    c.set_joint("elbow", -0.3, warn=False)
    c.run(2.0)
    qd_ss = c.trace.qd[c.trace.t > 1.0, 2].mean()
    check("elbow tracks a constant command within 2%",
          abs(qd_ss + 0.3) / 0.3 < 0.02, f"commanded -0.300, got {qd_ss:.4f}")

    c.stop()
    c.run(2.0)
    q_a = c.q.copy()
    c.run(1.0)
    check("stays put after the command goes to zero (<0.05 deg/s)",
          np.degrees(np.abs(c.q - q_a)).max() < 0.05,
          f"{np.degrees(np.abs(c.q - q_a)).max():.4f} deg in 1 s")

    over = VelocityController(mode="intvelocity", bench=True)
    over.set_joint_velocity([10, 0, 0, 0, 0], warn=False)
    check("commands beyond the URDF velocity limit are clamped",
          abs(over.cmd[0] - np.pi) < 1e-6, f"10 rad/s -> {over.cmd[0]:.4f}")

    if args.render:
        print("\n10. rendering")
        r = mujoco.Renderer(m, 720, 960)
        for k, tag in ((0, "home"), (1, "sleep")):
            mujoco.mj_resetDataKeyframe(m, d, k)
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera="track")
            out = os.path.join(HERE, f"render_{tag}.png")
            try:
                from PIL import Image
                Image.fromarray(r.render()).save(out)
            except ImportError:
                import imageio.v2 as imageio
                imageio.imwrite(out, r.render())
            print(f"  wrote {out}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*60}\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
