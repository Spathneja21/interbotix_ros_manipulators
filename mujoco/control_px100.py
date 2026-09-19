#!/usr/bin/env python3
"""Interactive control of the px100 arm in MuJoCo.

Joint-space commands (radians, ranges from px100_mjcf.xml):
    waist 0.5
    shoulder -0.3
    elbow 1.0
    wrist_angle 0.2
    gripper open
    gripper close
    home

Cartesian command — reach an (x, y, z) end-effector target, in metres, in the
px100/base_link frame, over the given duration in seconds:
    goto 0.15 0.0 0.15 3.0

    waist         -3.14 .. 3.14
    shoulder      -1.94 .. 1.87
    elbow         -2.11 .. 1.61
    wrist_angle   -1.75 .. 2.15

quit to exit.
"""
import threading
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("px100_mjcf.xml")
data = mujoco.MjData(model)

ARM_JOINTS = ["waist", "shoulder", "elbow", "wrist_angle"]
actuator_id = {name: model.actuator(name).id for name in ARM_JOINTS}
actuator_id["left_finger"] = model.actuator("left_finger").id
actuator_id["right_finger"] = model.actuator("right_finger").id
arm_qpos_adr = [model.joint(name).qposadr[0] for name in ARM_JOINTS]
arm_dof_adr = [model.joint(name).dofadr[0] for name in ARM_JOINTS]
ee_site_id = model.site("ee_site").id

GRIPPER_OPEN = (0.037, -0.037)
GRIPPER_CLOSE = (0.015, -0.015)

lock = threading.Lock()
motion = None  # (start_q: dict, target_q: dict, start_time, duration) or None


def set_gripper(open_: bool):
    l, r = GRIPPER_OPEN if open_ else GRIPPER_CLOSE
    data.ctrl[actuator_id["left_finger"]] = l
    data.ctrl[actuator_id["right_finger"]] = r


def solve_ik(target, iters=200, tol=1e-4, damping=1e-3):
    """Damped least-squares IK over the 4 arm joints, via the site Jacobian.

    Solves on a scratch copy of qpos (doesn't disturb the live sim while the
    command thread is computing this — the physics thread is still stepping
    with the *previous* ctrl in the meantime).
    """
    qpos = data.qpos.copy()
    target = np.array(target, dtype=float)
    jacp = np.zeros((3, model.nv))
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = qpos
    for _ in range(iters):
        mujoco.mj_forward(model, scratch)
        err = target - scratch.site_xpos[ee_site_id]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(model, scratch, jacp, None, ee_site_id)
        J = jacp[:, arm_dof_adr]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        for i, adr in enumerate(arm_qpos_adr):
            scratch.qpos[adr] = np.clip(
                scratch.qpos[adr] + dq[i],
                model.jnt_range[model.joint(ARM_JOINTS[i]).id][0],
                model.jnt_range[model.joint(ARM_JOINTS[i]).id][1],
            )
    reached = np.linalg.norm(err) < 0.01
    return {name: scratch.qpos[adr] for name, adr in zip(ARM_JOINTS, arm_qpos_adr)}, reached, err


def command_loop():
    global motion
    print(__doc__)
    while True:
        try:
            line = input("px100> ").strip().split()
        except EOFError:
            break
        if not line:
            continue
        cmd = line[0]
        if cmd == "quit":
            break
        elif cmd == "home":
            with lock:
                motion = None
            for name in ARM_JOINTS:
                data.ctrl[actuator_id[name]] = 0.0
        elif cmd == "gripper" and len(line) == 2 and line[1] in ("open", "close"):
            set_gripper(line[1] == "open")
        elif cmd in ARM_JOINTS and len(line) == 2:
            with lock:
                motion = None
            try:
                data.ctrl[actuator_id[cmd]] = float(line[1])
            except ValueError:
                print(f"not a number: {line[1]}")
        elif cmd == "goto" and len(line) == 5:
            try:
                x, y, z, duration = (float(v) for v in line[1:])
            except ValueError:
                print("usage: goto x y z duration_s")
                continue
            target_q, reached, err = solve_ik((x, y, z))
            if not reached:
                print(f"IK did not converge cleanly (residual {np.linalg.norm(err)*1000:.1f} mm) "
                      f"-- likely out of reach or near a singularity, moving to closest solution anyway")
            start_q = {name: data.ctrl[actuator_id[name]] for name in ARM_JOINTS}
            with lock:
                motion = (start_q, target_q, data.time, max(duration, 1e-3))
            print(f"checkpoint: IK solution {target_q}")
        else:
            print(f"unrecognized command: {' '.join(line)}")


with mujoco.viewer.launch_passive(model, data) as viewer:
    t = threading.Thread(target=command_loop, daemon=True)
    t.start()
    while viewer.is_running() and t.is_alive():
        with lock:
            if motion is not None:
                start_q, target_q, start_time, duration = motion
                alpha = min(1.0, (data.time - start_time) / duration)
                for name in ARM_JOINTS:
                    data.ctrl[actuator_id[name]] = start_q[name] + alpha * (target_q[name] - start_q[name])
                if alpha >= 1.0:
                    motion = None
        mujoco.mj_step(model, data)
        viewer.sync()
