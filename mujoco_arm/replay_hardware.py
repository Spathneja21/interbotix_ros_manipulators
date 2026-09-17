#!/usr/bin/env python3
"""Replay a trajectory recorded on the REAL LoCoBot in MuJoCo, and measure it.

    python3 replay_hardware.py ../hardware/reach3.csv              # analyse + plot
    python3 replay_hardware.py ../hardware/reach3.csv --view       # watch it
    python3 replay_hardware.py ../hardware/reach3.csv --gif reach3.gif
    python3 replay_hardware.py ../hardware/reach3.csv --dynamic velocity

Accepts any CSV with a `time` column and columns named after joints: the reach
logs from reach_and_log_hw.py, or a full recording from
hardware/record_joint_states.py. Joints are matched BY NAME; joints the file
does not contain stay at the `home` keyframe.

Uses models/locobot/ -- the whole robot (base, plate, camera tower, pan/tilt,
mobile_wx200 arm) expanded from the robot's own xacro. Its forward kinematics
match the TF recorded on the hardware to 0.16 mm when stationary.

Two kinds of replay
-------------------
KINEMATIC (default). Every recorded sample is written straight into qpos. This
is the hardware motion exactly, on the full robot model, which is what you want
for studying the path: where the end effector went, how close the arm came to
the robot's own body, where it was fast or jerky.

CLOSED LOOP (--closed-loop). Re-runs the same Cartesian servo in simulation,
from the recorded start to the hardware's final end-effector position, with the
same gains and caps. This is the fair "does the sim predict the hardware" test
for a reach run, and the one to use before trusting a path planned in sim.

OPEN LOOP (--dynamic intvelocity|velocity). Feeds the recorded joint-velocity
COMMANDS straight to the simulated actuators. Useful for understanding the
servo, but it cannot match a closed-loop hardware run, and on reach3 neither
mode does:
  * the hardware delivered only 68-84% of the joint motion it was commanded
    (stick-slip dwell), so intvelocity, which executes commands faithfully,
    ends 102 mm past it
  * velocity mode collapses under gravity (EE ends 10 cm below the arm base);
    the real Dynamixel velocity loop has integral action and holds the arm, so
    <velocity> is NOT a good analogue of it, despite the name
  * rounding the commands to the 0.229 rev/min velocity quantum changed the
    result by 0.5 mm: stick-slip comes from the servo's feedback loop, not from
    how commands are rounded

Every run writes <csv>_replay.csv (per-sample EE position, clearance, nearest
obstacle, and the sim deviation if dynamic) and <csv>_replay.png.
"""
import argparse
import csv
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models", "locobot")
ARM = ["waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate"]
CLEAR_MAX = 0.15        # m; clearance beyond this is reported as >= 150 mm


# ----------------------------------------------------------------------------
# data

def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path}: no rows")
    cols = rows[0].keys()
    num = lambda c: np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan
                              for r in rows])
    data = {"time": num("time"), "cols": list(cols)}
    for c in cols:
        if c != "time":
            try:
                data[c] = num(c)
            except ValueError:
                pass
    missing = [j for j in ARM if j not in data]
    if missing:
        sys.exit(f"{path}: no columns for {missing}")
    data["time"] = data["time"] - data["time"][0]
    return data


# ----------------------------------------------------------------------------
# model

class Robot:
    def __init__(self, dynamic=None):
        scene = f"scene_{dynamic}.xml" if dynamic else "scene.xml"
        self.m = mujoco.MjModel.from_xml_path(os.path.join(MODELS, scene))
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                            for i in range(m.njnt)]
        self.qadr = {n: m.jnt_qposadr[jid(n)] for n in self.joint_names}
        self.vadr = {n: m.jnt_dofadr[jid(n)] for n in self.joint_names}
        self.ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_gripper")
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "arm_base")
        self._setup_clearance()
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)

    def set_joints(self, values):
        for n, v in values.items():
            if n in self.qadr and np.isfinite(v):
                self.d.qpos[self.qadr[n]] = v
        # keep the finger mimic consistent when only one side was recorded
        if "left_finger" in values and "right_finger" not in values:
            self.d.qpos[self.qadr["right_finger"]] = -values["left_finger"]

    def ee_world(self):
        return self.d.site_xpos[self.ee].copy()

    def ee_arm_base(self):
        """EE in locobot/arm_base_link -- the frame the hardware logs use."""
        R = self.d.site_xmat[self.base].reshape(3, 3)
        return R.T @ (self.d.site_xpos[self.ee] - self.d.site_xpos[self.base])

    # -- clearance -----------------------------------------------------------

    def _setup_clearance(self):
        """Arm geoms vs everything the arm could hit: the robot's own body,
        the camera, and the floor. Pairs the model excludes as structural
        (the waist on its stand, and so on) are skipped, as is arm-vs-arm."""
        m = self.m
        shoulder = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "locobot/shoulder_link")

        def in_arm(b):
            while b > 0:
                if b == shoulder:
                    return True
                b = m.body_parentid[b]
            return False

        excluded = set()
        for s in m.exclude_signature[:m.nexclude]:
            a, b = int(s) >> 16, int(s) & 0xFFFF
            excluded |= {(a, b), (b, a)}

        collide = [g for g in range(m.ngeom)
                   if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.arm_geoms = [g for g in collide if in_arm(m.geom_bodyid[g])]
        self.obst_geoms = [g for g in collide if not in_arm(m.geom_bodyid[g])]
        self.pairs = [(a, o) for a in self.arm_geoms for o in self.obst_geoms
                      if (m.geom_bodyid[a], m.geom_bodyid[o]) not in excluded]
        self._pa = np.array([p[0] for p in self.pairs])
        self._po = np.array([p[1] for p in self.pairs])
        self.plane = {g for g in self.obst_geoms
                      if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE}

    def body_of(self, g):
        n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g])
        return (n or "floor").replace("locobot/", "")

    def clearance(self):
        """Smallest signed distance from the arm to anything it could hit (m),
        and the two bodies involved. Negative means penetration."""
        m, d = self.m, self.d
        # broadphase on bounding spheres; planes have no useful bound
        ca, co = d.geom_xpos[self._pa], d.geom_xpos[self._po]
        gap = (np.linalg.norm(ca - co, axis=1)
               - m.geom_rbound[self._pa] - m.geom_rbound[self._po])
        cand = np.nonzero((gap < CLEAR_MAX) |
                          np.isin(self._po, list(self.plane)))[0]
        best, who = CLEAR_MAX, ("", "")
        for k in cand:
            a, o = self.pairs[k]
            dist = mujoco.mj_geomDistance(m, d, a, o, CLEAR_MAX, None)
            if dist < best:
                best, who = dist, (self.body_of(a), self.body_of(o))
        return best, who


# ----------------------------------------------------------------------------
# replays

def kinematic(robot, data):
    n = len(data["time"])
    joints = [j for j in robot.joint_names if j in data]
    out = dict(ee=np.zeros((n, 3)), ee_world=np.zeros((n, 3)),
               clear=np.zeros(n), near=[None] * n, ncon=np.zeros(n, int))
    robot.reset()
    for i in range(n):
        robot.set_joints({j: data[j][i] for j in joints})
        mujoco.mj_forward(robot.m, robot.d)
        out["ee"][i] = robot.ee_arm_base()
        out["ee_world"][i] = robot.ee_world()
        out["clear"][i], out["near"][i] = robot.clearance()
        out["ncon"][i] = robot.d.ncon
    out["joints"] = joints
    return out


def dynamic(robot, data):
    """Open-loop replay of the recorded arm velocity commands."""
    t = data["time"]
    n = len(t)
    if all(f"qd_cmd_{j}" in data for j in ARM):
        cmd = np.column_stack([data[f"qd_cmd_{j}"] for j in ARM])
        source = "recorded qd_cmd"
    else:
        cmd = np.column_stack([np.gradient(data[j], t) for j in ARM])
        source = "d/dt of recorded joint angles (no qd_cmd columns in this file)"

    m, d = robot.m, robot.d
    aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_v") for j in ARM]
    hold = {j: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_p")
            for j in ("pan", "tilt", "left_finger")}

    robot.reset()
    robot.set_joints({j: data[j][0] for j in robot.joint_names if j in data})
    mujoco.mj_forward(m, d)
    for k, j in enumerate(ARM):                  # seed intvelocity setpoints
        if m.actuator_dyntype[aid[k]] == mujoco.mjtDyn.mjDYN_INTEGRATOR:
            d.act[m.actuator_actadr[aid[k]]] = d.qpos[robot.qadr[j]]
    for j, a in hold.items():
        d.ctrl[a] = d.qpos[robot.qadr[j]]

    q = np.zeros((n, 5))
    ee = np.zeros((n, 3))
    ee_world = np.zeros((n, 3))
    dt = m.opt.timestep
    for i in range(n):
        mujoco.mj_forward(m, d)
        q[i] = [d.qpos[robot.qadr[j]] for j in ARM]
        ee[i] = robot.ee_arm_base()
        ee_world[i] = robot.ee_world()
        if i + 1 < n:
            d.ctrl[aid] = cmd[i]
            for _ in range(max(1, int(round((t[i + 1] - t[i]) / dt)))):
                mujoco.mj_step(m, d)
    return dict(kind="open loop", t=t.copy(), q=q, ee=ee, ee_world=ee_world,
                cmd=cmd, source=source)


def closed_loop(data, kin, rate_hz=50.0, tol=0.005, stall_time=2.5, extra=5.0):
    """Re-run the hardware's Cartesian servo in simulation, same start, same
    goal, same gains and caps, closing the loop on SIMULATED state.

    This is the fair sim-vs-hardware test for a closed-loop run. Open-loop
    command replay cannot match one: on reach3 the hardware delivered only
    68-84% of the joint motion it was commanded (stick-slip dwell swallows part
    of every command and the servo keeps asking for more), so a sim that
    executes commands faithfully overshoots by construction.
    """
    sys.path.insert(0, os.path.join(HERE, "..", "hardware"))
    from cartesian_reach import CartesianReach

    robot = Robot(dynamic="intvelocity")
    reach = CartesianReach(model_path=os.path.join(MODELS, "locobot_wx200.xml"),
                           base_site="arm_base", tol=tol)
    m, d = robot.m, robot.d
    aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_v") for j in ARM]
    target = kin["ee"][-1]

    robot.reset()
    robot.set_joints({j: data[j][0] for j in robot.joint_names if j in data})
    mujoco.mj_forward(m, d)
    for k, j in enumerate(ARM):
        d.act[m.actuator_actadr[aid[k]]] = d.qpos[robot.qadr[j]]

    steps = max(1, int(round(1.0 / (rate_hz * m.opt.timestep))))
    T, Q, EE, EEW = [], [], [], []
    best, best_t = np.inf, 0.0
    t_end = data["time"][-1] + extra
    while d.time <= t_end:
        mujoco.mj_forward(m, d)
        q = np.array([d.qpos[robot.qadr[j]] for j in ARM])
        T.append(d.time)
        Q.append(q)
        EE.append(robot.ee_arm_base())
        EEW.append(robot.ee_world())
        qd, _, dist = reach.step(q, target)
        if dist < best - 0.0005:
            best, best_t = dist, d.time
        if dist < tol or d.time - best_t > stall_time:
            break
        d.ctrl[aid] = qd
        for _ in range(steps):
            mujoco.mj_step(m, d)
    return dict(kind="closed loop", t=np.array(T), q=np.array(Q), ee=np.array(EE),
                ee_world=np.array(EEW), target=target,
                source="CartesianReach re-run in sim (intvelocity), goal = "
                       "hardware's final EE position")


def at_time(sim, t):
    """Index into a sim run by time -- closed-loop runs have their own clock."""
    return min(int(np.searchsorted(sim["t"], t)), len(sim["t"]) - 1)


def path_distance(a, b):
    """Symmetric Hausdorff distance between two paths: timing-independent
    measure of whether they trace the same curve."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return max(d.min(axis=1).max(), d.min(axis=0).max())


# ----------------------------------------------------------------------------
# visual output

def add_trail(scn, pts, rgba, width=0.004, stride=1):
    pts = pts[::stride]
    for a, b in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.eye(3).ravel(),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scn.ngeom += 1


def pose_at(robot, data, kin, sim, tt):
    """Put the model in the pose for time tt. The arm follows the sim run if
    there is one (blue trail = hardware, green = sim); otherwise the hardware.
    Returns the trail lengths to draw."""
    th = data["time"]
    i = min(int(np.searchsorted(th, tt)), len(th) - 1)
    robot.reset()
    robot.set_joints({j: data[j][i] for j in kin["joints"]})
    k = None
    if sim is not None:
        k = at_time(sim, tt)
        robot.set_joints(dict(zip(ARM, sim["q"][k])))
    mujoco.mj_forward(robot.m, robot.d)
    return i, k


def draw_trails(scn, kin, sim, i, k):
    add_trail(scn, kin["ee_world"][:i + 1], (0.1, 0.5, 1, 1),
              stride=max(1, i // 250 + 1))
    if sim is not None:
        add_trail(scn, sim["ee_world"][:k + 1], (0.1, 0.9, 0.3, 1),
                  stride=max(1, k // 250 + 1))


def view(robot, data, kin, sim, speed, loop):
    import mujoco.viewer
    t_end = max(data["time"][-1], sim["t"][-1] if sim is not None else 0.0)
    with mujoco.viewer.launch_passive(robot.m, robot.d) as v:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        while v.is_running():
            wall0 = time.time()
            while v.is_running():
                tt = (time.time() - wall0) * speed
                if tt > t_end:
                    break
                i, k = pose_at(robot, data, kin, sim, tt)
                with v.lock():
                    v.user_scn.ngeom = 0
                    draw_trails(v.user_scn, kin, sim, i, k)
                v.sync()
                time.sleep(0.01)
            if not loop:
                while v.is_running():
                    time.sleep(0.05)


def gif(robot, data, kin, sim, path, fps=20, camera="track", size=(640, 480)):
    from PIL import Image
    t_end = max(data["time"][-1], sim["t"][-1] if sim is not None else 0.0)
    r = mujoco.Renderer(robot.m, size[1], size[0])
    frames = []
    for tt in np.append(np.arange(0.0, t_end, 1.0 / fps), t_end):
        i, k = pose_at(robot, data, kin, sim, tt)
        r.update_scene(robot.d, camera=camera)
        draw_trails(r.scene, kin, sim, i, k)
        frames.append(Image.fromarray(r.render()))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    return len(frames)


def plot(path, data, kin, dyn, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D           # noqa: F401

    t = data["time"]
    ee = kin["ee"]
    tf = (np.column_stack([data[f"ee_{a}"] for a in "xyz"])
          if all(f"ee_{a}" in data for a in "xyz") else None)

    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(*ee.T, lw=2.2, color="#1f77b4", label="hardware (replayed on model)")
    if tf is not None and np.isfinite(tf).any():
        ax.plot(*tf.T, lw=1, ls="--", color="#555555", label="hardware TF (as logged)")
    if dyn is not None:
        ax.plot(*dyn["ee"].T, lw=2, color="#2ca02c", label=f"simulated ({dyn['kind']})")
    ax.scatter(*ee[0], color="green", s=50)
    ax.scatter(*ee[-1], color="red", s=50)
    worst = int(np.argmin(kin["clear"]))
    ax.scatter(*ee[worst], color="orange", s=120, marker="X",
               label=f"closest approach {1000*kin['clear'][worst]:.0f} mm")
    pts = ee if dyn is None else np.vstack([ee, dyn["ee"]])
    c, rad = pts.mean(0), max(np.ptp(pts, axis=0).max(), 0.05) / 2
    ax.set_xlim(c[0] - rad, c[0] + rad)
    ax.set_ylim(c[1] - rad, c[1] + rad)
    ax.set_zlim(c[2] - rad, c[2] + rad)
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.set_zlabel("z (m)")
    ax.set_title("(a) end-effector path, arm_base_link frame", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t, 1000 * kin["clear"], lw=1.8, color="#d62728")
    ax.axhline(0, color="k", lw=0.8)
    ax.fill_between(t, 1000 * kin["clear"], 0, where=kin["clear"] < 0,
                    color="red", alpha=0.3, label="penetration")
    ax.axhline(20, color="orange", ls=":", lw=1, label="20 mm")
    ax.set_ylim(min(-5, 1000 * kin["clear"].min() - 5), 1000 * CLEAR_MAX + 5)
    near = kin["near"][worst]
    ax.annotate(f"{near[0]} ↔ {near[1]}", (t[worst], 1000 * kin["clear"][worst]),
                xytext=(10, 25), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("time (s)"), ax.set_ylabel("clearance (mm)")
    ax.set_title("(b) arm clearance to robot body and floor (capped at 150 mm)",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 2, 3)
    speed = np.linalg.norm(np.gradient(ee, t, axis=0), axis=1)
    ax.plot(t, 1000 * speed, lw=1.4, color="#1f77b4", label="hardware EE speed")
    if dyn is not None:
        ax.plot(dyn["t"], 1000 * np.linalg.norm(np.gradient(dyn["ee"], dyn["t"], axis=0),
                                                axis=1),
                lw=1.4, color="#2ca02c", label="simulated EE speed")
    ax.set_xlabel("time (s)"), ax.set_ylabel("EE speed (mm/s)")
    ax.set_title("(c) end-effector speed", loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 2, 4)
    if dyn is not None and dyn["kind"] == "closed loop":
        goal = dyn["target"]
        ax.semilogy(t, 1000 * np.linalg.norm(ee - goal, axis=1), lw=2,
                    color="#1f77b4", label="hardware")
        ax.semilogy(dyn["t"], 1000 * np.linalg.norm(dyn["ee"] - goal, axis=1), lw=2,
                    color="#2ca02c", label="sim, same controller")
        ax.set_ylabel("distance to goal (mm, log)")
        ax.set_title("(d) convergence: hardware vs simulated controller", loc="left",
                     fontweight="bold")
    elif dyn is not None:
        dev = 1000 * np.linalg.norm(dyn["ee"] - ee, axis=1)
        ax.plot(t, dev, lw=2, color="#9467bd", label="|sim − hardware| EE")
        ax.set_ylabel("deviation (mm)")
        ax.set_title("(d) sim-to-real gap, open-loop command replay", loc="left",
                     fontweight="bold")
    elif tf is not None and np.isfinite(tf).any():
        ax.plot(t, 1000 * np.linalg.norm(ee - tf, axis=1), lw=1.2, color="#7f7f7f",
                label="model FK vs logged TF")
        ax.set_ylabel("mm")
        ax.set_title("(d) model agreement with the robot's TF", loc="left",
                     fontweight="bold")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_csv(path, data, kin, dyn):
    t = data["time"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        head = (["time"] + ARM + ["ee_x", "ee_y", "ee_z", "clearance_m",
                                  "nearest_arm_body", "nearest_obstacle"])
        per_row = dyn is not None and dyn["kind"] == "open loop"
        if per_row:
            head += ([f"sim_{j}" for j in ARM] + ["sim_ee_x", "sim_ee_y", "sim_ee_z",
                                                  "sim_deviation_m"])
        w.writerow(head)
        for i in range(len(t)):
            row = ([f"{t[i]:.4f}"] + [f"{data[j][i]:.6f}" for j in ARM]
                   + [f"{v:.6f}" for v in kin["ee"][i]]
                   + [f"{kin['clear'][i]:.5f}", kin["near"][i][0], kin["near"][i][1]])
            if per_row:
                row += ([f"{v:.6f}" for v in dyn["q"][i]]
                        + [f"{v:.6f}" for v in dyn["ee"][i]]
                        + [f"{np.linalg.norm(dyn['ee'][i]-kin['ee'][i]):.6f}"])
            w.writerow(row)
    if dyn is not None and not per_row:
        # a closed-loop run has its own clock, so it gets its own file
        sim_path = path.replace("_replay.csv", "_closedloop.csv")
        with open(sim_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time"] + ARM + ["ee_x", "ee_y", "ee_z"])
            for i in range(len(dyn["t"])):
                w.writerow([f"{dyn['t'][i]:.4f}"] + [f"{v:.6f}" for v in dyn["q"][i]]
                           + [f"{v:.6f}" for v in dyn["ee"][i]])
        return sim_path


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    sim = ap.add_mutually_exclusive_group()
    sim.add_argument("--dynamic", choices=["intvelocity", "velocity"],
                     help="also re-simulate the recorded commands open loop")
    sim.add_argument("--closed-loop", action="store_true",
                     help="also re-run the Cartesian servo in sim from the same "
                          "start to the hardware's final EE position")
    ap.add_argument("--view", action="store_true", help="interactive viewer")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed")
    ap.add_argument("--loop", action="store_true", help="viewer: repeat")
    ap.add_argument("--gif", help="write an animation (headless)")
    ap.add_argument("--camera", default="track", choices=["track", "side", "front"])
    args = ap.parse_args()

    data = load_csv(args.csv)
    robot = Robot(dynamic=args.dynamic)
    stem = os.path.splitext(args.csv)[0]

    t0 = time.time()
    kin = kinematic(robot, data)
    if args.dynamic:
        dyn = dynamic(robot, data)
    elif args.closed_loop:
        dyn = closed_loop(data, kin)
    else:
        dyn = None
    elapsed = time.time() - t0

    t, ee = data["time"], kin["ee"]
    worst = int(np.argmin(kin["clear"]))
    path_len = np.linalg.norm(np.diff(ee, axis=0), axis=1).sum()
    print(f"\n{os.path.basename(args.csv)}: {len(t)} samples over {t[-1]:.2f} s "
          f"(replayed in {elapsed:.2f} s)")
    print(f"  joints matched by name : {kin['joints']}")
    print(f"  EE start -> end        : {ee[0].round(4)} -> {ee[-1].round(4)}")
    # Raw sample-to-sample path length is inflated by stick-slip zig-zag (the
    # joints dwell, then jump a few encoder counts), so it overstates any real
    # detour. Deviation from the straight line and a 0.5 s-smoothed length
    # are the honest geometric measures.
    straight = np.linalg.norm(ee[-1] - ee[0])
    w = max(1, int(round(0.5 / max(np.diff(t).mean(), 1e-9))))
    k = np.ones(w) / w
    smooth = np.column_stack([np.convolve(np.pad(ee[:, i], (w // 2, w - 1 - w // 2),
                                                 mode="edge"), k, mode="valid")
                              for i in range(3)])
    smooth_len = np.linalg.norm(np.diff(smooth, axis=0), axis=1).sum()
    if straight > 1e-6:
        u = (ee[-1] - ee[0]) / straight
        rel = ee - ee[0]
        perp = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1).max()
    else:
        perp = 0.0
    print(f"  straight-line distance : {1000*straight:.1f} mm")
    print(f"  max off-line deviation : {1000*perp:.1f} mm   <- geometric detour")
    print(f"  path length, 0.5 s avg : {1000*smooth_len:.1f} mm "
          f"(ratio {smooth_len/max(straight,1e-9):.2f})")
    print(f"  path length, raw       : {1000*path_len:.1f} mm "
          f"(ratio {path_len/max(straight,1e-9):.2f}; the excess over the "
          f"smoothed length is jitter)")
    print(f"  closest approach       : {1000*kin['clear'][worst]:+.1f} mm at "
          f"t={t[worst]:.2f} s, {kin['near'][worst][0]} <-> {kin['near'][worst][1]}")
    print(f"  samples in penetration : {int((kin['clear'] < 0).sum())}")
    if all(f"ee_{a}" in data for a in "xyz"):
        tf = np.column_stack([data[f"ee_{a}"] for a in "xyz"])
        ok = np.isfinite(tf).all(axis=1)
        if ok.any():
            e = 1000 * np.linalg.norm(ee[ok] - tf[ok], axis=1)
            print(f"  model FK vs logged TF  : mean {e.mean():.2f} mm, max {e.max():.2f} mm")
    if dyn is not None and dyn["kind"] == "open loop":
        dev = 1000 * np.linalg.norm(dyn["ee"] - ee, axis=1)
        print(f"\n  open-loop replay ({args.dynamic}), commands from {dyn['source']}")
        print(f"  sim vs hardware EE     : mean {dev.mean():.1f} mm, "
              f"max {dev.max():.1f} mm, final {dev[-1]:.1f} mm")
        print(f"  sim end                : {dyn['ee'][-1].round(4)}")
        cmd_ok = all(f"qd_cmd_{j}" in data for j in ARM)
        if cmd_ok:
            th = data["time"]
            print("  commanded vs achieved joint motion on the HARDWARE:")
            for k, j in enumerate(ARM):
                want = (dyn["cmd"][:-1, k] * np.diff(th)).sum()
                got = data[j][-1] - data[j][0]
                if abs(want) > 1e-3:
                    print(f"    {j:<13} commanded {np.degrees(want):+7.1f} deg, "
                          f"moved {np.degrees(got):+7.1f} deg ({100*got/want:.0f}%)")
    elif dyn is not None:
        goal = dyn["target"]
        within = lambda tt, p: next((tt[i] for i in range(len(tt))
                                     if np.linalg.norm(p[i] - goal) < 0.010), None)
        th_hw, th_sim = within(t, ee), within(dyn["t"], dyn["ee"])
        hd = path_distance(ee[::2], dyn["ee"][::2])
        fmt = lambda v: f"{v:.2f} s" if v is not None else "never"
        print(f"\n  closed-loop re-run: {dyn['source']}")
        print(f"  goal                   : {goal.round(4)}")
        print(f"  time to within 10 mm   : hardware {fmt(th_hw)}, sim {fmt(th_sim)}")
        print(f"  final distance         : hardware "
              f"{1000*np.linalg.norm(ee[-1]-goal):.1f} mm (goal is its own end), "
              f"sim {1000*np.linalg.norm(dyn['ee'][-1]-goal):.1f} mm")
        print(f"  path shape difference  : {1000*hd:.1f} mm "
              f"(Hausdorff distance, timing-independent)")

    out_csv = f"{stem}_replay.csv"
    extra = write_csv(out_csv, data, kin, dyn)
    out_png = f"{stem}_replay.png"
    label = ("" if dyn is None else
             f"  (+ {args.dynamic} open-loop sim)" if dyn["kind"] == "open loop"
             else "  (+ closed-loop sim)")
    plot(out_png, data, kin, dyn,
         f"LoCoBot hardware replay — {os.path.basename(args.csv)}{label}")
    print(f"\nwrote {out_csv}")
    if extra:
        print(f"wrote {extra}")
    print(f"wrote {out_png}")

    if args.gif:
        n = gif(robot, data, kin, dyn, args.gif, camera=args.camera)
        print(f"wrote {args.gif} ({n} frames)")
    if args.view:
        view(robot, data, kin, dyn, args.speed, args.loop)


if __name__ == "__main__":
    main()
