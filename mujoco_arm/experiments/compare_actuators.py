#!/usr/bin/env python3
"""Phase 2 / milestone M2: does the actuator choice actually matter?

    python3 experiments/compare_actuators.py

Runs two experiments against both actuator variants and writes
experiments/compare_actuators.png plus a CSV of the raw traces.

  HOLD   From the home pose (forearm cantilevered horizontally: 1.27 N m on the
         shoulder, 0.98 N m on the elbow), command qd = 0 for 3 s.
         This is the whole argument for intvelocity: a plain <velocity> servo
         produces torque only from velocity ERROR, so at qd_cmd = 0 it has no
         way to oppose gravity and simply sinks.

  STEP   Command the elbow -0.5 rad/s for 1.5 s, then 0 for 1.5 s.
         Shows tracking during motion, whether the UNCOMMANDED joints stay put
         while one joint moves, and what happens on release.

Pose choice matters more than it looks. The first version of this experiment
ran from the arm-extended-forward pose, which has the largest gravity torque
(2.4 N m) and looks like the obvious worst case. It was unusable:
  - the EE sits 6 cm off the ground there, so the sinking arm landed on the
    floor and we measured the ground reaction, not the servo;
  - with the floor removed, the upper arm reaches its own BASE after only
    ~10 deg of travel (shoulder 100.6 deg, well inside the 113 deg joint
    limit), so there was no room to sink either;
  - driving the elbow from that pose pushed the gripper into the floor and
    saturated the actuator at 12.1 of 13 N m.
The home pose has a real gravity load, ~100 deg of clearance to sink into, and
stays under 11% of the effort limit. run() now asserts both conditions rather
than trusting the pose.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_sim import ArmSim, ARM_JOINTS, MODES          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Home: upper arm vertical, forearm cantilevered horizontally forward. Gravity
# loads the shoulder (1.27 N m) and elbow (0.98 N m), and the arm has ~100 deg
# of room to sink before the upper arm reaches the base. See the module
# docstring for why the higher-torque extended pose does not work here.
START = [0.0, 0.0, 0.0, 0.0, 0.0]

HOLD_T = 3.0
STEP_V = -0.5          # elbow drops away from the base, into free space
STEP_T = 1.5
ELBOW = ARM_JOINTS.index("elbow")
SHOULDER = ARM_JOINTS.index("shoulder")


def run(mode, kind):
    # bench=True: no floor. Both experiments deliberately let the arm sink, and
    # with a floor in the scene the gripper lands and we end up measuring the
    # ground reaction instead of the servo. See ArmSim.__init__.
    sim = ArmSim(mode, bench=True)
    sim.set_qpos(START)
    q0 = sim.q.copy()

    rows = []
    ncon = 0
    total = HOLD_T if kind == "hold" else 2 * STEP_T
    for i in range(int(total / sim.dt)):
        t = i * sim.dt
        if kind == "hold":
            cmd = np.zeros(5)
        else:
            cmd = np.zeros(5)
            cmd[ELBOW] = STEP_V if t < STEP_T else 0.0
        sim.set_joint_velocity(cmd)
        sim.step()
        ncon = max(ncon, sim.data.ncon)
        rows.append(dict(t=t, cmd=cmd.copy(), q=sim.q, qd=sim.qd,
                         ee=sim.ee_pos, tau=sim.torque))

    # Any contact at all means the run measured something other than the servo.
    if ncon:
        raise RuntimeError(
            f"{mode}/{kind}: {ncon} contact(s) during the run -- the arm hit "
            f"something, so these numbers are not a servo measurement")

    # Saturating the effort limit also invalidates a tracking comparison: both
    # modes would just be reporting the torque ceiling.
    tau = np.array([r["tau"] for r in rows])
    lim = sim.model.jnt_actfrcrange[[sim.jid[k] for k in range(5)], 1]
    sat = np.abs(tau).max(axis=0) / lim
    if sat.max() > 0.95:
        j = ARM_JOINTS[int(sat.argmax())]
        raise RuntimeError(
            f"{mode}/{kind}: {j} actuator hit {100*sat.max():.0f}% of its "
            f"effort limit -- saturated, not a servo measurement")

    return dict(mode=mode, kind=kind, q0=q0, sat=sat,
                t=np.array([r["t"] for r in rows]),
                cmd=np.array([r["cmd"] for r in rows]),
                q=np.array([r["q"] for r in rows]),
                qd=np.array([r["qd"] for r in rows]),
                ee=np.array([r["ee"] for r in rows]),
                tau=np.array([r["tau"] for r in rows]))


def main():
    res = {(m, k): run(m, k) for m in MODES for k in ("hold", "step")}

    style = {"intvelocity": dict(color="#1b7f4b", lw=2.0),
             "velocity": dict(color="#c0392b", lw=2.0, ls="--")}

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    # (a) hold: shoulder and elbow drift ------------------------------------
    a = ax[0, 0]
    for m in MODES:
        r = res[(m, "hold")]
        for j, lbl, ls in ((SHOULDER, "shoulder", "-"), (ELBOW, "elbow", ":")):
            a.plot(r["t"], np.degrees(r["q"][:, j] - r["q0"][j]),
                   color=style[m]["color"], lw=2.0, ls=ls,
                   label=f"{m} · {lbl}")
    a.axhline(0, color="k", lw=0.6)
    a.set_title("(a) HOLD: joint drift with commanded velocity = 0", loc="left",
                fontweight="bold")
    a.set_xlabel("time (s)")
    a.set_ylabel("drift from start (deg)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # (b) hold: EE height ----------------------------------------------------
    a = ax[0, 1]
    for m in MODES:
        r = res[(m, "hold")]
        a.plot(r["t"], r["ee"][:, 2] * 100, label=m, **style[m])
    a.axhline(res[("intvelocity", "hold")]["ee"][0, 2] * 100,
              color="k", lw=0.6, ls=":", label="start height")
    a.set_title("(b) HOLD: end-effector height", loc="left", fontweight="bold")
    a.set_xlabel("time (s)")
    a.set_ylabel("EE height (cm)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # (c) step: elbow velocity tracking --------------------------------------
    a = ax[1, 0]
    r0 = res[("intvelocity", "step")]
    a.plot(r0["t"], r0["cmd"][:, ELBOW], color="k", lw=1.2, ls="-.",
           label="commanded")
    for m in MODES:
        r = res[(m, "step")]
        a.plot(r["t"], r["qd"][:, ELBOW], label=m, **style[m])
    a.set_title("(c) STEP: elbow velocity tracking", loc="left",
                fontweight="bold")
    a.set_xlabel("time (s)")
    a.set_ylabel("elbow velocity (rad/s)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # (d) step: elbow angle, i.e. does it stay where you left it? ------------
    a = ax[1, 1]
    for m in MODES:
        r = res[(m, "step")]
        a.plot(r["t"], np.degrees(r["q"][:, ELBOW]), label=m, **style[m])
    a.axvline(STEP_T, color="k", lw=0.8, ls=":")
    a.annotate("command -> 0", xy=(STEP_T, a.get_ylim()[0]),
               xytext=(STEP_T + 0.08, a.get_ylim()[0] + 3), fontsize=8)
    a.set_title("(d) STEP: elbow angle — does it hold after release?",
                loc="left", fontweight="bold")
    a.set_xlabel("time (s)")
    a.set_ylabel("elbow angle (deg)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    fig.suptitle("WX200 in MuJoCo — intvelocity vs velocity actuators "
                 "(home pose: forearm cantilevered, 1.27 N·m on the shoulder, 0.98 N·m on the elbow)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = os.path.join(HERE, "compare_actuators.png")
    fig.savefig(png, dpi=110)
    print(f"wrote {png}")

    # raw traces, so the numbers behind the plot are checkable ---------------
    csv_path = os.path.join(HERE, "compare_actuators.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "experiment", "time"]
                   + [f"cmd_{j}" for j in ARM_JOINTS]
                   + [f"q_{j}" for j in ARM_JOINTS]
                   + [f"qd_{j}" for j in ARM_JOINTS]
                   + [f"tau_{j}" for j in ARM_JOINTS]
                   + ["ee_x", "ee_y", "ee_z"])
        for (m, k), r in res.items():
            for i in range(len(r["t"])):
                w.writerow([m, k, f"{r['t'][i]:.4f}"]
                           + list(np.round(r["cmd"][i], 6))
                           + list(np.round(r["q"][i], 6))
                           + list(np.round(r["qd"][i], 6))
                           + list(np.round(r["tau"][i], 6))
                           + list(np.round(r["ee"][i], 6)))
    print(f"wrote {csv_path}")

    # -- the numbers that decide the phase 2 choice --------------------------
    print("\n" + "=" * 72)
    print("HOLD for 3 s at commanded velocity = 0")
    print("=" * 72)
    print(f"{'mode':<14}{'shoulder drift':>16}{'elbow drift':>14}"
          f"{'EE sink':>11}{'final |qd|':>13}")
    print(f"{'':14}{'deg':>16}{'deg':>14}{'cm':>11}{'rad/s':>13}")
    print("-" * 72)
    for m in MODES:
        r = res[(m, "hold")]
        ds = np.degrees(r["q"][-1, SHOULDER] - r["q0"][SHOULDER])
        de = np.degrees(r["q"][-1, ELBOW] - r["q0"][ELBOW])
        sink = (r["ee"][-1, 2] - r["ee"][0, 2]) * 100
        print(f"{m:<14}{ds:16.2f}{de:14.2f}{sink:11.2f}"
              f"{np.abs(r['qd'][-1]).max():13.4f}")

    print("\nis it still moving at t = 3 s? (a settled servo would read ~0)")
    for m in MODES:
        r = res[(m, "hold")]
        late = r["qd"][int(2.5 / 0.002):]
        print(f"  {m:<14} mean |qd| over the last 0.5 s = "
              f"{np.abs(late).max(axis=1).mean():.4f} rad/s")

    print("\n" + "=" * 72)
    print(f"STEP: elbow {STEP_V} rad/s for {STEP_T} s, then 0")
    print("=" * 72)
    for m in MODES:
        r = res[(m, "step")]
        moving = (r["t"] > 0.4) & (r["t"] < STEP_T)
        tracked = r["qd"][moving, ELBOW].mean()
        after = r["t"] > STEP_T
        drift = np.degrees(r["q"][-1, ELBOW] - r["q"][after, ELBOW][0])
        print(f"  {m:<14} tracked {tracked:+.4f} rad/s "
              f"(cmd {STEP_V:+.2f}, err {100*(tracked-STEP_V)/STEP_V:+5.1f}%) "
              f"| drift after release: {drift:+7.2f} deg")


if __name__ == "__main__":
    main()
