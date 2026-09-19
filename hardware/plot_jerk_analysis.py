#!/usr/bin/env python3
"""Where the jerk in a reach comes from, and what it costs in position error.

    python3 plot_jerk_analysis.py reach3.csv --target 0.35 0 0.20

Companion to plot_reach_3d.py: that one shows the path, this one shows why the
path is not smooth.

A steppy velocity trace has two possible explanations -- the arm really moves in
steps, or it moves smoothly and the sensor is coarse -- and velocity data alone
cannot tell them apart here, because BOTH available velocity signals are
quantization limited:

  qd_act   what the Dynamixel reports, quantized to 0.229 rev/min
           = 0.023981 rad/s (the velocity register LSB)
  qd_fd    position differentiated at the 50 Hz control rate. The position
           register is fine (0.088 deg) but over a 0.02 s sample that is still
           0.0767 rad/s per count -- 3.2x COARSER than the velocity register.
           So this is not the finer signal, despite the finer sensor.

The test that does discriminate is the distribution of per-sample ENCODER COUNT
increments (panel b). Smooth motion at a commanded rate v shows a tight cluster
around v*dt/LSB counts every sample. Stick-slip shows a large spike at zero
counts -- the joint standing still -- with occasional multi-count jumps.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOINTS = ["waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate"]
# Dynamixel X-series velocity register: 0.229 rev/min per LSB
VEL_LSB = 0.229 * 2 * np.pi / 60          # 0.023981 rad/s
# position register: 4096 counts / 2 pi
POS_LSB = 2 * np.pi / 4096                # 0.001534 rad = 0.088 deg


def smooth(y, n=9):
    if len(y) < n:
        return y
    k = np.ones(n) / n
    return np.convolve(np.pad(y, (n // 2, n // 2), mode="edge"), k, mode="valid")[:len(y)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ap.add_argument("--joint", default="elbow", help="joint to zoom on (panel b)")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    out = args.out or args.csv.replace(".csv", "_jerk.png")
    col = lambda n: np.array([float(r[n]) for r in rows])

    t = col("time")
    q = np.column_stack([col(j) for j in JOINTS])
    cmd = np.column_stack([col("qd_cmd_" + j) for j in JOINTS])
    act = np.column_stack([col("qd_act_" + j) for j in JOINTS])
    ee = np.column_stack([col("ee_" + a) for a in "xyz"])
    dist = col("dist") if "dist" in rows[0] else None
    tgt = np.array(args.target) if args.target else None

    # position-derived velocity, and EE kinematics
    qd_fd = np.column_stack([np.gradient(q[:, i], t) for i in range(5)])
    ee_v = np.column_stack([np.gradient(ee[:, i], t) for i in range(3)])
    ee_speed = np.linalg.norm(ee_v, axis=1)
    ee_a = np.column_stack([np.gradient(smooth(ee_v[:, i]), t) for i in range(3)])
    ee_j = np.column_stack([np.gradient(smooth(ee_a[:, i]), t) for i in range(3)])
    jerk = np.linalg.norm(ee_j, axis=1)

    zi = JOINTS.index(args.joint)
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    # -- (a) commanded vs reported vs position-derived -----------------------
    a = ax[0, 0]
    active = [i for i in range(5) if np.abs(cmd[:, i]).max() > 1e-6]
    for i in active:
        c = f"C{i}"
        a.plot(t, cmd[:, i], color=c, ls="--", lw=1.3)
        a.plot(t, act[:, i], color=c, lw=1.6, alpha=0.85, label=JOINTS[i])
    a.set_ylabel("joint velocity (rad/s)")
    a.set_xlabel("time (s)")
    a.set_title("(a) commanded (dashed) vs reported (solid)", loc="left",
                fontweight="bold")
    a.legend(fontsize=8, ncol=2)
    a.grid(alpha=0.3)

    # -- (b) stick-slip evidence: per-sample encoder count increments --------
    # This is the panel that actually discriminates. Both velocity signals are
    # quantization limited, so compare how the POSITION advances instead.
    a = ax[0, 1]
    dt = np.diff(t).mean()
    moving = np.abs(cmd[:-1, zi]) > 0.5 * VEL_LSB
    counts = np.round(np.diff(q[:, zi]) / POS_LSB).astype(int)[moving]
    expect = np.abs(cmd[:-1, zi][moving]).mean() * dt / POS_LSB
    if len(counts):
        lo, hi = counts.min(), counts.max()
        bins = np.arange(lo - 0.5, hi + 1.5)
        a.hist(np.abs(counts), bins=np.arange(-0.5, np.abs(counts).max() + 1.5),
               color="#c0392b", alpha=0.8, rwidth=0.85)
        a.axvline(expect, color="k", ls="--", lw=1.8,
                  label=f"expected if smooth: {expect:.2f} counts/sample")
        dwell = 100 * (counts == 0).mean()
        a.set_title(f"(b) {args.joint}: {dwell:.0f}% of samples the joint does "
                    f"NOT move", loc="left", fontweight="bold")
        a.legend(fontsize=8)
    a.set_xlabel("encoder counts moved per 20 ms sample")
    a.set_ylabel("number of samples")
    a.grid(alpha=0.3, axis="y")

    # -- (c) EE speed and jerk ----------------------------------------------
    a = ax[1, 0]
    a.plot(t, ee_speed * 1000, lw=1.6, color="#1f77b4", label="EE speed")
    a.set_ylabel("EE speed (mm/s)", color="#1f77b4")
    a.set_xlabel("time (s)")
    a2 = a.twinx()
    a2.plot(t, jerk, lw=1.1, color="#d62728", alpha=0.65, label="|jerk|")
    a2.set_ylabel("|EE jerk| (m/s³)", color="#d62728")
    a.set_title("(c) end-effector speed and jerk", loc="left", fontweight="bold")
    a.grid(alpha=0.3)

    # -- (d) convergence, with the stall called out --------------------------
    a = ax[1, 1]
    if dist is None and tgt is not None:
        dist = np.linalg.norm(ee - tgt, axis=1)
    if dist is not None:
        a.semilogy(t, dist * 1000, lw=2, color="#c0392b")
        a.axhline(5, color="k", ls=":", lw=1, label="5 mm tolerance")
        # the region where the commanded velocity is below one register quantum
        below = np.abs(cmd).max(axis=1) < VEL_LSB
        if below.any():
            first = t[np.argmax(below)]
            a.axvspan(first, t[-1], color="orange", alpha=0.15)
            a.text(first, dist.max() * 500, "  commanded velocity below\n"
                   "  one encoder quantum", fontsize=8, va="top")
        a.set_ylabel("distance to target (mm, log)")
        a.legend(fontsize=8)
    a.set_xlabel("time (s)")
    a.set_title("(d) convergence and the deadband floor", loc="left",
                fontweight="bold")
    a.grid(alpha=0.3, which="both")

    fig.suptitle(f"WX200 reach — jerk and quantization analysis "
                 f"({os.path.basename(args.csv)})", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    print(f"wrote {out}\n")

    # -- numbers -------------------------------------------------------------
    u = np.unique(np.abs(act.round(6)))
    u = u[u > 0]
    dt = np.diff(t).mean()
    print(f"velocity register quantum : {VEL_LSB:.6f} rad/s "
          f"(0.229 rev/min, Dynamixel X-series)")
    print(f"smallest reported |qd|    : {u.min():.6f} rad/s "
          f"-> {u.min()/VEL_LSB:.3f} quanta")
    print(f"position register         : {POS_LSB:.6f} rad "
          f"({np.degrees(POS_LSB):.3f} deg)")
    print(f"  ...but at {1/dt:.0f} Hz that is {POS_LSB/dt:.4f} rad/s per count, "
          f"{POS_LSB/dt/VEL_LSB:.1f}x COARSER\n  than the velocity register, so "
          f"differentiating position is not the finer signal.\n")

    # the discriminating statistic
    for i, j in enumerate(JOINTS):
        mv = np.abs(cmd[:-1, i]) > 0.5 * VEL_LSB
        if mv.sum() < 20:
            continue
        c = np.round(np.diff(q[:, i]) / POS_LSB).astype(int)[mv]
        exp = np.abs(cmd[:-1, i][mv]).mean() * dt / POS_LSB
        print(f"  {j:<13} expected {exp:4.2f} counts/sample if smooth; "
              f"actually still {100*(c==0).mean():4.1f}% of samples, "
              f"mean jump {np.abs(c[c != 0]).mean():.2f} counts")
    print()

    print(f"{'joint':<14}{'max cmd':>10}{'time cmd < 1 quantum':>22}"
          f"{'RMS(act-cmd)':>14}")
    for i, j in enumerate(JOINTS):
        if np.abs(cmd[:, i]).max() < 1e-9:
            continue
        frac = 100 * np.mean((np.abs(cmd[:, i]) > 0) & (np.abs(cmd[:, i]) < VEL_LSB))
        print(f"{j:<14}{np.abs(cmd[:, i]).max():10.4f}{frac:21.1f}%"
              f"{np.sqrt(((act[:, i]-cmd[:, i])**2).mean()):14.4f}")

    print(f"\nEE peak speed {1000*ee_speed.max():.1f} mm/s, "
          f"peak |jerk| {jerk.max():.2f} m/s^3, median {np.median(jerk):.2f}")
    if dist is not None:
        print(f"final distance {1000*dist[-1]:.2f} mm")


if __name__ == "__main__":
    main()
