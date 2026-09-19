#!/usr/bin/env python3
"""3D end-effector path plus convergence and error, from a reach CSV.

    python3 plot_reach_3d.py reach_trajectory.csv --target 0.35 0 0.20
    python3 plot_reach_3d.py reach_trajectory.csv --target 0.35 0 0.20 -o out.png

Reads what reach_and_log_hw.py (real arm) and test_reach_sim.py (simulation)
write. This is scripts/plot_reach_trajectory.py extended -- same idea, but it
also draws the target, the convergence curve and the per-axis error, and it
uses plain csv instead of pandas (which is not installed on the robot).
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401  (registers 3d)
import numpy as np

JOINTS = ["waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate"]


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} has no data rows")
    col = lambda n: np.array([float(r[n]) for r in rows])
    grab = lambda p: np.column_stack([col(f"{p}{a}") for a in "xyz"])
    d = dict(t=col("time"), ee=grab("ee_"), q=np.column_stack([col(j) for j in JOINTS]))
    d["fk"] = grab("fk_") if "fk_x" in rows[0] else d["ee"]
    d["dist"] = col("dist") if "dist" in rows[0] else None
    d["sigma"] = col("sigma") if "sigma" in rows[0] else None
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    d = load(args.csv)
    out = args.out or args.csv.replace(".csv", "_3d.png")
    ee, fk, t = d["ee"], d["fk"], d["t"]
    tgt = np.array(args.target) if args.target else None

    fig = plt.figure(figsize=(14, 9))

    # -- (a) the 3D path -----------------------------------------------------
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], lw=2, color="#1f77b4",
            label="EE path (robot TF)")
    if not np.allclose(ee, fk, equal_nan=True):
        ax.plot(fk[:, 0], fk[:, 1], fk[:, 2], lw=1.2, ls="--", color="#888888",
                label="model FK")
    ax.scatter(*ee[0], color="green", s=60, label="start")
    ax.scatter(*ee[-1], color="red", s=60, label="end")
    if tgt is not None:
        ax.scatter(*tgt, color="black", marker="*", s=220, label="target")
        ax.plot(*np.column_stack([ee[-1], tgt]), color="red", lw=1.5, ls=":")
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.set_zlabel("z (m)")
    ax.set_title("(a) end-effector path", loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    # equal aspect: matplotlib 3d has no set_box_aspect in 3.1, so pad manually
    pts = np.vstack([ee] + ([tgt[None, :]] if tgt is not None else []))
    c, r = pts.mean(0), max(np.ptp(pts, axis=0).max(), 0.05) / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)

    # -- (b) position vs time ------------------------------------------------
    ax = fig.add_subplot(2, 2, 2)
    for i, a in enumerate("xyz"):
        line, = ax.plot(t, ee[:, i], lw=1.8, label=f"ee_{a}")
        if tgt is not None:
            ax.axhline(tgt[i], color=line.get_color(), ls=":", lw=1.2)
    ax.set_xlabel("time (s)"), ax.set_ylabel("position (m)")
    ax.set_title("(b) EE position vs time (dotted = target)", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)

    # -- (c) convergence -----------------------------------------------------
    ax = fig.add_subplot(2, 2, 3)
    dist = d["dist"] if d["dist"] is not None else (
        np.linalg.norm(ee - tgt, axis=1) if tgt is not None else None)
    if dist is not None:
        ax.semilogy(t, dist * 1000, lw=2, color="#c0392b")
        ax.axhline(5, color="k", ls=":", lw=1, label="5 mm tolerance")
        ax.set_ylabel("distance to target (mm, log)")
        ax.legend(fontsize=8)
    if d["sigma"] is not None:
        ax2 = ax.twinx()
        ax2.plot(t, d["sigma"], lw=1.2, color="#2ca02c", alpha=0.7)
        ax2.set_ylabel("min singular value", color="#2ca02c")
    ax.set_xlabel("time (s)")
    ax.set_title("(c) convergence", loc="left", fontweight="bold")
    ax.grid(alpha=0.3, which="both")

    # -- (d) joint angles ----------------------------------------------------
    ax = fig.add_subplot(2, 2, 4)
    for i, j in enumerate(JOINTS):
        ax.plot(t, np.degrees(d["q"][:, i]), lw=1.8, label=j)
    ax.set_xlabel("time (s)"), ax.set_ylabel("joint angle (deg)")
    ax.set_title("(d) joint angles", loc="left", fontweight="bold")
    ax.legend(fontsize=8, ncol=2), ax.grid(alpha=0.3)

    title = f"WX200 Cartesian reach — {os.path.basename(args.csv)}"
    if tgt is not None:
        e = tgt - ee[-1]
        title += (f"   |error| = {1000*np.linalg.norm(e):.2f} mm  "
                  f"(dx {1000*e[0]:+.1f}, dy {1000*e[1]:+.1f}, dz {1000*e[2]:+.1f} mm)")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")

    # -- numbers -------------------------------------------------------------
    print(f"\n{len(t)} samples over {t[-1]-t[0]:.2f} s")
    print(f"start  {ee[0].round(4)}")
    print(f"end    {ee[-1].round(4)}")
    if tgt is not None:
        e = tgt - ee[-1]
        print(f"target {tgt.round(4)}")
        print(f"error  {e.round(5)} m   |e| = {1000*np.linalg.norm(e):.2f} mm")
    print(f"path length {1000*np.linalg.norm(np.diff(ee, axis=0), axis=1).sum():.1f} mm "
          f"(straight line {1000*np.linalg.norm(ee[-1]-ee[0]):.1f} mm)")


if __name__ == "__main__":
    main()
