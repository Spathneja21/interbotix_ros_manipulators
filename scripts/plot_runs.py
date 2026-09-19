#!/usr/bin/env python3
"""Overlay several reach_and_log.py runs: joint angles vs time + 3D end-effector paths.

Usage: python3 plot_runs.py test/test2_f.csv test/test2_b.csv [-o out.png]

Each run's clock is shifted to start at t=0 so the runs line up for comparison.
Joints are distinguished by colour, runs by line style.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers the '3d' projection

args = sys.argv[1:]
out_path = None
if "-o" in args:
    i = args.index("-o")
    out_path = args[i + 1]
    args = args[:i] + args[i + 2:]
if not args:
    sys.exit(__doc__)

paths = args
if out_path is None:
    out_path = os.path.join(os.path.dirname(paths[0]) or ".", "comparison.png")

styles = ["-", "--", "-.", ":"]
run_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

frames = []
for p in paths:
    df = pd.read_csv(p)
    df["t"] = (df["time"] - df["time"].iloc[0]).to_numpy()
    frames.append((os.path.splitext(os.path.basename(p))[0], df))

joint_cols = [c for c in frames[0][1].columns if c not in ("time", "t", "ee_x", "ee_y", "ee_z")]
joint_colors = dict(zip(joint_cols, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]))

fig = plt.figure(figsize=(13, 5.5))

ax1 = fig.add_subplot(1, 2, 1)
for k, (label, df) in enumerate(frames):
    for j in joint_cols:
        ax1.plot(df["t"].to_numpy(), df[j].to_numpy(), styles[k % len(styles)],
                 color=joint_colors[j], linewidth=1.4)
ax1.set_xlabel("time (s)")
ax1.set_ylabel("joint angle (rad)")
ax1.set_title("Joint angles vs time")
ax1.grid(alpha=0.3)
ax1.legend(handles=[Line2D([], [], color=joint_colors[j], label=j) for j in joint_cols]
                   + [Line2D([], [], color="black", linestyle=styles[k % len(styles)], label=lbl)
                      for k, (lbl, _) in enumerate(frames)],
           fontsize=8, ncol=2)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
for k, (label, df) in enumerate(frames):
    ex, ey, ez = (df[c].to_numpy() for c in ("ee_x", "ee_y", "ee_z"))
    ax2.plot(ex, ey, ez, color=run_colors[k % len(run_colors)], linewidth=1.6, label=label)
    ax2.scatter(ex[0], ey[0], ez[0], color=run_colors[k % len(run_colors)], marker="o", s=45)
    ax2.scatter(ex[-1], ey[-1], ez[-1], color=run_colors[k % len(run_colors)], marker="X", s=55)
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_zlabel("z (m)")
ax2.set_title("End-effector paths  (o = start, X = end)")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(out_path, dpi=130)
print(f"saved {out_path}")
plt.show()
