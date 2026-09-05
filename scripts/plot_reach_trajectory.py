#!/usr/bin/env python3
"""Plot joint angles vs time and the 3D end-effector path from reach_and_log.py's CSV.

Usage: python3 plot_reach_trajectory.py reach_trajectory.csv
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "reach_trajectory.csv"
df = pd.read_csv(path)
t = df["time"] - df["time"].iloc[0]
joint_cols = [c for c in df.columns if c not in ("time", "ee_x", "ee_y", "ee_z")]

fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_subplot(1, 2, 1)
for j in joint_cols:
    ax1.plot(t, df[j], label=j)
ax1.set_xlabel("time (s)")
ax1.set_ylabel("joint angle (rad)")
ax1.set_title("Joint angles vs time")
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot(df["ee_x"], df["ee_y"], df["ee_z"], marker="o", markersize=2)
ax2.scatter(df["ee_x"].iloc[0], df["ee_y"].iloc[0], df["ee_z"].iloc[0], color="green", label="start")
ax2.scatter(df["ee_x"].iloc[-1], df["ee_y"].iloc[-1], df["ee_z"].iloc[-1], color="red", label="end")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_zlabel("z (m)")
ax2.set_title("End-effector path")
ax2.legend()

plt.tight_layout()
plt.savefig(path.replace(".csv", ".png"))
plt.show()
