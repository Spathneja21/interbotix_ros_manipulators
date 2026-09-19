#!/usr/bin/env python3
"""Plot joint angles vs time and the 3D end-effector path from reach_and_log.py's CSV.

Usage: python3 plot_reach_trajectory.py reach_trajectory.csv
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers the '3d' projection

path = sys.argv[1] if len(sys.argv) > 1 else "reach_trajectory.csv"
df = pd.read_csv(path)
# .to_numpy() throughout: the system matplotlib indexes with x[:, None], which
# newer pandas rejects on a Series
t = (df["time"] - df["time"].iloc[0]).to_numpy()
joint_cols = [c for c in df.columns if c not in ("time", "ee_x", "ee_y", "ee_z")]
ex, ey, ez = (df[c].to_numpy() for c in ("ee_x", "ee_y", "ee_z"))

fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_subplot(1, 2, 1)
for j in joint_cols:
    ax1.plot(t, df[j].to_numpy(), label=j)
ax1.set_xlabel("time (s)")
ax1.set_ylabel("joint angle (rad)")
ax1.set_title("Joint angles vs time")
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot(ex, ey, ez, marker="o", markersize=2)
ax2.scatter(ex[0], ey[0], ez[0], color="green", label="start")
ax2.scatter(ex[-1], ey[-1], ez[-1], color="red", label="end")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_zlabel("z (m)")
ax2.set_title("End-effector path")
ax2.legend()

plt.tight_layout()
plt.savefig(path.replace(".csv", ".png"))
plt.show()
