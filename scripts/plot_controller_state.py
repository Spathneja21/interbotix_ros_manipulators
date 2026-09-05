#!/usr/bin/env python3
"""Plot desired vs actual joint position/error from a JointTrajectoryControllerState CSV.

Usage: python3 plot_controller_state.py arm_controller_state.csv [joint_index]
Record the CSV first with:
  rostopic echo -p /px100/arm_controller/state > arm_controller_state.csv
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1]
joint = int(sys.argv[2]) if len(sys.argv) > 2 else 0

df = pd.read_csv(path)
t = df["field.header.stamp"] / 1e9
t -= t.iloc[0]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
ax1.plot(t, df[f"field.desired.positions{joint}"], label="desired")
ax1.plot(t, df[f"field.actual.positions{joint}"], label="actual")
ax1.set_ylabel("position (rad)")
ax1.legend()
ax1.set_title(f"Joint {joint} step response")

ax2.plot(t, df[f"field.error.positions{joint}"], color="red")
ax2.set_ylabel("position error (rad)")
ax2.set_xlabel("time (s)")

plt.tight_layout()
plt.savefig(path.replace(".csv", f"_joint{joint}.png"))
plt.show()
