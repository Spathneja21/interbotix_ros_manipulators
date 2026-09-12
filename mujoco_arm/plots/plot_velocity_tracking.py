#!/usr/bin/env python3
"""Commanded vs actual joint velocity, the tracking error, and the resulting
joint angles.

This is scripts/plot_controller_state.py retargeted from position to velocity:
same idea (desired on top, error below), but reading a trace produced by
velocity_controller.py instead of a rostopic CSV dump.

    python3 plots/plot_velocity_tracking.py run.csv [out.png]

or from python:

    from plots.plot_velocity_tracking import plot_trace
    plot_trace(ctrl.trace, "run.png")
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_sim import ARM_JOINTS                       # noqa: E402

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def _figure(t, cmd, act, q, title):
    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    for i, j in enumerate(ARM_JOINTS):
        moved = np.abs(cmd[:, i]).max() > 1e-9 or np.ptp(q[:, i]) > 1e-4
        alpha = 1.0 if moved else 0.25
        ax[0].plot(t, cmd[:, i], color=COLORS[i], ls="--", lw=1.2, alpha=alpha)
        ax[0].plot(t, act[:, i], color=COLORS[i], lw=1.8, alpha=alpha, label=j)
        ax[1].plot(t, act[:, i] - cmd[:, i], color=COLORS[i], lw=1.5,
                   alpha=alpha, label=j)
        ax[2].plot(t, np.degrees(q[:, i]), color=COLORS[i], lw=1.8, alpha=alpha,
                   label=j)

    ax[0].set_ylabel("joint velocity (rad/s)")
    ax[0].set_title("commanded (dashed) vs actual (solid)", loc="left",
                    fontweight="bold")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("velocity error (rad/s)")
    ax[1].set_title("tracking error, actual − commanded", loc="left",
                    fontweight="bold")
    ax[2].set_ylabel("joint angle (deg)")
    ax[2].set_xlabel("time (s)")
    ax[2].set_title("resulting joint angles", loc="left", fontweight="bold")

    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8, ncol=5, loc="best")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_trace(trace, out_path, title="WX200 joint-velocity tracking"):
    fig = _figure(trace.t, trace.cmd, trace.qd, trace.q, title)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_csv(csv_path, out_path=None):
    out_path = out_path or csv_path.replace(".csv", ".png")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["time"]) for r in rows])
    grab = lambda p: np.array([[float(r[f"{p}{j}"]) for j in ARM_JOINTS]
                               for r in rows])
    fig = _figure(t, grab("qd_cmd_"), grab("qd_act_"),
                  np.array([[float(r[j]) for j in ARM_JOINTS] for r in rows]),
                  f"WX200 joint-velocity tracking — {os.path.basename(csv_path)}")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("wrote", plot_csv(sys.argv[1],
                            sys.argv[2] if len(sys.argv) > 2 else None))
