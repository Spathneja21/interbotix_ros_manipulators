#!/usr/bin/env python3
"""Standalone MuJoCo physics viewer for the px100 arm, no ROS involved.

Drive the joints from the viewer's own Control panel (sliders, one per
actuator: waist/shoulder/elbow/wrist_angle/left_finger/right_finger) and
watch real physics (gravity, inertia, joint limits, self-collision) respond.

Usage:
    python3 view_px100.py
"""
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("px100_mjcf.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
