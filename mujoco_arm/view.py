#!/usr/bin/env python3
"""Open the WX200 scene in the interactive MuJoCo viewer.

    python3 view.py            # start in the home pose
    python3 view.py --key sleep

Needs a display (we have :0 on the LoCoBot). There are no actuators on the model
yet -- that is phase 2 -- so the arm will simply fall under gravity when the
simulation runs. Press Space to pause/unpause.
"""
import argparse
import os

import mujoco
import mujoco.viewer

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "scene.xml")

ap = argparse.ArgumentParser()
ap.add_argument("--key", default="home", choices=["home", "sleep"])
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0 if args.key == "home" else 1)
mujoco.mj_forward(m, d)

print(f"{args.key} pose | EE at {d.site('ee_gripper').xpos.round(4)} (base_link frame)")
print("no actuators yet (phase 2) -- the arm will fall when unpaused")
mujoco.viewer.launch(m, d)
