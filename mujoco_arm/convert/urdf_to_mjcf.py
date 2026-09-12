#!/usr/bin/env python3
"""Build the MuJoCo WX200 model from the vendored Interbotix xacro. Reproducible:
delete models/wx200.xml and re-run to get a byte-identical result.

    python3 convert/urdf_to_mjcf.py

Pipeline:
  1. xacro -> plain URDF (use_world_frame:=true, so base_link is welded to world)
  2. copy the STL meshes, rewrite package:// refs to bare filenames
  3. inject the <mujoco><compiler/></mujoco> block the URDF importer requires
  4. compile with MuJoCo, save as MJCF
  5. patch the MJCF for everything the URDF format cannot express (see PATCHES)

Step 5 is scripted rather than hand-edited so that regenerating never silently
drops a fix.
"""
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # mujoco_arm/
MODELS = os.path.join(ROOT, "models")
MESHDIR = os.path.join(MODELS, "meshes")

XSARM_DESC = os.path.normpath(os.path.join(
    ROOT, "..", "interbotix_ros_xsarms", "interbotix_xsarm_descriptions"))
XACRO = os.path.join(XSARM_DESC, "urdf", "wx200.urdf.xacro")
SRC_MESHES = os.path.join(XSARM_DESC, "meshes", "wx200_meshes")

ROBOT = "wx200"

# --- facts taken from the Interbotix configs, see plan.md section 1 ----------

# interbotix_xsarm_control/config/wx200.yaml : joint_order
ARM_JOINTS = ["waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate"]

# wx200.yaml : sleep_positions (arm joints only)
SLEEP = [0.0, -1.88, 1.5, 0.8, 0.0]

# The URDF chain from gripper_link to ee_gripper_link is all fixed joints, so
# MuJoCo merges it away. Sum of those origins gives the site offset:
#   ee_arm 0.043 + gripper_bar 0 + ee_bar 0.023 + ee_gripper 0.027575
EE_SITE_OFFSET = 0.043 + 0.0 + 0.023 + 0.027575   # = 0.093575 m along +x

# Half-open finger width, inside the URDF limit [0.015, 0.037].
FINGER_HOME = 0.021

# Seeded from the Gazebo effort-controller gains in
# interbotix_xsarm_gazebo/config/trajectory_controllers/wx200_trajectory_controllers.yaml
# only as a *relative* scaling: the shoulder needs far more than the wrist.
JOINT_DAMPING = {
    "waist": 0.1,
    "shoulder": 0.5,
    "elbow": 0.3,
    "wrist_angle": 0.1,
    "wrist_rotate": 0.05,
}
JOINT_ARMATURE = 0.01

# URDF <limit velocity="${pi}"> on every arm joint.
VEL_LIMIT = 3.141592653589793

# Velocity-servo gains, DERIVED not tuned: regenerate with
# `python3 tools/compute_gains.py` and paste. kp/kv place a critically damped
# 40 rad/s closed loop on the worst-case joint inertia; kv_vel gives the plain
# <velocity> servo a matched 1/40 s first-order response.
GAINS = {
    "waist":        dict(kp=173.25, kv=8.663, kv_vel=4.331),
    "shoulder":     dict(kp=171.44, kv=8.572, kv_vel=4.286),
    "elbow":        dict(kp=58.28,  kv=2.914, kv_vel=1.457),
    "wrist_angle":  dict(kp=24.11,  kv=1.206, kv_vel=0.603),
    "wrist_rotate": dict(kp=20.91,  kv=1.045, kv_vel=0.523),
}

INTERBOTIX_BLACK = "0.15 0.15 0.15 1"


def run_xacro(dst):
    cmd = ["xacro", XACRO,
           f"robot_name:={ROBOT}",
           "use_world_frame:=true",
           "show_gripper_bar:=true",
           "show_gripper_fingers:=true",
           "show_ar_tag:=false"]
    env = dict(os.environ)
    # xacro lives in the ROS install; make sure its python deps resolve
    env.setdefault("PYTHONPATH", "/opt/ros/noetic/lib/python3/dist-packages")
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if out.returncode != 0:
        sys.exit(f"xacro failed:\n{out.stderr}")
    with open(dst, "w") as f:
        f.write(out.stdout)


def stage_meshes():
    os.makedirs(MESHDIR, exist_ok=True)
    for fn in sorted(os.listdir(SRC_MESHES)):
        if fn.endswith(".stl"):
            shutil.copy2(os.path.join(SRC_MESHES, fn), os.path.join(MESHDIR, fn))


def prepare_urdf(src, dst):
    """Rewrite mesh paths and inject the <mujoco> compiler block."""
    text = open(src).read()

    # package://interbotix_xsarm_descriptions/meshes/[wx200_meshes/]foo.stl -> foo.stl
    text = re.sub(r"package://interbotix_xsarm_descriptions/meshes/(?:wx200_meshes/)?",
                  "", text)

    # The PNG texture has no UVs on these STLs; drop it and use a flat colour so
    # the model does not render with garbage texture coordinates.
    text = re.sub(r'<texture\s+filename="interbotix_black\.png"\s*/>',
                  f'<color rgba="{INTERBOTIX_BLACK}"/>', text)

    block = ('  <mujoco>\n'
             '    <compiler meshdir="meshes/" balanceinertia="true"\n'
             '              discardvisual="false" strippath="false"/>\n'
             '  </mujoco>\n')
    text = text.replace(f'<robot name="{ROBOT}">',
                        f'<robot name="{ROBOT}">\n{block}', 1)

    with open(dst, "w") as f:
        f.write(text)


# --- MJCF patches -----------------------------------------------------------

def sub(parent, tag, **kw):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in kw.items()})


def find_or_make(root, tag, index=None):
    el = root.find(tag)
    if el is None:
        el = ET.Element(tag)
        root.insert(len(root) if index is None else index, el)
    return el


def patch_mjcf(path):
    tree = ET.parse(path)
    root = tree.getroot()
    root.set("model", "wx200")

    # -- compiler / solver options ------------------------------------------
    comp = find_or_make(root, "compiler", 0)
    comp.set("angle", "radian")
    comp.set("autolimits", "true")

    opt = find_or_make(root, "option", 1)
    opt.set("timestep", "0.002")
    # implicitfast integrates joint damping stably at this timestep, which
    # matters once velocity servos are added in phase 2.
    opt.set("integrator", "implicitfast")

    # -- joint damping + armature -------------------------------------------
    # Without these the arm is numerically twitchy and velocity control rings.
    bodies = {}
    for body in root.iter("body"):
        bodies[body.get("name")] = body
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name in JOINT_DAMPING:
            joint.set("damping", str(JOINT_DAMPING[name]))
            joint.set("armature", str(JOINT_ARMATURE))
        elif name in ("left_finger", "right_finger"):
            joint.set("damping", "5.0")
        elif name == "gripper":
            joint.set("damping", "0.05")

    # -- name the unnamed geoms so contacts are debuggable -------------------
    # worldbody is included: base_link was merged into it by the fixed
    # world->base_link joint, so its two geoms live there and would stay
    # nameless (and unreadable in contact dumps) otherwise.
    containers = [(root.find("worldbody"), "base_link")]
    containers += [(b, b.get("name", "b").split("/")[-1]) for b in root.iter("body")]
    for container, short in containers:
        if container is None:
            continue
        for n, geom in enumerate(container.findall("geom")):
            if not geom.get("name"):
                geom.set("name", f"{short}_g{n}")

    # -- reference frames ----------------------------------------------------
    # base_link was merged into worldbody by the fixed world->base_link joint,
    # so the base_link frame is exactly the world frame. This site makes that
    # explicit and gives us the frame reach_and_log.py logs in.
    wb = root.find("worldbody")
    sub(wb, "site", name="base_link", pos="0 0 0", size="0.01",
        rgba="0 0 1 0.4", group="3")

    # The whole gripper_link -> ee_gripper_link chain is fixed joints, so it was
    # merged away; re-add the EE frame as a site. Phase 4 takes the Cartesian
    # Jacobian here (mj_jacSite) and the logger reads its position.
    grip = bodies.get(f"{ROBOT}/gripper_link")
    if grip is None:
        sys.exit("could not find gripper_link body to attach the EE site to")
    sub(grip, "site", name="ee_gripper", pos=f"{EE_SITE_OFFSET} 0 0",
        size="0.008", rgba="1 0 0 0.6", group="3")

    # -- gripper mimic -------------------------------------------------------
    # URDF <mimic joint="left_finger" multiplier="-1"/> is dropped by the
    # importer. polycoef c0..c4 means: right = c0 + c1*left + ...
    eq = find_or_make(root, "equality")
    sub(eq, "joint", name="finger_mimic", joint1="right_finger",
        joint2="left_finger", polycoef="0 -1 0 0 0", solimp="0.99 0.999 0.001")

    # -- contact exclusions --------------------------------------------------
    # MuJoCo auto-excludes contacts between a body and its parent, but NOT when
    # the parent is the world body. base_link was merged into world by the fixed
    # world->base_link joint, so base_link/shoulder_link -- which are adjacent
    # across the waist joint and always touching -- are not filtered. Exclude
    # them explicitly, or every step reports a bogus contact.
    contact = find_or_make(root, "contact")
    sub(contact, "exclude", name="base_shoulder",
        body1="world", body2=f"{ROBOT}/shoulder_link")

    # -- keyframes -----------------------------------------------------------
    # qpos order: waist shoulder elbow wrist_angle wrist_rotate gripper
    #             left_finger right_finger
    # Note qpos0 defaults to 0 for the fingers, which is OUTSIDE their
    # [0.015, 0.037] limit, so a model reset without a keyframe starts the
    # fingers jammed against a limit. Always reset to one of these.
    kf = find_or_make(root, "keyframe")
    home = [0, 0, 0, 0, 0, 0, FINGER_HOME, -FINGER_HOME]
    sleep = SLEEP + [0, FINGER_HOME, -FINGER_HOME]
    sub(kf, "key", name="home", qpos=" ".join(str(v) for v in home))
    sub(kf, "key", name="sleep", qpos=" ".join(str(v) for v in sleep))

    indent(root)
    tree.write(path, encoding="utf-8", xml_declaration=False)


def write_actuators(mjcf_path):
    """Emit the two velocity-actuator variants as includable fragments.

    They are separate files rather than a runtime flag so that each variant is
    a real, inspectable model you can open in `simulate` -- and so the actuator
    parameters are diffable when we retune.
    """
    m = mujoco.MjModel.from_xml_path(mjcf_path)
    out = {}

    for mode in ("intvelocity", "velocity"):
        root = ET.Element("mujoco", model=f"wx200_{mode}")
        root.append(ET.Comment(
            f" GENERATED by convert/urdf_to_mjcf.py -- do not edit.\n"
            f"       {mode} actuators for the 5 arm joints, in wx200.yaml order.\n"
            f"       ctrlrange is the URDF velocity limit (+-pi rad/s).\n"
            f"       Force limiting comes from actuatorfrcrange on the joints,\n"
            f"       which the URDF importer already set from the effort limits. "))
        act = ET.SubElement(root, "actuator")

        for j in ARM_JOINTS:
            g = GAINS[j]
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            lo, hi = m.jnt_range[jid]
            common = dict(name=f"{j}_v", joint=j,
                          ctrlrange=f"{-VEL_LIMIT:.6f} {VEL_LIMIT:.6f}")
            if mode == "intvelocity":
                # actrange clamps the INTEGRATED position setpoint to the joint
                # limits, so the servo cannot wind up past a hard stop.
                ET.SubElement(act, "intvelocity", dict(
                    common, kp=f"{g['kp']:.2f}", kv=f"{g['kv']:.3f}",
                    actrange=f"{lo:.5f} {hi:.5f}"))
            else:
                ET.SubElement(act, "velocity", dict(
                    common, kv=f"{g['kv_vel']:.3f}"))

        path = os.path.join(MODELS, f"actuators_{mode}.xml")
        indent(root)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)
        out[mode] = path
    return out


def indent(el, level=0):
    pad = "\n" + "  " * level
    if len(el):
        if not (el.text or "").strip():
            el.text = pad + "  "
        for child in el:
            indent(child, level + 1)
        if not (el[-1].tail or "").strip():
            el[-1].tail = pad
    if level and not (el.tail or "").strip():
        el.tail = pad


def main():
    os.makedirs(MODELS, exist_ok=True)
    raw_urdf = os.path.join(MODELS, "_wx200_raw.urdf")
    prep_urdf = os.path.join(MODELS, "_wx200_mj.urdf")
    out = os.path.join(MODELS, "wx200.xml")

    print("1/6 xacro -> urdf")
    run_xacro(raw_urdf)

    print("2/6 staging meshes")
    stage_meshes()

    print("3/6 preparing urdf for the mujoco importer")
    prepare_urdf(raw_urdf, prep_urdf)

    print("4/6 compiling to mjcf")
    model = mujoco.MjModel.from_xml_path(prep_urdf)
    mujoco.mj_saveLastXML(out, model)

    print("5/6 patching mjcf")
    patch_mjcf(out)

    print("6/6 writing actuator variants")
    for mode, p in write_actuators(out).items():
        print(f"      {os.path.basename(p)}")

    # the intermediates are only useful for debugging a failed import
    for f in (raw_urdf, prep_urdf):
        os.remove(f)

    m = mujoco.MjModel.from_xml_path(out)
    print(f"\nwrote {out}")
    print(f"  nq={m.nq} nv={m.nv} nbody={m.nbody} ngeom={m.ngeom} "
          f"nsite={m.nsite} neq={m.neq} nkey={m.nkey}")


if __name__ == "__main__":
    main()
