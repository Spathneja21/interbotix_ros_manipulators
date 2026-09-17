#!/usr/bin/env python3
"""Build a MuJoCo model of the WHOLE LoCoBot -- Create3 base, plate, camera
tower, pan/tilt and the mobile_wx200 arm -- from the robot's own xacro.

    python3 convert/locobot_to_mjcf.py

Writes models/locobot/. Reproducible: re-run to regenerate.

Why a second model
------------------
models/wx200.xml is the STANDALONE arm. Checked against the real robot it is
wrong in three ways that matter for studying hardware trajectories:
  * the waist sits 5.825 mm too high (0.072 vs mobile_wx200's 0.066175)
  * its sleep pose is the standalone arm's [0,-1.88,1.5,0.8,0], not the
    LoCoBot's [0,-1.3,1.55,0.7,0]
  * it has no base, plate or camera tower, so it cannot show the arm hitting
    its own robot -- the main thing a path planner needs to know
This model is expanded with exactly the arguments
interbotix_xslocobot_descriptions/launch/xslocobot_description.launch uses, so
it is the robot's own robot_description, and its FK matches the TF recorded on
the hardware to 0.16 mm when stationary with NO correction.

Differences from the plain URDF import, all scripted below:
  * fusestatic="false" -- every URDF link stays a named body, so frames such as
    locobot/arm_base_link and locobot/ee_gripper_link exist exactly as in TF
  * the Create3 body and bumper are .dae meshes, which MuJoCo cannot load. The
    base's own URDF collision geometry is a cylinder (r 0.164, h 0.06), so the
    visual uses that cylinder too; the bumper becomes a thin ring
  * the finger <mimic> is re-added as an equality constraint
  * contact exclusions are DERIVED from the kinematic tree (see exclusions())
  * home and sleep keyframes, sleep read from the robot's config yaml
  * joint order is identical to /locobot/joint_states, so recorded hardware
    data maps straight onto qpos by name
"""
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# reuse the arm facts and XML helpers from the standalone converter
from urdf_to_mjcf import (ARM_JOINTS, FINGER_HOME, GAINS, JOINT_ARMATURE,  # noqa: E402
                          JOINT_DAMPING, VEL_LIMIT, find_or_make, indent, sub)

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "models", "locobot")
MESHDIR = os.path.join(OUT, "meshes")

WS = "/home/locobot/interbotix_ws/src"
LOCO = os.path.join(WS, "interbotix_ros_rovers", "interbotix_ros_xslocobots")
DESC = os.path.join(LOCO, "interbotix_xslocobot_descriptions")
XACRO = os.path.join(DESC, "urdf", "locobot.urdf.xacro")
ROBOT_YAML = os.path.join(LOCO, "interbotix_xslocobot_control", "config",
                          "locobot_wx200.yaml")
PKG_ROOTS = {"interbotix_xslocobot_descriptions": DESC}

ROBOT = "locobot"
# the arguments xslocobot_description.launch passes for robot_model:=locobot_wx200
XACRO_ARGS = ["robot_model:=locobot_wx200", f"robot_name:={ROBOT}",
              "arm_model:=mobile_wx200", "base_model:=create3",
              "show_lidar:=false", "show_gripper_bar:=true",
              "show_gripper_fingers:=true"]

BLACK = "0.15 0.15 0.15 1"
CREATE3_GREY = "0.55 0.55 0.58 1"


def run_xacro():
    env = dict(os.environ)
    env.setdefault("ROS_PACKAGE_PATH", WS)
    out = subprocess.run(["xacro", XACRO] + XACRO_ARGS, capture_output=True,
                         text=True, env=env)
    if out.returncode != 0:
        sys.exit(f"xacro failed:\n{out.stderr}")
    return out.stdout


def prepare_urdf(text):
    os.makedirs(MESHDIR, exist_ok=True)

    def mesh(m):
        pkg, rel = m.group(1), m.group(2)
        if pkg not in PKG_ROOTS:
            sys.exit(f"unhandled mesh package {pkg}")
        fn = os.path.basename(rel)
        shutil.copy2(os.path.join(PKG_ROOTS[pkg], rel), os.path.join(MESHDIR, fn))
        return fn

    text = re.sub(r"package://([^/\"]+)/([^\"]+?\.stl)", mesh, text)

    # Create3 visuals are COLLADA, which MuJoCo cannot read. Use the base's own
    # URDF collision cylinder for the body, and a thin ring for the bumper.
    text = re.sub(r'<mesh filename="package://irobot_create_description/meshes/'
                  r'body_visual\.dae"[^>]*/>',
                  '<cylinder radius="0.164" length="0.06"/>', text)
    text = re.sub(r'<mesh filename="package://irobot_create_description/meshes/'
                  r'bumper_visual\.dae"[^>]*/>',
                  '<cylinder radius="0.171" length="0.03"/>', text)
    # the bumper collision mesh is only for the Create3's bump sensors
    text = re.sub(r'<collision[^>]*>\s*(<origin[^>]*/>\s*)?<geometry>\s*<mesh '
                  r'filename="package://irobot_create_description/meshes/'
                  r'bumper_collision\.dae"[^>]*/>\s*</geometry>\s*</collision>',
                  "", text)
    if "package://irobot_create_description" in text:
        sys.exit("an irobot_create .dae reference survived the substitution")

    text = re.sub(r'<texture filename="[^"]*"\s*/>', f'<color rgba="{BLACK}"/>',
                  text)
    block = ('<mujoco><compiler meshdir="meshes/" balanceinertia="true" '
             'discardvisual="false" strippath="false" fusestatic="false"/>'
             '</mujoco>')
    text = re.sub(r'(<robot name="[^"]*">)', r"\1\n" + block, text, count=1)
    return text


def decompose_collision_meshes(root, threshold=0.05, max_hulls=32):
    """Replace each concave COLLISION mesh with its convex decomposition.

    MuJoCo collides meshes through their convex hull. For most arm links that is
    close enough, but the camera tower is an open frame the arm folds into and
    the cradle is a U the arm rests in: their hulls fill exactly that space. With
    raw hulls the robot's own sleep pose reports the upper arm 134 mm deep in the
    tower, so any planner built on the model would forbid the rest pose.

    CoACD splits each collision mesh into convex parts; a mesh that is already
    convex comes back as one part and is left alone. Visual geoms are untouched.
    Results are cached in meshes/cvx/, keyed on the source file and parameters.
    """
    import hashlib
    import coacd
    import trimesh
    coacd.set_log_level("error")

    cache = os.path.join(MESHDIR, "cvx")
    os.makedirs(cache, exist_ok=True)
    asset = root.find("asset")
    meshes = {m.get("name"): m for m in asset.findall("mesh")}
    parts_of = {}
    report = []

    for mesh_name, mel in meshes.items():
        used_for_collision = any(
            g.get("mesh") == mesh_name and g.get("contype") != "0"
            for g in root.iter("geom"))
        if not used_for_collision:
            continue
        src = os.path.join(MESHDIR, mel.get("file"))
        digest = hashlib.sha1(open(src, "rb").read()
                              + f"{threshold}/{max_hulls}".encode()).hexdigest()[:12]
        stamp = os.path.join(cache, f"{mesh_name}.{digest}.n")
        if os.path.exists(stamp):
            n = int(open(stamp).read())
        else:
            t = trimesh.load(src, force="mesh")
            parts = coacd.run_coacd(coacd.Mesh(t.vertices, t.faces),
                                    threshold=threshold, max_convex_hull=max_hulls,
                                    preprocess_mode="auto", seed=0)
            n = len(parts)
            for i, (v, f) in enumerate(parts):
                trimesh.Trimesh(v, f).export(os.path.join(cache, f"{mesh_name}_cvx{i}.stl"))
            open(stamp, "w").write(str(n))
        report.append((mesh_name, n))
        if n <= 1:
            continue
        parts_of[mesh_name] = n
        for i in range(n):
            el = ET.SubElement(asset, "mesh", name=f"{mesh_name}_cvx{i}",
                               file=f"cvx/{mesh_name}_cvx{i}.stl")
            if mel.get("scale"):
                el.set("scale", mel.get("scale"))

    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            mesh_name = geom.get("mesh")
            if mesh_name not in parts_of or geom.get("contype") == "0":
                continue
            idx = list(body).index(geom)
            body.remove(geom)
            for i in range(parts_of[mesh_name]):
                g = ET.Element("geom", dict(geom.attrib))
                g.set("mesh", f"{mesh_name}_cvx{i}")
                g.set("name", f"{geom.get('name', mesh_name)}_cvx{i}")
                g.set("group", "3")          # hidden by default; visual mesh shows
                body.insert(idx + i, g)
    return report


def exclusions(model):
    """Body pairs whose contacts are structural, not collisions.

    With fusestatic off, a rigid assembly is many separate bodies (plate,
    battery, camera tower, arm stand ... all welded together), and MuJoCo's
    automatic parent/child filter only covers DIRECT parents, and never the
    world body. Deriving the exclusions from the tree instead of from a contact
    sweep keeps real collisions visible:

      1. bodies in the same rigid group (same nearest moving ancestor) can never
         move relative to each other -- their contacts carry no information
      2. a moving body against the rigid group it is mounted on -- the waist on
         the arm stand, a wheel in the chassis, pan on the camera tower

    Everything else (the arm against the tower, the plate, the base, the floor)
    is left enabled on purpose.
    """
    def rigid_root(b):
        while b != 0 and model.body_dofnum[b] == 0:
            b = model.body_parentid[b]
        return b

    collide = {model.geom_bodyid[g] for g in range(model.ngeom)
               if model.geom_contype[g] or model.geom_conaffinity[g]}
    collide.add(0)          # world: the floor lives there once scene.xml adds it
    root = {b: rigid_root(b) for b in range(model.nbody)}
    name = lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "world"

    pairs = set()
    bodies = sorted(collide)
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            if root[a] == root[b]:
                pairs.add((a, b))
    for a in bodies:
        if model.body_dofnum[a] > 0:
            mount = root[model.body_parentid[a]]
            for b in bodies:
                if b != a and root[b] == mount:
                    pairs.add(tuple(sorted((a, b))))
    return sorted((name(a), name(b)) for a, b in pairs)


def patch(path, compiled):
    tree = ET.parse(path)
    root = tree.getroot()
    root.set("model", "locobot_wx200")

    comp = find_or_make(root, "compiler", 0)
    comp.set("angle", "radian")
    comp.set("autolimits", "true")
    opt = find_or_make(root, "option", 1)
    opt.set("timestep", "0.002")
    opt.set("integrator", "implicitfast")

    for joint in root.iter("joint"):
        n = joint.get("name")
        if n in JOINT_DAMPING:
            joint.set("damping", str(JOINT_DAMPING[n]))
            joint.set("armature", str(JOINT_ARMATURE))
        elif n in ("left_finger", "right_finger"):
            joint.set("damping", "5.0")
        elif n in ("gripper", "pan", "tilt"):
            joint.set("damping", "0.05")
        elif n and n.endswith("wheel_joint"):
            joint.set("damping", "0.01")

    # geoms: readable names, and the Create3 shell a lighter grey
    for body in root.iter("body"):
        short = body.get("name", "b").split("/")[-1]
        for k, geom in enumerate(body.findall("geom")):
            if not geom.get("name"):
                geom.set("name", f"{short}_g{k}")
            if short in ("base_link", "bump_front_center") and geom.get("type") == "cylinder":
                geom.set("rgba", CREATE3_GREY)

    report = decompose_collision_meshes(root)
    split = [(n, k) for n, k in report if k > 1]
    print(f"      convex decomposition: {len(split)} of {len(report)} collision "
          f"meshes were concave")
    for n, k in split:
        print(f"        {n:<42} -> {k} convex parts")

    bodies = {b.get("name"): b for b in root.iter("body")}
    # Sites on the bodies TF knows by the same name. Bodies already carry the
    # frames; the sites are what mj_jacSite and the replay tools address.
    for body_name, site, rgba in ((f"{ROBOT}/ee_gripper_link", "ee_gripper", "1 0 0 0.6"),
                                  (f"{ROBOT}/arm_base_link", "arm_base", "0 0 1 0.4")):
        if body_name not in bodies:
            sys.exit(f"body {body_name} missing -- did fusestatic get applied?")
        sub(bodies[body_name], "site", name=site, pos="0 0 0", size="0.008",
            rgba=rgba, group="3")

    eq = find_or_make(root, "equality")
    sub(eq, "joint", name="finger_mimic", joint1="right_finger",
        joint2="left_finger", polycoef="0 -1 0 0 0", solimp="0.99 0.999 0.001")

    contact = find_or_make(root, "contact")
    excl = exclusions(compiled)
    for a, b in excl:
        sub(contact, "exclude", body1=a, body2=b)

    # keyframes, in /locobot/joint_states order
    names = [mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(compiled.njnt)]
    sleep_cfg = yaml.safe_load(open(ROBOT_YAML))["sleep_positions"][:5]

    def key(arm):
        q = dict(zip(names, [0.0] * len(names)))
        q.update(dict(zip(ARM_JOINTS, arm)))
        q["left_finger"], q["right_finger"] = FINGER_HOME, -FINGER_HOME
        return " ".join(f"{q[n]:g}" for n in names)

    kf = find_or_make(root, "keyframe")
    sub(kf, "key", name="home", qpos=key([0.0] * 5))
    sub(kf, "key", name="sleep", qpos=key(sleep_cfg))

    indent(root)
    tree.write(path, encoding="utf-8", xml_declaration=False)
    return names, sleep_cfg, len(excl)


def write_actuators():
    """intvelocity / velocity variants for the arm, same gains as the
    standalone model (identical link inertias). pan, tilt and the fingers get
    position servos so a dynamic replay holds them where they were recorded."""
    for mode in ("intvelocity", "velocity"):
        root = ET.Element("mujoco", model=f"locobot_{mode}")
        root.append(ET.Comment(" GENERATED by convert/locobot_to_mjcf.py -- do not edit "))
        act = ET.SubElement(root, "actuator")
        m = mujoco.MjModel.from_xml_path(os.path.join(OUT, "locobot_wx200.xml"))
        for j in ARM_JOINTS:
            g = GAINS[j]
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            lo, hi = m.jnt_range[jid]
            common = dict(name=f"{j}_v", joint=j,
                          ctrlrange=f"{-VEL_LIMIT:.6f} {VEL_LIMIT:.6f}")
            if mode == "intvelocity":
                ET.SubElement(act, "intvelocity", dict(
                    common, kp=f"{g['kp']:.2f}", kv=f"{g['kv']:.3f}",
                    actrange=f"{lo:.5f} {hi:.5f}"))
            else:
                ET.SubElement(act, "velocity", dict(common, kv=f"{g['kv_vel']:.3f}"))
        for j, kp in (("pan", "2"), ("tilt", "2"), ("left_finger", "50")):
            ET.SubElement(act, "position", dict(name=f"{j}_p", joint=j, kp=kp,
                                                dampratio="1"))
        indent(root)
        ET.ElementTree(root).write(os.path.join(OUT, f"actuators_{mode}.xml"),
                                   encoding="utf-8", xml_declaration=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    prep = os.path.join(OUT, "_locobot_mj.urdf")
    out = os.path.join(OUT, "locobot_wx200.xml")

    print("1/5 xacro (xslocobot_description.launch arguments)")
    text = run_xacro()
    print("2/5 staging meshes, replacing Create3 COLLADA with primitives")
    open(prep, "w").write(prepare_urdf(text))
    print("3/5 compiling to mjcf")
    compiled = mujoco.MjModel.from_xml_path(prep)
    mujoco.mj_saveLastXML(out, compiled)
    os.remove(prep)
    print("4/5 patching")
    names, sleep_cfg, nexcl = patch(out, compiled)
    print("5/5 actuators")
    write_actuators()

    m = mujoco.MjModel.from_xml_path(out)
    print(f"\nwrote {out}")
    print(f"  nq={m.nq} nbody={m.nbody} ngeom={m.ngeom} nkey={m.nkey} "
          f"exclusions={nexcl}")
    print(f"  qpos order: {names}")
    print(f"  sleep (from {os.path.basename(ROBOT_YAML)}): {sleep_cfg}")


if __name__ == "__main__":
    main()
