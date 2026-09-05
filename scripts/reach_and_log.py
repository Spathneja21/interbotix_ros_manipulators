#!/usr/bin/env python3
"""Plan a path to a 3D goal via MoveIt, execute it via arm_controller directly,
logging joint angles + EE position vs time.

Usage:
  roslaunch interbotix_xsarm_gazebo xsarm_gazebo.launch robot_model:=px100 use_trajectory_controllers:=true
  roslaunch interbotix_xsarm_moveit move_group.launch robot_model:=px100 robot_name:=px100 dof:=4 allow_trajectory_execution:=false
  python3 reach_and_log.py <x> <y> <z> [out.csv]

Positions are in meters, in the px100/base_link frame.

Note: execution goes straight to arm_controller's FollowJointTrajectory action
instead of MoveIt's own execution path, because move_group's trajectory
execution manager calls /px100/controller_manager/list_controllers at startup,
and that service hangs indefinitely with this Gazebo/ros_control combo
(reproduced directly with `rosservice call`). Planning alone doesn't need it.
"""
import sys
import csv
import rospy
import tf2_ros
import actionlib
import moveit_commander
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

robot_name = "px100"
x, y, z = (float(v) for v in sys.argv[1:4])
out_path = sys.argv[4] if len(sys.argv) > 4 else "reach_trajectory.csv"

print("checkpoint: init_node", flush=True)
rospy.init_node("reach_and_log", anonymous=True)
print("checkpoint: roscpp_initialize", flush=True)
moveit_commander.roscpp_initialize(sys.argv)
print("checkpoint: constructing MoveGroupCommander", flush=True)
group = moveit_commander.MoveGroupCommander(
    "interbotix_arm", robot_description=f"{robot_name}/robot_description", ns=f"/{robot_name}"
)
print("checkpoint: MoveGroupCommander ready", flush=True)

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

rows = []
joint_names = []


def on_joint_state(msg):
    joint_names[:] = msg.name
    try:
        t = tf_buffer.lookup_transform(f"{robot_name}/base_link", f"{robot_name}/ee_gripper_link", rospy.Time(0))
    except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
        return
    p = t.transform.translation
    rows.append([rospy.get_time(), *msg.position, p.x, p.y, p.z])


sub = rospy.Subscriber(f"/{robot_name}/joint_states", JointState, on_joint_state)

print("checkpoint: get_current_pose", flush=True)
pose = group.get_current_pose().pose
print(f"checkpoint: got pose {pose.position}", flush=True)
pose.position.x, pose.position.y, pose.position.z = x, y, z
group.set_pose_target(pose)
rospy.sleep(0.5)  # let a few tf/joint_state samples land before moving

print("checkpoint: planning", flush=True)
success, plan, _, error_code = group.plan()
print(f"checkpoint: plan done, success={success}", flush=True)
if not success:
    print(f"FAILED to plan: {error_code}")
    sys.exit(1)

client = actionlib.SimpleActionClient(f"/{robot_name}/arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
print("checkpoint: waiting for arm_controller action server", flush=True)
client.wait_for_server()
print("checkpoint: action server found, sending goal", flush=True)
goal = FollowJointTrajectoryGoal(trajectory=plan.joint_trajectory)
client.send_goal(goal)
client.wait_for_result()
print("checkpoint: execution done", flush=True)

group.clear_pose_targets()
rospy.sleep(0.3)
sub.unregister()

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time", *joint_names, "ee_x", "ee_y", "ee_z"])
    w.writerows(rows)

print(f"{'reached' if success else 'FAILED to reach'} target, {len(rows)} samples logged to {out_path}")
