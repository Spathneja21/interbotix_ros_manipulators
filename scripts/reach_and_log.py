#!/usr/bin/env python3
"""Reach a 3D goal via a plain compute_ik service call + arm_controller action,
logging joint angles + EE position vs time.

Usage:
  roslaunch interbotix_xsarm_gazebo xsarm_gazebo.launch robot_model:=px100 use_trajectory_controllers:=true
  ROS_NAMESPACE=px100 roslaunch interbotix_xsarm_moveit move_group.launch robot_model:=px100 robot_name:=px100 dof:=4 allow_trajectory_execution:=false
  python3 reach_and_log.py <x> <y> <z> [duration_s] [out.csv]

Positions are in meters, in the px100/base_link frame.

Note: this deliberately avoids moveit_commander.MoveGroupCommander. Constructing
it in a fresh client process reproducibly hangs forever right after "Loading
robot model" (confirmed with move_group's official moveit_commander_cmdline.py
tool too, and with checkpoint prints narrowing it to inside the C++
MoveGroupInterface constructor, before any topic subscriptions happen) even
though move_group itself is fully up and serving requests. The compute_ik
service and the arm_controller action are both plain, already-proven-working
ROS interfaces that MoveGroupInterface itself is a (broken, in this
environment) wrapper around, so we call them directly instead.
"""
import os
import sys
import csv
import math
import rospy
import tf2_ros
import actionlib
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import PositionIKRequest
from geometry_msgs.msg import PoseStamped
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

robot_name = "px100"
group_name = "interbotix_arm"
arm_joints = ["waist", "shoulder", "elbow", "wrist_angle"]
x, y, z = (float(v) for v in sys.argv[1:4])
duration = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
out_path = sys.argv[5] if len(sys.argv) > 5 else "reach_trajectory.csv"
if not os.path.dirname(out_path):  # bare filename -> repo's test/ dir, whatever the cwd
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test")
    os.makedirs(test_dir, exist_ok=True)
    out_path = os.path.join(test_dir, out_path)

rospy.init_node("reach_and_log", anonymous=True)

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

rows = []


def on_joint_state(msg):
    try:
        t = tf_buffer.lookup_transform(f"{robot_name}/base_link", f"{robot_name}/ee_gripper_link", rospy.Time(0))
    except tf2_ros.TransformException:
        return  # tf tree not fully connected yet during the first few samples
    p = t.transform.translation
    positions = [msg.position[msg.name.index(j)] for j in arm_joints]
    rows.append([rospy.get_time(), *positions, p.x, p.y, p.z])


sub = rospy.Subscriber(f"/{robot_name}/joint_states", JointState, on_joint_state)

print("checkpoint: waiting for current joint state", flush=True)
current_state = rospy.wait_for_message(f"/{robot_name}/joint_states", JointState, timeout=10.0)

print("checkpoint: calling compute_ik", flush=True)
rospy.wait_for_service(f"/{robot_name}/compute_ik", timeout=10.0)
compute_ik = rospy.ServiceProxy(f"/{robot_name}/compute_ik", GetPositionIK)

pose = PoseStamped()
pose.header.frame_id = f"{robot_name}/base_link"
pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
# ignored: kinematics.yaml sets position_only_ik, since px100's 4 joints cannot
# hit an arbitrary orientation (yaw is locked to atan2(y, x) by the waist)
pose.pose.orientation.w = 1.0

req = GetPositionIKRequest()
req.ik_request = PositionIKRequest(
    group_name=group_name,
    pose_stamped=pose,
    ik_link_name=f"{robot_name}/ee_gripper_link",
    avoid_collisions=True,
)
req.ik_request.robot_state.joint_state = current_state
req.ik_request.timeout = rospy.Duration(2.0)

resp = compute_ik(req)
if resp.error_code.val != 1:  # moveit_msgs/MoveItErrorCodes.SUCCESS
    # px100 link lengths from px100.urdf.xacro joint origins
    shoulder_height, max_reach = 0.0931, 0.3195
    needed = math.sqrt(x * x + y * y + (z - shoulder_height) ** 2)
    print(f"FAILED to find IK solution, error code {resp.error_code.val}")
    print(f"  target needs {needed:.4f} m of reach; px100 max is {max_reach:.4f} m")
    if needed > max_reach:
        print(f"  -> OUT OF WORKSPACE by {(needed - max_reach) * 1000:.1f} mm, pick a closer point")
    else:
        print("  -> within reach, so this is a joint-limit or near-singular pose; nudge the target")
    sys.exit(1)

target_positions = [resp.solution.joint_state.position[resp.solution.joint_state.name.index(j)] for j in arm_joints]
print(f"checkpoint: IK solution {dict(zip(arm_joints, target_positions))}", flush=True)

client = actionlib.SimpleActionClient(f"/{robot_name}/arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
print("checkpoint: waiting for arm_controller action server", flush=True)
client.wait_for_server()

goal = FollowJointTrajectoryGoal()
goal.trajectory.joint_names = arm_joints
goal.trajectory.points.append(JointTrajectoryPoint(positions=target_positions, time_from_start=rospy.Duration(duration)))

print("checkpoint: sending trajectory goal", flush=True)
client.send_goal(goal)
client.wait_for_result()
print("checkpoint: execution done", flush=True)

rospy.sleep(0.3)
sub.unregister()

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time", *arm_joints, "ee_x", "ee_y", "ee_z"])
    w.writerows(rows)

print(f"reached target, {len(rows)} samples logged to {out_path}")
