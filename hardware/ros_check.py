"""Fail fast, with instructions, when the robot's ROS stack is not running.

Without this, rospy.init_node() waits forever printing "Unable to register with
master node ... Will keep trying", which reads like a bug in the script rather
than "start the robot first".
"""
import socket
import sys

BRINGUP = """
The robot's ROS stack is not running (no ROS master at {uri}).
Start it in its own terminal and leave it running:

  source /home/locobot/interbotix_ws/devel/setup.bash
  source /home/locobot/UAN/Unexplored_Workspace_Navigation/uan_ws/devel/setup.bash
  roslaunch uan_base_control uan_bringup.launch robot_model:=locobot_wx200
"""


def require_master(timeout=2.0, need_node="/locobot/xs_sdk"):
    import rosgraph

    uri = rosgraph.get_master_uri()
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        master = rosgraph.Master("/uan_ros_check")
        master.getPid()
        if need_node:
            state = master.getSystemState()
            nodes = {n for group in state for _, ns in group for n in ns}
            if need_node not in nodes:
                sys.exit(f"\nROS master is up, but {need_node} is not running -- "
                         f"the robot driver has not started (or crashed).\n"
                         + BRINGUP.format(uri=uri))
    except (socket.error, OSError):
        sys.exit(BRINGUP.format(uri=uri))
    finally:
        socket.setdefaulttimeout(old)
