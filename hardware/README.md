# hardware — velocity control on the real WX200

Sends joint velocities to the **actual arm** on the LoCoBot, via the Interbotix
XS SDK. The simulation counterpart is [`../mujoco_arm/`](../mujoco_arm/); this
folder imports nothing from it on purpose, so nothing on the sim side can break
the hardware path.

This is [plan.md](../plan.md) phase 7 (sim → real) pulled forward, with phase 5's
safety layer included because velocity mode on hardware is not safe without it.

## Read this before running it

In **position** mode the arm holds still if your program dies. In **velocity**
mode a Dynamixel keeps turning at the last commanded speed until something stops
it — there is no command timeout in the servo. A crash, a badly-timed Ctrl-C or
a dropped connection means the arm drives itself into its hard stops.

`arm_velocity_hw.py` therefore:

| guard | default |
| --- | --- |
| per-joint speed cap | 0.4 rad/s (the URDF allows 3.14) |
| hard duration cap | 5 s |
| soft limit braking — scales to zero near a joint limit, from live `joint_states` | 0.20 rad margin |
| stops if `joint_states` goes stale | 0.5 s |
| stops on every exit path — normal, exception, Ctrl-C, rospy shutdown | always |
| restores `position` mode on exit | always |

The stop path uses `time.sleep`, never `rospy.sleep`: it runs when things are
going wrong, and `rospy.sleep` blocks once the master is gone — which is exactly
when the stop matters. An earlier version hung there when roscore went down
mid-command.

## Bring the robot up first

Nothing works without the control stack:

```bash
source /home/locobot/interbotix_ws/devel/setup.bash
source /home/locobot/UAN/Unexplored_Workspace_Navigation/uan_ws/devel/setup.bash
roslaunch uan_base_control uan_bringup.launch robot_model:=locobot_wx200
```

Check it is alive:

```bash
rosnode list | grep xs_sdk
rosservice call /locobot/get_robot_info "cmd_type: 'group'
name: 'arm'"
```

## Move the arm

```bash
cd /home/locobot/UAN/interbotix_ros_manipulators/hardware

# always look first -- reads real state, changes no mode, moves nothing
python3 arm_velocity_hw.py --joint elbow --vel -0.2 -t 1.5 --dry-run

# then for real
python3 arm_velocity_hw.py --joint elbow --vel -0.2 -t 1.5
python3 arm_velocity_hw.py --qd 0.2,0,-0.1,0,0 -t 1.5
```

Same arguments as `mujoco_arm/arm_velocity_publisher.py`, so anything you tried
in sim transfers. Joint names: `waist shoulder elbow wrist_angle wrist_rotate`.

Emergency stop: Ctrl-C. The handler zeroes the velocity and restores position
mode. If you want the arm limp, `rosservice call /locobot/torque_enable
"{cmd_type: 'group', name: 'arm', enable: false}"` — but it will fall.

## Things measured on this robot

- Joint names, order and limits are **identical to the sim model** — verified
  live against `get_robot_info`.
- `/locobot/joint_states` leads with `left_wheel_joint` and `right_wheel_joint`,
  so the arm joints are at indices 2–6, not 0–4. Always map **by name**;
  `get_robot_info`'s `joint_state_indices` reports `[0,1,2,3,4]`, which does not
  match the topic layout.
- The arm boots in `position` mode with a `time` profile, 2000/300 — that is
  what `restore_mode()` puts back.
- **In the rest pose the elbow sits at 1.678 rad, about 3.1° past its declared
  upper limit of 1.623.** The URDF limits are tighter than the servos' own
  limits. Limit braking handles it correctly (it blocks further positive elbow
  motion and allows negative), but a naive clamp-to-limits controller would
  either refuse to move or jump.
- Switching operating mode torques the motors off and back on, so expect a small
  sag at the switch. Start from the sleep/rest pose, where gravity torque is
  lowest.
