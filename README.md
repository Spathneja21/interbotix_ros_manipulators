![manipulator_banner](images/manipulator_banner.png)

## Overview
![manipulator_repo_structure](images/manipulator_repo_structure.png)
Welcome to the *interbotix_ros_manipulators* repository! This repo contains custom ROS packages to control the various types of arms sold at [Trossen Robotics](https://www.trossenrobotics.com/). These ROS packages build upon the ROS driver nodes found in the [interbotix_ros_core](https://github.com/Interbotix/interbotix_ros_core) repository. Support-level software can be found in the [interbotix_ros_toolboxes](https://github.com/Interbotix/interbotix_ros_toolboxes) repository.

### Build Status

| ROS Distro | X-Series ROS Manipulators Build |
| :------- | :------- |
| ROS 1 Noetic | [![build-xs-noetic](https://github.com/Interbotix/interbotix_ros_manipulators/actions/workflows/xs-noetic.yaml/badge.svg)](https://github.com/Interbotix/interbotix_ros_manipulators/actions/workflows/xs-noetic.yaml) |
| ROS 2 Galactic | [![build-xs-galactic](https://github.com/Interbotix/interbotix_ros_manipulators/actions/workflows/xs-galactic.yaml/badge.svg)](https://github.com/Interbotix/interbotix_ros_manipulators/actions/workflows/xs-galactic.yaml) |
| ROS 2 Humble | [![build-xs-humble](https://github.com/Interbotix/interbotix_ros_manipulators/actions/workflows/xs-humble.yaml/badge.svg)](https://github.com/Interbotix/interbotix_ros_manipulators/actions/workflows/xs-humble.yaml) |
| ROS 2 Rolling | [![build-xs-rolling](https://github.com/Interbotix/interbotix_ros_manipulators/actions/workflows/xs-rolling.yaml/badge.svg)](https://github.com/Interbotix/interbotix_ros_manipulators/actions/workflows/xs-rolling.yaml) |

## Repo Structure
```
GitHub Landing Page: Explains repository structure and contains a single directory for each type of manipulator.
├── Manipulator Type X Landing Page: Contains 'core' arm ROS packages.
│   ├── Core Arm ROS Package 1
│   ├── Core Arm ROS Package 2
│   ├── Core Arm ROS Package X
│   └── Examples: contains 'demo' arm ROS packages that build upon some of the 'core' arm ROS packages
│       ├── Demo Arm ROS Package 1
│       ├── Demo Arm ROS Package 2
│       ├── Demo Arm ROS Package X
│       └── Demo Scripts: contains example scripts that build upon the interface modules in the interbotix_ros_toolboxes repository
│           ├── Demo Script 1
│           ├── Demo Script 2
|           └── Demo Script X
├── CITATION.cff
├── LICENSE
└── README.md
```
As shown above, there are five main levels to this repository. To clarify some of the terms above, refer to the descriptions below.

- **Manipulator Type** - Any robotic arm that can use the same *interbotix_XXarm_control* package is considered to be of the same type. For the most part, this division lies on the type of actuator that makes up the robot. As an example, all the X-Series arms are considered the same type of manipulator since they all use various Dynamixel X-Series servos (despite the fact that they come in different sizes, DOF, and motor versions). However, a robotic arm made up of some other manufacturer's servos, or even half made up of Dynamixel servos and half made up of some other manufacturer's servos would be considered a different manipulator type.

- **Core Arm ROS Package** - This refers to 'High Profile' ROS packages that are essential to make a given arm work. Examples of 'High Profile' ROS packages include:
    - *interbotix_XXarm_control* - sets up the proper configurations and makes it possible to control the physical arm
    - *interbotix_XXarm_moveit* - sets up the proper configurations and makes it possible to control an arm via MoveIt
    - *interbotix_XXarm_gazebo* - sets up the proper configurations and makes it possible to control a Gazebo simulated arm
    - *interbotix_XXarm_ros_control*  - ROS control package used with MoveIt to control the physical arms
    - *interbotix_XXarm_descriptions* - contains URDFs and meshes of the arms, making it possible to visualize them in RViz

- **Demo Arm ROS Package** - This refers to demo ROS packages that build upon the **Core Arm ROS Packages**. ROS researchers could use these packages as references to learn how to develop their own ROS packages and to get a feel for how the robot works. Typical demos for a given manipulator type include:
    - *interbotix_XXarm_joy* - manipulate an arm's end-effector using a joystick controller
    - *interbotix_XXarm_puppet* - make one or more 'puppet' arms copy the motion of a 'master' arm
    - *interbotix_XXarm_moveit_interface* - learn how to use MoveIt!'s MoveGroup Python or C++ APIs to control a robot arm

- **Demo Script** - This refers to demo scripts that build upon the interface modules in the *interbotix_ros_toolboxes* repository. These modules essentially abstract away all ROS code, making it easy for a researcher with no ROS experience to interface with an arm as if it was just another object. It also makes sequencing robot motion a piece of cake. These scripts are written in languages that users may feel more comfortable with like Python and MATLAB. The directories that contain demo scripts for each language may be found the in example directory, or in the package that specifically relates to their usage, such as the perception packages.

Over time, the repo will grow to include more types of manipulators.

## Contributing
Feel free to send PRs to add features to currently existing Arm ROS packages or to include new ones. Note that all PRs should follow the structure and naming conventions outlined in the repo including documentation.

## Contributors
- [Solomon Wiznitzer](https://github.com/swiz23) - **ROS Engineer**
- [Luke Schmitt](https://github.com/lsinterbotix) - **Robotics Software Engineer**
- [Levi Todes](https://github.com/LeTo37) - **CAD Engineer**

## Citing

If using this software for your research, please include the following citation in your publications:

```bibtex
@software{Wiznitzer_interbotix_ros_manipulators,
  author = {Wiznitzer, Solomon and Schmitt, Luke and Trossen, Matt},
  license = {BSD-3-Clause},
  title = {{interbotix_ros_manipulators}},
  url = {https://github.com/Interbotix/interbotix_ros_manipulators}
}


```bash
joint-space inertia M_ii over 600 random poses (kg m^2), incl. armature 0.01
joint                min       max    median
waist            0.01518   0.10828   0.03423
shoulder         0.02369   0.10709   0.07588
elbow            0.02444   0.03642   0.03304
wrist_angle      0.01506   0.01507   0.01507
wrist_rotate     0.01307   0.01307   0.01307

gravity torque worst case (N m):
  waist          max= 0.000   effort limit= 8.0  (0%)
  shoulder       max= 2.424   effort limit=18.0  (13%)
  elbow          max= 0.980   effort limit=13.0  (8%)
  wrist_angle    max= 0.192   effort limit= 5.0  (4%)
  wrist_rotate   max= 0.010   effort limit= 1.0  (1%)



measured over 2000 random poses, omega_n = 40.0 rad/s

joint            M_max   tau_g       kp      kv  kv(vel)    droop
                kg m^2     N m  N m/rad   N m s    N m s      deg
-----------------------------------------------------------------
waist          0.10828   0.000   173.25   8.663    4.331     0.00
shoulder       0.10715   2.428   171.44   8.572    4.286     0.81
elbow          0.03642   0.984    58.28   2.914    1.457     0.97
wrist_angle    0.01507   0.193    24.11   1.206    0.603     0.46
wrist_rotate   0.01307   0.010    20.91   1.045    0.523     0.03

# paste into arm_sim.py
GAINS = {
    "waist": dict(kp=173.25, kv=8.663, kv_vel=4.331),
    "shoulder": dict(kp=171.44, kv=8.572, kv_vel=4.286),
    "elbow": dict(kp=58.28, kv=2.914, kv_vel=1.457),
    "wrist_angle": dict(kp=24.11, kv=1.206, kv_vel=0.603),
    "wrist_rotate": dict(kp=20.91, kv=1.045, kv_vel=0.523),
}

sanity checks
  waist          gravity uses  0.0% of the 8 N m limit; saturates at   2.6 deg of setpoint error
  shoulder       gravity uses 13.5% of the 18 N m limit; saturates at   6.0 deg of setpoint error
  elbow          gravity uses  7.6% of the 13 N m limit; saturates at  12.8 deg of setpoint error
  wrist_angle    gravity uses  3.9% of the 5 N m limit; saturates at  11.9 deg of setpoint error
  wrist_rotate   gravity uses  1.0% of the 1 N m limit; saturates at   2.7 deg of setpoint error

  a plain <velocity> servo cannot hold a static load: at zero
  commanded velocity its torque is kv*(0 - qd), so it settles into a
  CONSTANT SINK RATE tau_g/kv, it does not merely droop:
    shoulder        0.567 rad/s (  32.5 deg/s)
    elbow           0.675 rad/s (  38.7 deg/s)
    wrist_angle     0.319 rad/s (  18.3 deg/s)
    wrist_rotate    0.019 rad/s (   1.1 deg/s)
```