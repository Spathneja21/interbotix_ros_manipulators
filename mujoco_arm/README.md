# mujoco_arm — WX200 in MuJoCo

MuJoCo model of the Interbotix WX200 arm (the one on our LoCoBot), built for
velocity control experiments. See [../plan.md](../plan.md) for the full roadmap.

**Status:** phases 0–3 done — environment, model, velocity actuators, and the
joint-space velocity API. Cartesian (end-effector twist) control is phase 4.

## Layout

| Path | What |
| --- | --- |
| `convert/urdf_to_mjcf.py` | Regenerates the models from the vendored xacro |
| `tools/compute_gains.py` | Derives the servo gains from measured joint inertia |
| `models/wx200.xml` | Generated MJCF — **do not hand-edit**, patch the converter |
| `models/actuators_*.xml` | Generated: the two velocity-actuator variants |
| `models/scene.xml` | Hand-maintained: floor, lights, cameras + the arm |
| `models/scene_*.xml` | scene + one actuator variant ← **use these** |
| `models/bench_*.xml` | arm + actuators, **no floor** — for dynamics experiments |
| `models/meshes/` | STLs copied from `interbotix_xsarm_descriptions` |
| `arm_sim.py` | `ArmSim`: load, command, step, read state |
| `velocity_controller.py` | `VelocityController` + `Trace`: commands, control tick, logging |
| `arm_velocity_publisher.py` | CLI — the arm's `velocity_publisher.py` |
| `teleop_keyboard.py` | Keyboard teleop, mirrors the base's |
| `plots/plot_velocity_tracking.py` | Commanded vs actual vs error |
| `validate_model.py` | Checkpoint for phases 1–3, 35 checks |
| `experiments/compare_actuators.py` | Milestone M2: intvelocity vs velocity |
| `experiments/verify_tracking.py` | Milestone M3: tracking across joints and speeds |
| `view.py` | Interactive viewer |

## Use

### Command the arm

Mirrors `rosrun uan_base_control velocity_publisher.py -x 0.1 -t 3`:

```bash
python3 arm_velocity_publisher.py --joint elbow --vel -0.3 -t 2
python3 arm_velocity_publisher.py --qd 0.3,0,-0.2,0.15,0 -t 2
python3 arm_velocity_publisher.py --joint waist --vel 0.5 -t 2 --view
python3 arm_velocity_publisher.py --joint elbow --vel -0.3 -t 2 \
        --csv run.csv --plot run.png
python3 teleop_keyboard.py                   # 1/2 waist, 3/4 shoulder, ... x stops
```

```python
from velocity_controller import VelocityController
ctrl = VelocityController(mode="intvelocity", rate_hz=50)
ctrl.reset("home")
ctrl.set_joint_velocity([0, 0, -0.3, 0, 0])   # rad/s, wx200.yaml order
ctrl.run_and_stop(2.0)
ctrl.trace.to_csv("run.csv")
```

### Checks and regeneration

```bash
python3 validate_model.py                    # 35/35 expected
python3 validate_model.py --render           # + render_home.png / render_sleep.png
python3 experiments/compare_actuators.py     # M2: intvelocity vs velocity
python3 experiments/verify_tracking.py       # M3: tracking across joints and speeds
python3 tools/compute_gains.py               # re-derive gains if the model changes
python3 view.py --key sleep                  # interactive (needs a display; we have :0)
python3 convert/urdf_to_mjcf.py              # regenerate everything from source
```

## Velocity actuators (phase 2)

`intvelocity` is the default; `velocity` exists to justify that choice.
Measured over 3 s at **zero commanded velocity** from the home pose:

| | joint drift | EE sink | still moving at t=3 s? | step tracking |
| --- | --- | --- | --- | --- |
| **`intvelocity`** | **0.97°** | 1.0 cm | no (1e-4 rad/s) | **+0.7% error** |
| `velocity` | 43.8° | 44.8 cm | yes, 0.24 rad/s | −111% (wrong sign) |

A `<velocity>` servo's torque is `kv·(qd_cmd − qd)`, so at zero command it makes
zero torque and has nothing to oppose gravity with. `intvelocity` integrates the
command into a position setpoint and position-servos to it, so zero command
means *hold*. Its `actrange` also clamps that setpoint to the joint limits, so
the servo cannot wind up past a hard stop.

Gains are **derived, not hand-tuned**: `tools/compute_gains.py` measures the
worst-case joint inertia over the workspace and places a critically damped
40 rad/s loop (`kp = M·ω²`, `kv = 2·M·ω`). Predicted droop matched measurement
to 0.01°. Re-run it and paste the table into `convert/urdf_to_mjcf.py` if the
model changes.

**Use `bench_*.xml` (no floor) for any experiment that lets the arm sink.** With
the floor present, "the servo could not hold" silently becomes "the gripper
landed on the ground".

## Joint-velocity control (phase 3)

Every joint tracks a commanded velocity to **within 1.2% across two decades**
(0.02 → 2.0 rad/s), and stays put once stopped (<0.014° of creep per second).

Commands are applied on a **50 Hz control tick** while physics runs at 500 Hz.
That mirrors the real stack (the base publishes `cmd_vel` at 20 Hz) and gives
phase 5's watchdog a natural home. The command is re-issued every tick rather
than latched once, for the same reason.

Two numbers not to confuse when a joint is told to stop:

| | what it is | at 2 rad/s |
| --- | --- | --- |
| **stopping distance** | travel while decelerating — legitimately scales with speed | up to 6.6° |
| **creep** | motion *after* it has stopped — should be ~0 | 0.013°/s |

The settling tail is exponential but not instant: `wrist_rotate` is the lightest
joint with the lowest `kp`, and from 2 rad/s its residual velocity decays
5e-3 → 2e-3 → 5e-4 → 2e-5 rad/s over the following seconds. Measure creep at
least 2 s after the stop or you are measuring the tail of the transient.

## Model facts

- **8 DOF:** `waist, shoulder, elbow, wrist_angle, wrist_rotate` (the arm group,
  in `wx200.yaml` order) + `gripper`, `left_finger`, `right_finger`.
- **Two keyframes:** `home` (all arm joints 0) and `sleep`
  (`[0, -1.88, 1.5, 0.8, 0]`, from `wx200.yaml`).
  **Always reset to a keyframe, never `mj_resetData`** — `qpos0` is 0 for the
  fingers, which is outside their `[0.015, 0.037]` limit, so a bare reset starts
  them jammed against a stop.
- **`ee_gripper` site** is the end-effector frame. The URDF chain
  `gripper_link → ee_arm → gripper_bar → ee_bar → ee_gripper_link` is all fixed
  joints, so MuJoCo merged it away; the site re-adds that frame at
  `x = 0.093575` in `gripper_link`. Phase 4 takes `mj_jacSite` here.
- **`base_link` site** is at the world origin. The fixed `world → base_link`
  joint means the base_link frame *is* the world frame, so logged EE positions
  are directly comparable to `scripts/reach_and_log.py`, which works in
  `wx200/base_link`.
- **Home pose EE:** `(0.40858, 0, 0.31065)`. The zero pose is upper-arm-up,
  forearm-forward-horizontal — *not* straight up.
- **`actuatorfrcrange`** was carried over from the URDF effort limits
  (8/18/13/5/1 N·m), so torque limiting is already in place for phase 2.
- **`frictionloss = 0.1`** on every joint comes from the URDF's
  `<dynamics friction="0.1"/>`. On `wrist_rotate` that is 10% of the 1 N·m
  effort limit. ~~Expect a velocity deadband.~~ **Measured in phase 3: there
  isn't one** — worst error below 0.1 rad/s is 0.88%, and even at 0.02 rad/s
  tracking is within 0.9%. `intvelocity` is a *position* servo on an
  ever-advancing setpoint, so Coulomb friction costs a small constant position
  lag rather than a velocity deadband. A plain `velocity` servo would have had
  the deadband.

## Gotchas found while building this

- MuJoCo's URDF importer **requires** a `<mujoco><compiler/></mujoco>` block
  inside `<robot>`; without it, mesh paths do not resolve.
- URDF `<mimic>` is silently dropped. The `right_finger = -left_finger` coupling
  is re-added as an `<equality><joint/></equality>` constraint.
- MuJoCo auto-excludes parent/child contacts **except when the parent is the
  world body**. `base_link` merged into worldbody, so `base_link ↔ shoulder_link`
  reported a permanent bogus contact until excluded explicitly.
- The STL texture PNG has no UVs on these meshes; the converter swaps it for a
  flat `interbotix_black` colour.
- **The upper arm reaches its own base at shoulder ≈ 100.6°, inside the 113°
  joint limit.** Joint limits alone do not keep this arm out of trouble; the
  self-collision is real geometry, not a modelling artefact.
- Pose choice can silently invalidate a dynamics experiment. The arm-extended-
  forward pose has the largest gravity torque (2.4 N·m) and looks like the
  obvious worst case, but the EE sits 6 cm off the floor and the shoulder has
  only ~10° of travel before hitting the base — so a "does the servo hold?"
  test there measures the floor and the base, not the servo. The home pose has
  a real load *and* room to fall. `compare_actuators.py` now asserts zero
  contacts and <95% effort saturation rather than trusting the pose.
