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


bindings:
  0 -> wrist_rotate  +
  1 -> waist         -
  2 -> waist         +
  3 -> shoulder      -
  4 -> shoulder      +
  5 -> elbow         -
  6 -> elbow         +
  7 -> wrist_angle   -
  8 -> wrist_angle   +
  9 -> wrist_rotate  -

status line sample:
  waist=+0.10 shoul=+0.00 elbow=-0.20 wrist=+0.00 wrist=+0.00  | step 0.10  | ee (+0.408,+0.000,+0.311)   

--- non-tty guard ---
teleop needs a real terminal (stdin is not a tty)