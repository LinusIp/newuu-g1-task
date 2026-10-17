# Part B — G1 Walking Policy: Walking in a Circle

**Video:** [`circle_walk.mp4`](circle_walk.mp4). This is 20 s of simulation. The orange
dots trace the pelvis path on the ground.

## What I changed

I changed **one value**: the velocity command `cmd_init` in
`deploy/deploy_mujoco/configs/g1.yaml`.

```diff
-cmd_init: [0.5, 0, 0]      # vx = 0.5 m/s, vy = 0, yaw rate = 0        -> straight line
+cmd_init: [0.5, 0, 0.5]    # vx = 0.5 m/s, vy = 0, yaw rate = 0.5 rad/s -> circle
```

The policy weights (`deploy/pre_train/g1/motion.pt`) and the control loop are unchanged.

| File | Purpose |
|---|---|
| `g1_circle.patch` | The change as a patch to the official repo (`git apply` inside `unitree_rl_gym`) |
| `g1_circle.yaml` | The full config with the change (and `simulation_duration: 20`) |
| `circle_walk.py` | Same control loop as `deploy_mujoco.py`, but it renders offscreen to MP4, draws the path, and prints the measured radius |

### Reproduce

From the repository root, with `unitree_rl_gym` cloned there:

```bash
pip install -r requirements.txt

# (a) recorded video, no window needed
python submission/part_b_walking_policy/circle_walk.py
#   -> pelvis path: centre ~(0.16, 0.99) m, radius 1.04 +/- 0.16 m, final height 0.77 m

# (b) or the official interactive viewer
cd unitree_rl_gym
git apply ../submission/part_b_walking_policy/g1_circle.patch
pip install -e .          # deploy_mujoco.py imports LEGGED_GYM_ROOT_DIR from legged_gym
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

### Why this gives a circle

The policy was trained to track a commanded body velocity `[vx, vy, ωz]`. The commands
are expressed in the robot's own (heading) frame. A constant forward speed `v` plus a
constant yaw rate `ω` means the heading keeps rotating at `ω` while the robot keeps moving
"forward" along that rotating heading. The path is the curve with constant curvature
`ω / v`, which is a circle of radius

```
R = v / ω = 0.5 m/s / 0.5 rad/s = 1.0 m        (one lap every 2π/ω ≈ 12.6 s)
```

The measured path in simulation has a radius of **1.04 m**, centred about 1 m to the
robot's left, which matches the prediction. A positive `ωz` is counter-clockwise (a left
turn). A negative value would circle to the right. Raising `vx` or lowering `ωz` makes the
circle larger. I kept the values inside what the policy handles well. A much higher yaw
rate makes tracking less accurate.

## Q1. What inputs does the policy receive, and where do they come from?

The policy receives a **47-dimensional observation vector**, rebuilt every 20 ms (50 Hz)
in `deploy_mujoco.py`:

| Index | Size | Quantity | Source in the sim (real robot equivalent) | Scale |
|---|---|---|---|---|
| 0–2 | 3 | Base angular velocity | `d.qvel[3:6]`, the pelvis free joint's angular velocity in the body frame (IMU gyroscope) | × 0.25 |
| 3–5 | 3 | Projected gravity, i.e. the gravity direction seen from the body (tells the policy its tilt) | Computed from the pelvis quaternion `d.qpos[3:7]` by `get_gravity_orientation()` (IMU orientation) | — |
| 6–8 | 3 | Velocity command `[vx, vy, ωz]` | `cmd_init` in the YAML (on the real robot, the joystick) — **the value I changed** | × `[2, 2, 0.25]` |
| 9–20 | 12 | Leg joint positions relative to the default pose, `q − q_default` | `d.qpos[7:]` (joint encoders) | × 1.0 |
| 21–32 | 12 | Leg joint velocities | `d.qvel[6:]` (joint encoders) | × 0.05 |
| 33–44 | 12 | The policy's own previous action | Kept from the last inference step | — |
| 45–46 | 2 | Gait clock `sin(2πφ)`, `cos(2πφ)` | φ = (sim time mod 0.8 s) / 0.8 s, a 0.8 s stepping period | — |

All robot-state values are read straight from MuJoCo's state arrays (`qpos`, `qvel`). On
the real G1, the same quantities come from the IMU and joint encoders. The scale factors
are the same normalisations used in training, so the inputs land in the ranges the
network saw then. There is no linear-velocity input, so the policy has to infer how fast
it is moving from the other inputs.

## Q2. What does the policy output, and how is that converted into joint torques?

**Output:** 12 numbers, one per leg joint (6 per leg: hip pitch/roll/yaw, knee, ankle
pitch/roll). They are *not* torques. They are scaled offsets from the default standing
pose, which gives a **target joint position**:

```
q_target = action × action_scale + q_default        (action_scale = 0.25)
```

This runs at **50 Hz** (`simulation_dt 0.002 s × control_decimation 10`).

**Torques:** a **PD controller** turns the targets into torques at the physics rate,
**500 Hz**, on every `mj_step`, while holding the latest target between policy updates:

```
τ = Kp · (q_target − q) + Kd · (0 − q̇)
```

- `Kp = [100, 100, 100, 150, 40, 40]` per leg (hip pitch/roll/yaw, knee, ankle pitch/roll)
- `Kd = [2, 2, 2, 4, 2, 2]`
- the target velocity is zero, so the D term is pure damping.

The torques are written to `d.ctrl`. In `g1_12dof.xml` the actuators are `<motor>`
elements, i.e. direct torque motors, so MuJoCo applies `d.ctrl` as joint torque. This
setup (low-rate policy, high-rate PD loop) matches how the policy was trained in Isaac
Gym, and how the real G1's motor drivers track position targets.
