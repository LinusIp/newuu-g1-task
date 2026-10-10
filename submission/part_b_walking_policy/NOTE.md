# Part B — Walking Policy Notes

## Modification Summary
We modified `deploy/deploy_mujoco/deploy_mujoco.py` by overriding `cmd_init` to set `cmd = np.array([0.5, 0.0, 0.5], dtype=np.float32)`. Providing a forward velocity ($v_x = 0.5 \text{ m/s}$) combined with a constant yaw turning rate ($\omega_z = 0.5 \text{ rad/s}$) forces the policy to continuously steer while walking forward, resulting in a circular trajectory.

## Questions & Answers

### 1. What inputs does the policy receive, and where do they come from?
- **Inputs:**
  - **Target Velocity Commands:** $[v_x, v_y, \omega_z]$ (forward, lateral, yaw velocity commands).
  - **IMU Sensor Feedback:** Base angular velocities and projected gravity vector measured by the robot's IMU.
  - **Joint States:** Current joint positions relative to default stance ($q - q_{\text{default}}$) and joint velocities ($\dot{q}$).
  - **Action History:** Target joint angle commands from the previous control timestep ($a_{t-1}$).
- **Source:** Direct physical state queries from the MuJoCo simulation environment (`data.qpos`, `data.qvel`, sensor outputs) combined with user-defined trajectory commands.

### 2. What does the policy output, and how is that converted into joint torques?
- **Policy Output:** Target joint position offsets $\Delta q_{\text{target}}$ for all controlled actuators.
- **Torque Conversion:** Target joint positions are derived by scaling and adding offsets to default angles:
  $$q_{\text{target}} = q_{\text{default}} + K_{\text{scale}} \cdot \Delta q_{\text{target}}$$
  These positions are converted into motor torques $\tau$ via a Proportional-Derivative (PD) controller:
  $$\tau = K_p (q_{\text{target}} - q) - K_d \dot{q}$$
  where $K_p$ and $K_d$ are the proportional and damping gains specified in the environment setup.
