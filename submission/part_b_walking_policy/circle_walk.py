"""Run the pretrained Unitree G1 walking policy in MuJoCo and record it to MP4.

This is the control loop from unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py,
unchanged, except that:
  * it reads g1_circle.yaml (cmd_init = [0.5, 0, 0.5] -> constant yaw rate -> circle),
  * it renders offscreen to an MP4 instead of opening the interactive viewer,
  * it draws the path of the pelvis so the circle is visible in the video.

Usage:
    python circle_walk.py --rl-gym /path/to/unitree_rl_gym
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent


def get_gravity_orientation(q):
    qw, qx, qy, qz = q
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-gym", default=str(HERE.parents[1] / "unitree_rl_gym"),
                    help="path to a clone of unitree_rl_gym")
    ap.add_argument("--config", default=str(HERE / "g1_circle.yaml"))
    ap.add_argument("--out", default=str(HERE / "circle_walk.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    root = str(Path(args.rl_gym).resolve())
    cfg = yaml.safe_load(open(args.config))
    policy_path = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", root)
    xml_path = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", root)

    sim_dt = cfg["simulation_dt"]
    decim = cfg["control_decimation"]
    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)
    ang_vel_scale, dof_pos_scale = cfg["ang_vel_scale"], cfg["dof_pos_scale"]
    dof_vel_scale, action_scale = cfg["dof_vel_scale"], cfg["action_scale"]
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    n_act, n_obs = cfg["num_actions"], cfg["num_obs"]
    cmd = np.array(cfg["cmd_init"], dtype=np.float32)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt
    policy = torch.jit.load(policy_path)

    action = np.zeros(n_act, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(n_obs, dtype=np.float32)

    W, H = 960, 720
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    renderer = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 4.5, -45.0, 135.0

    steps = int(cfg["simulation_duration"] / sim_dt)
    render_every = int(round(1.0 / (args.fps * sim_dt)))
    trail = []
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")

    for counter in range(1, steps + 1):
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        if counter % decim == 0:
            qj = (d.qpos[7:] - default_angles) * dof_pos_scale
            dqj = d.qvel[6:] * dof_vel_scale
            grav = get_gravity_orientation(d.qpos[3:7])
            omega = d.qvel[3:6] * ang_vel_scale
            phase = (counter * sim_dt) % 0.8 / 0.8
            obs[:3] = omega
            obs[3:6] = grav
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + n_act] = qj
            obs[9 + n_act:9 + 2 * n_act] = dqj
            obs[9 + 2 * n_act:9 + 3 * n_act] = action
            obs[9 + 3 * n_act:9 + 3 * n_act + 2] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

        if counter % render_every == 0:
            trail.append(d.qpos[:2].copy())
            cam.lookat[:] = [0.0, 1.0, 0.5]  # fixed view over the circle (centre is ~1 m to the left)
            renderer.update_scene(d, cam)
            scn = renderer.scene
            for p in trail[::3]:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([0.03, 0, 0]), np.array([p[0], p[1], 0.02]),
                                    np.eye(3).flatten(), np.array([1, 0.5, 0, 1], dtype=np.float32))
                scn.ngeom += 1
            writer.append_data(renderer.render())

    writer.close()
    xy = np.array(trail)
    c = xy.mean(0)
    r = np.linalg.norm(xy - c, axis=1)
    print(f"saved {args.out}")
    print(f"pelvis path: centre ~({c[0]:.2f}, {c[1]:.2f}) m, radius {r.mean():.2f} +/- {r.std():.2f} m, "
          f"final height {d.qpos[2]:.2f} m")


if __name__ == "__main__":
    main()
