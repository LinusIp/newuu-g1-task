"""Play a LAFAN-style G1 motion CSV (exported from UltimateBots Studio) on the
Unitree G1 model from unitree_rl_gym in MuJoCo and render it to an MP4.

CSV layout (one row per frame, no header, 30 FPS, 36 columns):
    0-2   pelvis (root) position x, y, z in metres, world frame
    3-6   pelvis orientation quaternion in  x, y, z, w  order  (scalar LAST)
    7-35  29 joint angles in radians, same order as the joints in g1_29dof.xml
          (left leg 6, right leg 6, waist 3, left arm 7, right arm 7)

MuJoCo's free joint expects the quaternion as w, x, y, z, so it is reordered.

Usage:
    python playback.py --rl-gym /path/to/unitree_rl_gym [--csv motion.csv] [--out motion.mp4]
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
FPS = 30  # UltimateBots Studio exports at 30 FPS (300 rows == 10 s source clip)


def build_model(rl_gym: Path) -> mujoco.MjModel:
    """Load the 29-DoF G1 and add a floor + light (g1_29dof.xml has neither)."""
    xml = rl_gym / "resources/robots/g1_description/g1_29dof.xml"
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                     rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8],
                     width=300, height=300)
    mat = spec.add_material(name="grid")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [5, 5]
    mat.texuniform = True
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid")
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
    return spec.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-gym", default=str(HERE.parents[1] / "unitree_rl_gym"),
                    help="path to a clone of unitree_rl_gym")
    ap.add_argument("--csv", default=str(HERE / "motion.csv"))
    ap.add_argument("--out", default=str(HERE / "motion.mp4"))
    args = ap.parse_args()

    model = build_model(Path(args.rl_gym).resolve())
    data = mujoco.MjData(model)

    motion = np.loadtxt(args.csv, delimiter=",")
    n_joints = model.nq - 7
    assert motion.shape[1] == 7 + n_joints, f"expected {7 + n_joints} columns, got {motion.shape[1]}"
    print(f"{len(motion)} frames, {motion.shape[1]} columns, {len(motion) / FPS:.1f} s")

    # Joint limits — report (but don't clip) any frame that violates them.
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    viol = np.maximum(lo - motion[:, 7:], motion[:, 7:] - hi).max()
    print(f"max joint-limit violation: {max(viol, 0):.4f} rad")

    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 2.6, -12.0, 135.0

    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p")
    for row in motion:
        data.qpos[0:3] = row[0:3]                      # root position
        x, y, z, w = row[3:7]
        data.qpos[3:7] = [w, x, y, z]                  # xyzw -> wxyz
        data.qpos[7:] = row[7:]                        # 29 joint angles
        mujoco.mj_kinematics(model, data)              # kinematic replay, no physics

        cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.6]   # follow the robot
        renderer.update_scene(data, cam)
        writer.append_data(renderer.render())
    writer.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
