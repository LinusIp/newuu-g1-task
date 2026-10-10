import cv2
import mujoco
import numpy as np
import pandas as pd
from pathlib import Path

# Paths relative to part_a_motion_retargeting directory
XML_PATH = Path("../../unitree_rl_gym/resources/robots/g1_description/scene.xml")
CSV_PATH = Path("motion.csv")
OUTPUT_VIDEO = Path("motion.mp4")

# Load Model
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

# Read CSV without header
df = pd.read_csv(CSV_PATH, header=None)
print(f"Loaded CSV with {len(df)} frames and {df.shape[1]} columns.")

# Renderer setup matching XML framebuffer size (640x480)
FPS = 30
WIDTH, HEIGHT = 640, 480
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (WIDTH, HEIGHT))

for idx, row in df.iterrows():
    qpos = np.zeros(model.nq)

    # 1. Floating base position (x, y, z)
    qpos[0:3] = row.iloc[0:3].values

    # 2. Floating base quaternion (quat_w, quat_x, quat_y, quat_z)
    qpos[3:7] = row.iloc[3:7].values

    # 3. Leg joint angles (12 DOFs)
    qpos[7:19] = row.iloc[7:19].values

    # Apply qpos and update forward kinematics
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    # Render frame
    renderer.update_scene(data)
    frame = renderer.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

out.release()
print(f"Successfully generated {OUTPUT_VIDEO}")
