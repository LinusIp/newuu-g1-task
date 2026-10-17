# Part A — Human Motion to Robot

Motion generated in **UltimateBots Studio** for the Unitree G1, exported as a LAFAN CSV,
and replayed on the G1 model from **unitree_rl_gym** in MuJoCo.

| File | What it is |
|---|---|
| `source_prompt.txt` | Text prompt used in UltimateBots Studio |
| `media.mp4` | Studio's own preview of the generated motion (for reference) |
| `motion.csv` | Exported LAFAN motion (300 frames @ 30 FPS = 10 s) |
| `playback.py` | Loads G1 in MuJoCo, plays `motion.csv`, renders `motion.mp4` |
| `motion.mp4` | Rendered result |

## Reproduce

From the repository root:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_gym.git   # model files
pip install -r requirements.txt

cd submission/part_a_motion_retargeting
python playback.py --rl-gym ../../unitree_rl_gym
# -> motion.mp4
```

On a headless Linux machine, prefix the command with `MUJOCO_GL=egl` (or `MUJOCO_GL=osmesa`).

The script uses `resources/robots/g1_description/g1_29dof.xml`. The repo's default
`scene.xml` loads the **12-DoF** legs-only model, which cannot show the arm motion. A floor
and light are added in code with `MjSpec`. Each CSV row is written straight into
`data.qpos`, followed by `mj_kinematics`. This is a kinematic replay, so no physics or
controller runs, and the camera follows the pelvis.

## Q1. How did you determine what the columns in the CSV represent?

The file has no header, so I worked the layout out from the data and checked it against
the model:

1. **Column count.** Every row has 36 values. The free-floating 29-DoF G1 in MuJoCo has
   `nq = 36` (7 for the free base + 29 hinge joints). 36 = 3 + 4 + 29.
2. **Columns 0–2 are root position (x, y, z, metres).** Column 2 starts at ≈0.77 m, which is
   the G1 pelvis standing height, and stays between 0.69 and 0.77. Column 0 grows from 0 to
   ≈5 m during the "walking" part of the prompt.
3. **Columns 3–6 are a quaternion, stored x, y, z, w (scalar last).** Their norm is 1.000
   on every row, so it's a unit quaternion. In frame 0 the value ≈1 is in **column 6**, not
   column 3, and an upright, unrotated body has a quaternion of (0, 0, 0, 1) in xyzw order.
   During the 360° turn, column 5 (z) rises to 1 and column 6 (w) goes from 1 through 0 to
   −1, which is a full rotation about the vertical axis. MuJoCo wants w, x, y, z, so the
   script reorders it. *(My first submission skipped this step, and the robot rendered
   lying sideways.)*
4. **Columns 7–35 are the 29 joint angles (radians) in the same order as the joints in
   `g1_29dof.xml`.** The script prints the model's joints: left leg (6), right leg (6),
   waist yaw/roll/pitch (3), left arm (7), right arm (7). I compared each column's min/max
   to that joint's `range` in the XML, and every column is inside its limit (the script
   reports a max violation of 0.0000 rad). The signs also fit. Knees (cols 10 and 16) are
   always positive and bent. The left shoulder-roll column only goes positive (arm out to
   the left), and the right one only goes negative (arm out to the right). The
   shoulder/elbow columns move a lot in the first seconds, when the robot waves both hands.
5. **Frame rate.** 300 rows match the 10 s, 30 FPS preview video `media.mp4` exported from
   the Studio, so playback runs at 30 FPS. A side-by-side comparison of `motion.mp4` and
   `media.mp4` shows the same motion: a two-hand wave, a 360° turn, then walking.
