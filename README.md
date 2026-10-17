# NewUU Engineering Internship — Task 1 (Junior Track)

Unitree G1 in MuJoCo: (A) replaying a generated human motion, (B) making the pretrained walking policy walk in a circle.

| Part | Folder | Deliverables |
|---|---|---|
| A — Human motion to robot | [`submission/part_a_motion_retargeting`](submission/part_a_motion_retargeting) | `motion.mp4`, `motion.csv`, `source_prompt.txt`, `playback.py`, [README with Q1 answer](submission/part_a_motion_retargeting/README.md) |
| B — Walking policy | [`submission/part_b_walking_policy`](submission/part_b_walking_policy) | `circle_walk.mp4`, `g1_circle.yaml`, `circle_walk.py`, `g1_circle.patch`, [NOTE with Q1–Q2 answers](submission/part_b_walking_policy/NOTE.md) |

## Quick start

```bash
git clone https://github.com/LinusIp/newuu-g1-task.git
cd newuu-g1-task
git clone https://github.com/unitreerobotics/unitree_rl_gym.git   # G1 model + pretrained policy
pip install -r requirements.txt

# Part A  -> submission/part_a_motion_retargeting/motion.mp4
python submission/part_a_motion_retargeting/playback.py

# Part B  -> submission/part_b_walking_policy/circle_walk.mp4
python submission/part_b_walking_policy/circle_walk.py
```

Both scripts render offscreen, so no window is needed. On a headless Linux machine, set
`MUJOCO_GL=egl` (or `osmesa`). Both look for `unitree_rl_gym` in the repo root by default.
Pass `--rl-gym /path/to/unitree_rl_gym` to point them somewhere else.

Tested with Python 3.13, MuJoCo 3.x, PyTorch 2.x (CPU is enough).
