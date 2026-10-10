# Part A — Human Motion to Robot

## Overview
This directory contains the pipeline for generating and replaying human motion on the Unitree G1 robot model in MuJoCo.

## Files
- `source_promt.txt`: Prompt or reference used to generate the motion.
- `motion.csv`: Exported LAFAN motion trajectory file from UltimateBots Studio.
- `playback.py`: Python script loading the G1 model in MuJoCo, stepping through joint angles, and rendering the motion.
- `motion.mp4`: Final rendered video of the Unitree G1 executing the motion.

## Reproduction Steps
1. Ensure required dependencies are installed:
   ```bash
   pip install mujoco opencv-python pandas numpy
