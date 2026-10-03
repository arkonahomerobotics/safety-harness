"""Minimal reach check for a handful of Stage-2 C_POS candidates -- same methodology as
reach_map.py but a tiny grid (a few envs, not reach_map.py's full 288) to keep the GPU footprint
small while other training/eval jobs are also running. Edit XS/YS/ZS/ROLLS below for new
candidates; used to pick C_POS=(0.05, 0.25, 0.72) (22cm from B_POS, 0.76-0.86cm pos error) after
the first candidate (0.0, 0.35), only 14cm from B_POS, let the scripted expert's approach knock the
A-on-B tower over."""

import argparse
import itertools

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab.utils.math import quat_error_magnitude, quat_from_angle_axis, quat_mul  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# Candidates for the NEW C_POS (>=20cm from B_POS=(-0.14,0.36), outside the tower's swept area):
# (0.05,0.25): dist to B = 24.8cm   (0.05,0.20): dist to B = 27.7cm   (-0.05,0.45): dist to B = 13.5cm (too close, included as a negative control)
XS = [0.05, -0.05]
YS = [0.25, 0.20, 0.45]
ZS = [0.73, 0.78]
ROLLS = [0.0, 20.0, 30.0]
GRID = list(itertools.product(XS, YS, ZS, ROLLS))
N = len(GRID)

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
env_cfg.scene.robot_pov_cam = None  # unconditional XR leftover, crashes init without enable_cameras
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

with torch.inference_mode():
    obs, _ = env.reset()
    p = obs["policy"]
    dev = env.device
    t = torch.tensor([g[:3] for g in GRID], device=dev)
    roll = torch.deg2rad(torch.tensor([g[3] for g in GRID], device=dev))
    q_rot = quat_from_angle_axis(roll, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(N, 3))
    q = quat_mul(q_rot, p["left_eef_quat"])
    a = torch.zeros(N, 28, device=dev)
    a[:, 3:7] = q
    a[:, 7:10] = p["right_eef_pos"]
    a[:, 10:14] = p["right_eef_quat"]
    start = p["left_eef_pos"].clone()
    for k in range(150):
        f = min(1.0, (k + 1) / 80)
        a[:, 0:3] = start + f * (t - start)
        obs, *_ = env.step(a)
    e = torch.linalg.norm(obs["policy"]["left_eef_pos"] - t, dim=1)
    qe = quat_error_magnitude(obs["policy"]["left_eef_quat"], q)
    for i, (x, y, z, rl) in enumerate(GRID):
        print(f"x={x} y={y} z={z} roll={rl}: pos_err_cm={e[i]*100:.2f} rot_err_deg={qe[i]*57.3:.1f}")

simulation_app.close()
