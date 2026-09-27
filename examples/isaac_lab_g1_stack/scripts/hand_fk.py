"""Measure the G1 left Dex3 hand's link positions in the WRIST frame at several closure fractions,
with the hand held in free space (block untouched). The grasp pocket is where thumb and finger tips
converge when closed -- the block center should sit there, rather than at a guessed standoff.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

HAND_JOINT_NAMES = [
    "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
    "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "right_hand_thumb_2_joint",
]
LEFT_CURL = [0, 1, 6, 7, 8, 12]
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
THUMB_YAWS = [-0.6, 0.0, 0.6]
CONFIGS = [(f, t) for t in THUMB_YAWS for f in FRACS]
N = len(CONFIGS)

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

with torch.inference_mode():
    obs, _ = env.reset()
    p = obs["policy"]
    robot = env.scene["robot"]
    dev = env.device
    hand_ids, _ = robot.find_joints(HAND_JOINT_NAMES)
    lim = robot.data.soft_joint_pos_limits.torch[0, hand_ids]
    lo, hi = lim[:, 0], lim[:, 1]
    open_vals = torch.clamp(torch.zeros_like(lo), lo, hi)
    closed_vals = torch.where(lo.abs() > hi.abs(), lo, hi)

    hand = open_vals.unsqueeze(0).repeat(N, 1)
    for i, (f, t) in enumerate(CONFIGS):
        hand[i, LEFT_CURL] = open_vals[LEFT_CURL] + f * (closed_vals[LEFT_CURL] - open_vals[LEFT_CURL])
        hand[i, 2] = t

    a = torch.zeros(N, 28, device=dev)
    a[:, 0:3] = p["left_eef_pos"]
    a[:, 0] += 0.05  # lift the hand clear of the table/block
    a[:, 2] += 0.10
    a[:, 3:7] = p["left_eef_quat"]
    a[:, 7:10] = p["right_eef_pos"]
    a[:, 10:14] = p["right_eef_quat"]
    a[:, 14:28] = hand
    for _ in range(80):
        env.step(a)

    names = robot.data.body_names
    wid = names.index("left_wrist_yaw_link")
    wpos = robot.data.body_pos_w.torch[:, wid]
    wquat = robot.data.body_quat_w.torch[:, wid]
    print("wrist quat (world, env0):", [round(v, 3) for v in wquat[0].tolist()])
    print("left_eef_quat obs (env0):", [round(v, 3) for v in p["left_eef_quat"][0].tolist()])
    left = [n for n in names if n.startswith("left_hand")]
    for i, (f, t) in enumerate(CONFIGS):
        print(f"--- closure={f:.2f} thumb_yaw={t:+.1f}  (positions in wrist frame, cm)")
        for n in left:
            bid = names.index(n)
            rel = quat_apply_inverse(wquat[i : i + 1], (robot.data.body_pos_w.torch[i : i + 1, bid] - wpos[i : i + 1]))[0]
            print(f"   {n:28s} {rel[0]*100:7.2f} {rel[1]*100:7.2f} {rel[2]*100:7.2f}")
    # the same links in the WORLD frame (for env0 = open) to confirm the wrist-frame axes
    print("world-frame offsets, open hand, env0 (cm):")
    for n in left:
        bid = names.index(n)
        rel = robot.data.body_pos_w.torch[0, bid] - wpos[0]
        print(f"   {n:28s} {rel[0]*100:7.2f} {rel[1]*100:7.2f} {rel[2]*100:7.2f}")

simulation_app.close()
