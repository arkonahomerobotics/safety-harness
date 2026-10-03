"""Verify the Stage-2 RC reset mix: reset a batch, classify each env as handoff / expert / default
by checking C's height (expert-mid-carry => C well above the table) and A's robot-relative
proximity vs the plain 5mm-jitter default (handoff => A within the handoff snapshot's own spread,
indistinguishable in practice -- so classify by C height and by whether robot joints are
near-default/rest, which differs sharply between the three cases) and print the fractions, plus
confirm the default-fallback case is a physically valid seated A-on-B with hand clear."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-Stage2-RC-v0")
parser.add_argument("--num_envs", type=int, default=512)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

with torch.inference_mode():
    obs, _ = env.reset()
    dev, N = env.device, env.num_envs
    o_ = env.scene.env_origins
    pa = env.scene["object"].data.root_pos_w.torch - o_
    pb = env.scene["block_b"].data.root_pos_w.torch - o_
    pc = env.scene["block_c"].data.root_pos_w.torch - o_

    # expert mid-carry: C is well above the table (held, not resting) -- C_POS z=0.72, table reset jitter is 0
    expert_like = pc[:, 2] > 0.74
    # default/handoff share the same A-on-B seated geometry; distinguish by A's exact offset spread:
    # the default per-block reset uses a tight +/-5mm xy jitter around a FIXED seat, the handoff
    # snapshot carries whatever natural variation model_5797's own episodes produced -- check the
    # robot's own hand joint state instead, which is unambiguous: handoff snapshots capture the
    # policy's RETRACTED post-success pose; a fresh default reset uses the architecture's default
    # joint pose. Compare left_wrist_yaw_link height, which differs meaningfully between a
    # just-reset default arm and a retracted-after-success arm.
    robot = env.scene["robot"]
    widx = robot.data.body_names.index("left_wrist_yaw_link")
    wrist_z = robot.data.body_pos_w.torch[:, widx, 2] - o_[:, 2]
    print(f"wrist_z stats: min={wrist_z.min():.3f} max={wrist_z.max():.3f} mean={wrist_z.mean():.3f} std={wrist_z.std():.3f}")
    print(f"EXPERT-LIKE (C held, z>0.74): {int(expert_like.sum())}/{N} ({100*expert_like.float().mean():.1f}%)")
    non_expert = ~expert_like
    print(f"NOT expert-like (handoff or default): {int(non_expert.sum())}/{N}")

    # A-on-B validity check for ALL envs (handoff + default should both be physically valid)
    dxy = torch.linalg.norm((pa - pb)[:, :2], dim=1)
    dz = pa[:, 2] - pb[:, 2] - 0.045
    wrist_pos = robot.data.body_pos_w.torch[:, widx] - o_
    hand_clear = torch.linalg.norm(pa - wrist_pos, dim=1) > 0.05
    valid_ab = (dxy < 0.03) & (dz.abs() < 0.02)
    print(f"A-on-B geometrically valid (dxy<3cm, dz within 2cm of one block height): {int(valid_ab.sum())}/{N}")
    print(f"hand clear of A (>5cm): {int(hand_clear.sum())}/{N}")
    print(f"A-on-B valid AND hand clear: {int((valid_ab & hand_clear).sum())}/{N}")
    if not bool(valid_ab.all()):
        bad = torch.nonzero(~valid_ab).flatten()[:5].tolist()
        for i in bad:
            print(f"  INVALID env={i} pa={pa[i].tolist()} pb={pb[i].tolist()} dxy={dxy[i]:.4f} dz={dz[i]:+.4f}")

simulation_app.close()
