"""Sanity check: reset Stage-2's RC env, take ZERO actions for 20 steps, and count how many envs
trip block_a_off_b anyway -- split by which snapshot bucket each env actually drew (handoff /
expert / default), inferred the same way verify_rc_mix.py does (C height for expert-like; the rest
are handoff-or-default, further split by whether A-on-B matches the tight default reset spread vs
the handoff snapshot's own natural variation isn't reliably separable post-hoc, so handoff+default
are reported together as "non-expert"). If block_a_off_b fires here, the loaded pose or settling
physics crosses the drop threshold on its own, independent of anything the policy does."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-Stage2-RC-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=20)
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
    pc0 = env.scene["block_c"].data.root_pos_w.torch - o_
    expert_like = pc0[:, 2] > 0.74
    tm = env.termination_manager

    tripped_step = torch.full((N,), -1, dtype=torch.long, device=dev)
    zero_act = torch.zeros(N, *env.action_space.shape[1:], device=dev)
    for step in range(args_cli.steps):
        obs, rew, term, trunc, extras = env.step(zero_act)
        fired = tm.get_term("block_a_off_b") & (tripped_step < 0)
        tripped_step[fired] = step

    tripped = tripped_step >= 0
    print(f"ZERO-ACTION block_a_off_b trips in {args_cli.steps} steps: {int(tripped.sum())}/{N} ({100*tripped.float().mean():.1f}%)")
    print(f"  of those, expert-like start: {int((tripped & expert_like).sum())}/{int(expert_like.sum())} expert-like envs")
    print(f"  of those, non-expert (handoff/default) start: {int((tripped & ~expert_like).sum())}/{int((~expert_like).sum())} non-expert envs")
    hist = torch.bincount(tripped_step[tripped], minlength=args_cli.steps)
    print(f"  trip-step histogram (step 0..{args_cli.steps-1}): {hist.tolist()}")

simulation_app.close()
