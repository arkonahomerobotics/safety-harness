"""Deterministic evaluation from normal starts: does the policy actually stack (robosuite definition)?

Runs each env's first full episode on the plain robosuite-reward task (no expert start states) and reports:
stacked ever, first-stack step, stacked at episode end, fraction of the last 50 steps stacked, early
terminations. "Stacked" = red lifted, touching blue, and not grasped by both fingers (contact sensors).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-RL-Robosuite-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--stochastic", action="store_true", help="Sample actions (as in training) instead of the mean.")
parser.add_argument("--stochastic_dims", type=str, default="", help="Comma list of action dims to sample; others use the mean.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app


import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.stack.mdp.robosuite_rewards import _in_contact
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint, map_location=agent_cfg.device)
policy = runner.get_inference_policy(device=u.device)
params = u.reward_manager.get_term_cfg("stack").params
lift_z, thr = params.get("lift_z", 0.0403), params.get("contact_threshold", 0.01)

n, dev = u.num_envs, u.device
T = int(u.max_episode_length)
finished = torch.zeros(n, dtype=torch.bool, device=dev)
early_term = torch.zeros(n, dtype=torch.bool, device=dev)
stacked_ever = torch.zeros(n, dtype=torch.bool, device=dev)
first_stack = torch.full((n,), -1, dtype=torch.long, device=dev)
last_stacked = torch.zeros(n, dtype=torch.bool, device=dev)
tail_hist = []

obs = env.get_observations()
with torch.inference_mode():
    for t in range(T):
        red = u.scene["cube_2"].data.root_pos_w.torch
        grasping = _in_contact(u, "left_finger_red_contact", thr) & _in_contact(u, "right_finger_red_contact", thr)
        touching = _in_contact(u, "red_blue_contact", thr)
        lifted = (red[:, 2] - u.scene.env_origins[:, 2]) > lift_z
        stacked = lifted & touching & ~grasping & ~finished
        newly = stacked & ~stacked_ever
        first_stack = torch.where(newly, torch.full_like(first_stack, t), first_stack)
        stacked_ever |= stacked
        last_stacked = torch.where(finished, last_stacked, stacked)
        if t >= T - 50:
            tail_hist.append(stacked.float())

        if args_cli.stochastic:
            actions = policy(obs, stochastic_output=True)
        elif args_cli.stochastic_dims:
            actions = policy(obs)
            dims = [int(d) for d in args_cli.stochastic_dims.split(",")]
            actions[:, dims] = policy(obs, stochastic_output=True)[:, dims]
        else:
            actions = policy(obs)
        obs, _, dones, extras = env.step(actions)
        policy.reset(dones)
        done = dones.bool() & ~finished
        timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
        early_term |= done & ~timeouts
        finished |= done

tail = torch.stack(tail_hist).mean(dim=0) if tail_hist else torch.zeros(n, device=dev)
fs = first_stack[first_stack >= 0].float()
print("=" * 90)
print(f"EVAL checkpoint: {args_cli.checkpoint}")
mode = "stochastic" if args_cli.stochastic else (f"sampled dims {args_cli.stochastic_dims}, rest mean" if args_cli.stochastic_dims else "deterministic")
print(f"EVAL envs={n} episode_len={T}  (normal starts, {mode} policy)")
print(f"EVAL stacked ever:              {stacked_ever.float().mean():.3f}")
print(f"EVAL stacked at episode end:    {last_stacked.float().mean():.3f}")
print(f"EVAL mean share of last 50 steps stacked: {tail.mean():.3f}")
if len(fs):
    print(f"EVAL first stack step: median={fs.median():.0f}  p10={fs.quantile(0.1):.0f}  p90={fs.quantile(0.9):.0f}")
print(f"EVAL early terminations (e.g. cube dropped): {early_term.float().mean():.3f}")
print("=" * 90)
env.close()
simulation_app.close()
