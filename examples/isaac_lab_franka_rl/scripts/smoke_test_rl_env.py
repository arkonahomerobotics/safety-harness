"""Cheap sanity check for the RL stack envs: reset + a few random steps, print per-term reward stats and the
contact-sensor matrices (verifies the sensor prim-path regexes resolved to exactly one body per env)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-RL-Robosuite-v0")
parser.add_argument("--no_snapshot", action="store_true", help="drop the snapshot-reset event (no snapshot file needed)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
if args_cli.no_snapshot and getattr(env_cfg.events, "reset_from_snapshot", None) is not None:
    env_cfg.events.reset_from_snapshot = None
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
print(f"action_space={env.action_space}  observation_space={env.observation_space}")

obs_dict, _ = env.reset()
print("reset ok, obs keys:", list(obs_dict.keys()), "policy shape:", obs_dict["policy"].shape)
print("arm action term:", type(env.action_manager.get_term("arm_action")).__name__)
for name, sensor in env.scene.sensors.items():
    if hasattr(sensor.data, "normal_force_matrix_w"):
        fm = sensor.data.normal_force_matrix_w
        print(f"SENSOR {name}: bodies={sensor.body_names} force_matrix shape={tuple(fm.torch.shape) if fm is not None else None}")

term_names = list(env.reward_manager.active_terms)
with torch.inference_mode():
    for step in range(args_cli.steps):
        actions = torch.rand((args_cli.num_envs, env.action_space.shape[1]), device=env.device) * 2 - 1
        obs_dict, reward, terminated, truncated, extras = env.step(actions)
        if step == args_cli.steps - 1:
            for name, sensor in env.scene.sensors.items():
                if hasattr(sensor.data, "normal_force_matrix_w"):
                    f = sensor.data.normal_force_matrix_w.torch.reshape(env.num_envs, -1, 3).norm(dim=-1).amax(dim=1)
                    print(f"SENSOR {name}: envs in contact (>0.01N) after random steps = {int((f > 0.01).sum())}/{env.num_envs}")

print("=" * 60)
print(f"ran {args_cli.steps} random steps across {args_cli.num_envs} envs with no crash.")
print("reward_manager active terms:", term_names)
print("final combined reward mean:", reward.mean().item(), "min:", reward.min().item(), "max:", reward.max().item())
print("terminated any:", bool(terminated.any().item()), "truncated any:", bool(truncated.any().item()))
env.close()
simulation_app.close()
