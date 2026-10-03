"""Record the stage-1 policy's handoff states (red stacked and released for --handoff_steps consecutive
steps, from normal starts) as stage-2 start states. Skill chaining: the next skill trains on the previous
skill's terminal-state distribution. Same file format as the expert snapshots (joint_pos, cube_pose).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-RL-FullStack-Snap-Sparse-v0")
parser.add_argument("--stage1", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--handoff_steps", type=int, default=10)
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
env_cfg.events.reset_from_snapshot = None
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.stage1, map_location=agent_cfg.device)
policy = runner.get_inference_policy(device=u.device)
robot = u.scene["robot"]
LIFT_Z, THR = 0.0403, 0.01
n, dev = u.num_envs, u.device
joints, poses, kinds = [], [], []

with torch.inference_mode():
    for rnd in range(args_cli.rounds):
        stochastic = rnd % 2 == 1
        obs, _ = env.reset()
        run = torch.zeros(n, dtype=torch.long, device=dev)
        taken = torch.zeros(n, dtype=torch.bool, device=dev)
        for t in range(args_cli.steps):
            oz = u.scene.env_origins[:, 2]
            red_z = u.scene["cube_2"].data.root_pos_w.torch[:, 2] - oz
            g_red = _in_contact(u, "left_finger_red_contact", THR) & _in_contact(u, "right_finger_red_contact", THR)
            s1 = (red_z > LIFT_Z) & _in_contact(u, "red_blue_contact", THR) & ~g_red
            run = torch.where(s1, run + 1, torch.zeros_like(run))
            new = (run >= args_cli.handoff_steps) & ~taken
            if new.any():
                ids = torch.nonzero(new).flatten()
                joints.append(robot.data.joint_pos.torch[ids].cpu())
                ps = []
                for name in ("cube_1", "cube_2", "cube_3"):
                    p = u.scene[name].data.root_link_pose_w.torch[ids].clone()
                    p[:, :3] -= u.scene.env_origins[ids]
                    ps.append(p.cpu())
                poses.append(torch.stack(ps, dim=1))
                kinds.append(torch.full((len(ids),), int(stochastic)))
                taken |= new
            actions = policy(obs, stochastic_output=True) if stochastic else policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            taken |= dones.bool()  # don't record from a fresh episode within this round
        print(f"HANDOFF round {rnd} ({'stochastic' if stochastic else 'deterministic'}): captured {int(sum(len(k) for k in kinds))} so far")

if not joints:  # port addition: a weak stage-1 policy can produce no handoff states at all
    print("HANDOFF saved=0: no env reached the handoff condition; nothing written")
    env.close()
    simulation_app.close()
    raise SystemExit(1)
jp, cp, kd = torch.cat(joints), torch.cat(poses), torch.cat(kinds)
torch.save({"joint_pos": jp, "cube_pose": cp, "kind": kd}, args_cli.out)
gap = (cp[:, 1, 2] - cp[:, 0, 2] - 0.0468) * 100
xy = torch.linalg.norm(cp[:, 1, :2] - cp[:, 0, :2], dim=1) * 100
print(f"HANDOFF saved={len(jp)} -> {args_cli.out}; red-on-blue gap median {gap.median():.2f}cm, xy median {xy.median():.2f}cm p90 {xy.quantile(0.9):.2f}")
env.close()
simulation_app.close()
