"""Diagnose a BC (or any rsl_rl) stage-1 policy: deterministic rollout from normal starts with a shadow scripted expert
following along (phase machine fed with the same states), reporting how far the policy gets (reach / grasp / lift /
over blue / seated / released) and the policy-vs-expert action error per expert phase on the policy's own states.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-RL-Robosuite-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils import to_dict  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bc_expert as cbf  # noqa: E402

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
if getattr(env_cfg.events, "reset_from_snapshot", None) is not None:
    env_cfg.events.reset_from_snapshot = None
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint, map_location=agent_cfg.device)
policy = runner.get_inference_policy(device=u.device)
n, dev = u.num_envs, u.device
robot = u.scene["robot"]
fid, _ = robot.find_joints("panda_finger_joint.*")
arm = u.action_manager.get_term("arm_action")
origins = u.scene.env_origins
experts = [cbf.Expert([("cube_2", "cube_1")]) for _ in range(n)]
min_reach = np.full(n, 9.0)
lifted = np.zeros(n, bool)
over_blue = np.zeros(n, bool)
seated_ever = np.zeros(n, bool)
released_seated = np.zeros(n, bool)
err = {}
obs = env.get_observations()
with torch.inference_mode():
    for t in range(args_cli.steps):
        eef = (u.scene["ee_frame"].data.target_pos_w.torch[:, 0, :] - origins).cpu().numpy().astype(np.float64)
        cubes = {k: (u.scene[k].data.root_pos_w.torch - origins).cpu().numpy().astype(np.float64) for k in ("cube_1", "cube_2", "cube_3")}
        finger = robot.data.joint_pos.torch[:, fid].mean(1).cpu().numpy()
        arm.process_actions(torch.zeros(n, arm.action_dim, device=dev))
        scale = arm._scale[:, 0].cpu().numpy().astype(np.float64)
        act = policy(obs)
        a = act.cpu().numpy()
        for i in range(n):
            ph = experts[i].phase + ("*" if experts[i].done else "")
            e = experts[i].act(eef[i], {k: v[i] for k, v in cubes.items()}, float(finger[i]))
            lab = np.concatenate([0.5 * e[:3] / scale[i], e[3:]])
            d = err.setdefault(ph, [0, np.zeros(7), 0])
            d[0] += 1
            d[1] += np.abs(a[i] - lab)
            d[2] += int(np.sign(a[i, 6]) != np.sign(lab[6]))
        red, blue = cubes["cube_2"], cubes["cube_1"]
        min_reach = np.minimum(min_reach, np.linalg.norm(red - eef, axis=1))
        lifted |= red[:, 2] > 0.045
        over_blue |= (red[:, 2] > 0.045) & (np.linalg.norm((red - blue)[:, :2], axis=1) < 0.015)
        st = cbf.seated(red, blue)
        seated_ever |= st
        released_seated |= st & (finger > 0.035)
        obs, _, dones, _ = env.step(act)
        if t in (100, 200, 300, 450, args_cli.steps - 1):
            print(f"t={t}: min reach dist median {np.median(min_reach) * 100:.1f}cm, reached<1cm {np.mean(min_reach < 0.01):.3f}, "
                  f"lifted {lifted.mean():.3f}, over blue {over_blue.mean():.3f}, seated {seated_ever.mean():.3f}, "
                  f"released seated {released_seated.mean():.3f}", flush=True)
print("shadow-expert phase: count, mean |policy - expert| per dim, grip sign mismatch rate")
for ph, (c, s, g) in sorted(err.items(), key=lambda kv: -kv[1][0]):
    print(f"  {ph:10s} n={c:7d}  {np.round(s / c, 3).tolist()}  grip_mismatch={g / c:.3f}")
eef = (u.scene["ee_frame"].data.target_pos_w.torch[:, 0, :] - origins).cpu().numpy()
print("final eef z median", np.median(eef[:, 2]), "final red-eef dist median", np.median(np.linalg.norm(cubes["cube_2"] - eef, axis=1)), flush=True)
os._exit(0)
