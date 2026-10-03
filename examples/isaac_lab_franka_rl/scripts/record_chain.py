"""Close-up video of one env building the full tower by skill chaining (stage-1 policy, then the stage-2 skill
with rebound cube roles after red has stood stacked for --handoff_steps). Several episodes back to back,
normal starts.

usage: record_chain.py --stage1 CK1 --stage2 CK2 --out DIR [--episodes 6] [--episode_s 12]

Isaac Lab 3.x port: ``render_mode="rgb_array"`` + gym RecordVideo no longer produce frames (env.render() returns None
and warns), so frames come from a dedicated 1280x720 demo camera placed at the Brev viewer pose (eye (1.0, 0.45, 0.42),
look-at (0.5, 0.0, 0.06), env-relative) and are written with imageio to <out>/chain.mp4.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-RL-FullStack-Snap-Sparse-v0")
parser.add_argument("--stage1", type=str, required=True)
parser.add_argument("--stage2", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--episodes", type=int, default=6)
parser.add_argument("--episode_s", type=float, default=12.0)
parser.add_argument("--handoff_steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import copy
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.stack.mdp.robosuite_rewards import _in_contact
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env_cfg.seed = args_cli.seed
env_cfg.episode_length_s = args_cli.episode_s
EYE, LOOKAT = (1.0, 0.45, 0.42), (0.5, 0.0, 0.06)
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402

env_cfg.scene.demo_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/demo_cam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)),
)
env_cfg.events.reset_from_snapshot = None
s2 = copy.deepcopy(env_cfg.observations.policy)
roles = {"cube_1_cfg": SceneEntityCfg("cube_2"), "cube_2_cfg": SceneEntityCfg("cube_3"), "cube_3_cfg": SceneEntityCfg("cube_1")}
for term in ("object", "cube_positions", "cube_orientations"):
    getattr(s2, term).params.update(roles)
env_cfg.observations.policy_s2 = s2
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

os.makedirs(args_cli.out, exist_ok=True)
steps = int(args_cli.episode_s / (env_cfg.sim.dt * env_cfg.decimation)) * args_cli.episodes
env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
_o = u.scene.env_origins
u.scene["demo_cam"].set_world_poses_from_view(
    torch.tensor([EYE], device=u.device) + _o, torch.tensor([LOOKAT], device=u.device) + _o
)
frames = []
pols = []
for ck in (args_cli.stage1, args_cli.stage2):
    r = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    r.load(ck, map_location=agent_cfg.device)
    pols.append(r.get_inference_policy(device=u.device))
LIFT_Z, THR = 0.0403, 0.01

obs = env.get_observations()
run = torch.zeros(1, dtype=torch.long, device=u.device)
handed = torch.zeros(1, dtype=torch.bool, device=u.device)
log = []
with torch.inference_mode():
    for t in range(steps + 1):
        red_z = u.scene["cube_2"].data.root_pos_w.torch[:, 2] - u.scene.env_origins[:, 2]
        g_red = _in_contact(u, "left_finger_red_contact", THR) & _in_contact(u, "right_finger_red_contact", THR)
        s1 = (red_z > LIFT_Z) & _in_contact(u, "red_blue_contact", THR) & ~g_red
        run = torch.where(s1, run + 1, torch.zeros_like(run))
        if not handed.item() and run.item() >= args_cli.handoff_steps:
            log.append(f"step {t}: handoff to stage 2")
        handed |= run >= args_cli.handoff_steps
        obs2 = obs.clone()
        obs2["policy"] = obs["policy_s2"]
        actions = pols[1](obs2) if handed.item() else pols[0](obs)
        obs, _, dones, _ = env.step(actions)
        frames.append(u.scene["demo_cam"].data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy())
        for p in pols:
            p.reset(dones)
        if dones.item():
            log.append(f"step {t}: episode end")
            handed[:] = False
            run[:] = 0
import imageio.v2 as imageio  # noqa: E402

video_path = os.path.join(args_cli.out, "chain.mp4")
imageio.mimwrite(video_path, frames, fps=int(round(1.0 / u.step_dt)))
env.close()
print("\n".join(log))
print(f"VIDEO_DONE {steps} steps -> {video_path}")
simulation_app.close()
