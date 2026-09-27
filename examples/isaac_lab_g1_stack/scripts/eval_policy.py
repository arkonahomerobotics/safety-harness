"""Evaluate a trained G1 stacking policy (RSL-RL checkpoint) with its deterministic actor.

Each env runs exactly one episode from a normal start (no reverse-curriculum snapshots); the outcome
is read from the env's own termination terms: success (seated, settled, released), block dropped
off the table, or time out. This is also the driver the safety-harness tests wrap.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--video", type=str, default="")
parser.add_argument("--full_episode", action="store_true", help="no success termination: run all 500 steps, judge the FINAL state")
parser.add_argument("--noise", type=float, default=0.0, help="gaussian action noise, to mimic PPO's sampling")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def load_actor(path, device):
    sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    ks = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    layers = []
    for j, k in enumerate(ks):
        w, b = sd[f"mlp.{k}.weight"], sd[f"mlp.{k}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0]).to(device)
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if j < len(ks) - 1:
            layers.append(nn.ELU())
    mlp = nn.Sequential(*layers).eval()
    mean, std = sd["obs_normalizer._mean"].to(device), sd["obs_normalizer._std"].to(device)
    return lambda o: mlp((o - mean) / (std + 1e-2))  # identical to rsl_rl EmpiricalNormalization.forward


env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed
if args_cli.full_episode:
    env_cfg.terminations.success = None
if args_cli.video:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    env_cfg.scene.demo_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCam", update_period=0.0, height=540, width=960, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
    )
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

with torch.inference_mode():
    obs, _ = env.reset()
    dev, N = env.device, env.num_envs
    # the RL wrapper randomizes initial episode lengths for training; evaluate full episodes instead
    env.episode_length_buf[:] = 0
    policy = load_actor(args_cli.checkpoint, dev)
    if args_cli.video:
        o = env.scene.env_origins
        mid = torch.tensor([-0.20, 0.36, 0.75], device=dev)
        env.scene["demo_cam"].set_world_poses_from_view((mid + torch.tensor([0.30, 0.50, 0.32], device=dev)).repeat(N, 1) + o, mid.repeat(N, 1) + o)
    tm = env.termination_manager
    outcome = torch.full((N,), -1, dtype=torch.long, device=dev)  # -1 running, 0 success, 1 dropped, 2 timeout
    ep_len = torch.zeros(N, device=dev)
    frames = []
    for step in range(env.max_episode_length + 5):
        act = policy(obs["policy"])
        if args_cli.noise > 0:
            act = act + args_cli.noise * torch.randn_like(act)
        act = act.clamp(-1, 1)
        obs, rew, term, trunc, extras = env.step(act)
        if args_cli.video:
            frames.append(env.scene["demo_cam"].data.output["rgb"][0, ..., :3].clone().cpu())
        done = (term | trunc) & (outcome < 0)
        if done.any():
            succ = (tm.get_term("success") if "success" in tm.active_terms else torch.zeros_like(done)) & done
            drop = (tm.get_term("block_a_dropped") | tm.get_term("block_b_dropped")) & done & ~succ
            outcome[succ] = 0
            outcome[drop] = 1
            outcome[done & ~succ & ~drop] = 2
            ep_len[done] = step + 1
        if bool((outcome >= 0).all()):
            break
    if args_cli.full_episode or "success" not in tm.active_terms:
        # judge the end state: blue seated on red, at rest, hand clear
        o_ = env.scene.env_origins
        pa = env.scene["object"].data.root_pos_w.torch - o_
        pb = env.scene["block_b"].data.root_pos_w.torch - o_
        r = env.scene["robot"]
        w = r.data.body_pos_w.torch[:, r.data.body_names.index("left_wrist_yaw_link")] - o_
        dxy = torch.linalg.norm((pa - pb)[:, :2], dim=1)
        dz = pa[:, 2] - pb[:, 2] - 0.045
        still = torch.linalg.norm(env.scene["object"].data.root_vel_w.torch[:, :3], dim=1) < 0.03
        stacked = (dxy < 0.015) & (dz.abs() < 0.012) & still & (torch.linalg.norm(pa - w, dim=1) > 0.11)
        outcome = torch.where(stacked, torch.zeros_like(outcome), torch.where(outcome == 1, outcome, torch.full_like(outcome, 2)))
        print(f"FULL-EPISODE end state: stacked & released at t=10s: {int(stacked.sum())}/{N}")
    s = (outcome == 0)
    print(f"EVAL {args_cli.checkpoint}: success {int(s.sum())}/{N} ({s.float().mean()*100:.1f}%)  "
          f"dropped {int((outcome == 1).sum())}  timeout {int((outcome == 2).sum())}  "
          f"mean success ep len {ep_len[s].mean() if s.any() else float('nan'):.0f} steps")
    if args_cli.video and frames:
        import imageio

        with imageio.get_writer(args_cli.video, fps=30, codec="libx264", quality=8) as w:
            for fr in frames:
                w.append_data(fr.numpy().astype("uint8"))
        print(f"VIDEO {args_cli.video} frames={len(frames)} env0 outcome={int(outcome[0])}")

simulation_app.close()
