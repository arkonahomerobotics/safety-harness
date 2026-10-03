"""Body-sweep diagnostic: for handoff-start envs specifically, track the minimum distance from
EVERY robot body link to A and to B over the first N steps, plus waist joint deltas, to check
whether the torso/elbow/forearm (not just the wrist) sweeps through the tower even though the wrist
itself starts 13-15cm away -- the waist joints are part of this task's action space, so a torso
rotation can swing the whole arm through space the wrist reference point never visits."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-Stage2-RC-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--seed", type=int, default=123)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
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
    return lambda o: mlp((o - mean) / (std + 1e-2))


env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

with torch.inference_mode():
    obs, _ = env.reset()
    dev, N = env.device, env.num_envs
    env.episode_length_buf[:] = 0
    policy = load_actor(args_cli.checkpoint, dev)
    robot = env.scene["robot"]
    object_, block_b, block_c = env.scene["object"], env.scene["block_b"], env.scene["block_c"]
    o_ = env.scene.env_origins
    body_names = robot.data.body_names
    print(f"{len(body_names)} robot bodies: {body_names}")

    waist_ids, _ = robot.find_joints(["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])
    waist0 = robot.data.joint_pos.torch[:, waist_ids].clone()

    pc0 = block_c.data.root_pos_w.torch - o_
    expert_like = pc0[:, 2] > 0.74
    joint_diff = (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch).abs().sum(dim=1)
    touched = joint_diff > 0.05
    handoff_like = touched & ~expert_like
    print(f"handoff-like envs: {int(handoff_like.sum())}/{N}")

    a_z0 = (object_.data.root_pos_w.torch - o_)[:, 2].clone()
    b_z0 = (block_b.data.root_pos_w.torch - o_)[:, 2].clone()

    # per-env running minimum distance from each body to A and to B
    min_dist_a = torch.full((N, len(body_names)), 1e9, device=dev)
    min_dist_b = torch.full((N, len(body_names)), 1e9, device=dev)
    off_step = torch.full((N,), -1, dtype=torch.long, device=dev)

    for step in range(args_cli.steps):
        pa = object_.data.root_pos_w.torch - o_
        pb = block_b.data.root_pos_w.torch - o_
        body_pos = robot.data.body_pos_w.torch - o_.unsqueeze(1)  # (N, B, 3)
        d_a = torch.linalg.norm(body_pos - pa.unsqueeze(1), dim=2)  # (N, B)
        d_b = torch.linalg.norm(body_pos - pb.unsqueeze(1), dim=2)
        min_dist_a = torch.minimum(min_dist_a, d_a)
        min_dist_b = torch.minimum(min_dist_b, d_b)

        a_z = pa[:, 2]
        b_z = pb[:, 2]
        off_now = (a_z - b_z) < 0.025
        newly_off = off_now & (off_step < 0) & (torch.arange(N, device=dev) >= 0)
        off_step[newly_off] = step

        act = policy(obs["policy"]).clamp(-1, 1)
        obs, rew, term, trunc, extras = env.step(act)

    waist_now = robot.data.joint_pos.torch[:, waist_ids]
    waist_delta = (waist_now - waist0).abs()

    if handoff_like.any():
        hi = torch.nonzero(handoff_like).flatten()
        print(f"\n--- handoff-start envs, min distance to A over first {args_cli.steps} steps ---")
        ma = min_dist_a[hi]  # (H, B)
        closest_body_idx = ma.argmin(dim=1)
        overall_min = ma.min(dim=1).values
        print(f"overall min body-to-A distance: min={overall_min.min():.3f} max={overall_min.max():.3f} mean={overall_min.mean():.3f} median={overall_min.median():.3f}")
        for i in range(min(10, hi.numel())):
            bidx = int(closest_body_idx[i])
            print(f"  env {int(hi[i])}: closest body={body_names[bidx]} dist={float(overall_min[i]):.3f}  "
                  f"off_step={int(off_step[hi[i]])}  waist_delta_deg={[round(float(v)*57.3,1) for v in waist_delta[hi[i]]]}")
        print(f"\nwaist delta (deg) over handoff envs: mean_abs={[round(float(v)*57.3,1) for v in waist_delta[hi].abs().mean(0)]} "
              f"max_abs={[round(float(v)*57.3,1) for v in waist_delta[hi].abs().max(0).values]}")
        # how many handoff envs have SOME body within 5cm of A within the window
        close5 = (overall_min < 0.05).float().mean()
        close10 = (overall_min < 0.10).float().mean()
        print(f"fraction of handoff envs with some body <5cm of A: {close5:.2f}   <10cm: {close10:.2f}")

simulation_app.close()
