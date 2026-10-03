"""Run a checkpoint as N sequential single-env episodes (num_envs=1, natural randomized resets --
no pose overrides) to compare against the batched (num_envs=100) success rate for the same
checkpoint. Isaac Lab's env.reset() re-randomizes block_a/block_b each call via the task's own
reset events, so looping reset() in one process gives real independent episodes without the
per-env repro technique's batch-size-dependent physics divergence (every env here IS env 0 of a
num_envs=1 sim, so there's no cross-batch comparison left to diverge from).

Uses the same end-state check and reset-read fix as eval_policy.py --full_episode: the scene is
snapshotted the step before an episode's time_out fires, since Isaac Lab resets an env inside the
very step() call that times it out."""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=20)
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


env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env_cfg.seed = args_cli.seed
env_cfg.scene.robot_pov_cam = None
env_cfg.terminations.success = None  # full episode, judge the end state ourselves
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

with torch.inference_mode():
    dev = env.device
    policy = load_actor(args_cli.checkpoint, dev)
    object_ = env.scene["object"]
    block_b = env.scene["block_b"]
    robot = env.scene["robot"]
    widx = robot.data.body_names.index("left_wrist_yaw_link")
    o_ = env.scene.env_origins

    def stacked_now() -> bool:
        pa = object_.data.root_pos_w.torch[0] - o_[0]
        pb = block_b.data.root_pos_w.torch[0] - o_[0]
        w = robot.data.body_pos_w.torch[0, widx] - o_[0]
        dxy = float(torch.linalg.norm((pa - pb)[:2]))
        dz = float(pa[2] - pb[2] - 0.045)
        still = float(torch.linalg.norm(object_.data.root_vel_w.torch[0, :3])) < 0.03
        b_on_table = abs(float(pb[2]) - env.cfg.scene.block_b.init_state.pos[2]) < 0.01
        clear = float(torch.linalg.norm(pa - w)) > 0.11
        return (dxy < 0.025) and (abs(dz) < 0.012) and still and b_on_table and clear  # Kaoru-approved 2.5cm, 2026-10-03

    print(f"max_episode_length={env.max_episode_length}", flush=True)
    results = []
    offsets = []  # (dxy, dz) for every non-success episode, to split beside-base vs offset-on-base
    for ep in range(args_cli.episodes):
        print(f"EPISODE {ep} starting...", flush=True)
        obs, _ = env.reset()
        env.episode_length_buf[:] = 0
        pre_stacked = False
        final_outcome = "timeout"
        t0 = time.time()
        for step in range(env.max_episode_length + 5):
            pre_stacked = stacked_now()
            pre_pa = object_.data.root_pos_w.torch[0] - o_[0]
            pre_pb = block_b.data.root_pos_w.torch[0] - o_[0]
            act = policy(obs["policy"]).clamp(-1, 1)
            obs, rew, term, trunc, extras = env.step(act)
            if step % 50 == 0:
                print(f"  ep={ep} step={step} elapsed={time.time()-t0:.1f}s term={bool(term[0])} trunc={bool(trunc[0])}", flush=True)
            if bool(term[0]) or bool(trunc[0]):
                dropped = float(pre_pa[2]) < 0.5 or float(pre_pb[2]) < 0.5
                final_outcome = "success" if pre_stacked else ("dropped" if dropped else "timeout")
                if not pre_stacked and not dropped:
                    off = pre_pa - pre_pb
                    offsets.append((float(torch.linalg.norm(off[:2])), float(off[2] - 0.045)))
                break
        results.append(final_outcome)
        print(f"EPISODE {ep}: {final_outcome}")

    succ = sum(1 for r in results if r == "success")
    dropped = sum(1 for r in results if r == "dropped")
    timeout = sum(1 for r in results if r == "timeout")
    print(f"SEQUENTIAL SINGLE-ENV RESULT: {args_cli.checkpoint}")
    print(f"success {succ}/{args_cli.episodes}  dropped {dropped}  timeout(not-stacked) {timeout}")
    beside_base = sum(1 for dxy, dz in offsets if dz < -0.02)
    offset_on_base = len(offsets) - beside_base
    print(f"of the {len(offsets)} non-success/non-dropped: beside_base(dz<-0.02)={beside_base} "
          f"offset_on_base(dz~0, dxy>0.025)={offset_on_base}")
    for dxy, dz in offsets:
        print(f"  final_dxy={dxy:.3f} final_dz={dz:+.3f}")

simulation_app.close()
