"""Test the "termination is cheaper than staying alive" hypothesis: run a checkpoint full-episode
and log each reward TERM's per-episode cumulative value, split by how the episode ended
(block_a_off_b / timeout / block_c_dropped / block_b_dropped), to check whether a per-step
negative term (action_rate/joint_vel/grip_smoothness) outweighs the achievement rewards in
episodes that flail without progress -- which would make ending the episode early (no penalty is
currently attached to block_a_off_b itself) a higher reward-RATE strategy than persisting.

Reads IsaacLab's own per-term episode accumulator (RewardManager._episode_sums), the same buffer
"Episode_Reward/*" in the training log is built from, latched the step BEFORE each env's own
reset (same reset-read-trap avoidance as every other diagnostic in this session)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-Stage2-RC-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
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
    tm = env.termination_manager
    rm = env.reward_manager
    term_names = [t for t in rm.active_terms]
    print(f"reward terms: {term_names}")

    outcome = torch.full((N,), -1, dtype=torch.long, device=dev)  # -1 running, 0 off_b, 1 timeout, 2 c_dropped, 3 b_dropped
    ep_len = torch.zeros(N, device=dev)
    per_term_sum = {t: torch.zeros(N, device=dev) for t in term_names}

    for step in range(env.max_episode_length + 5):
        # latch pre-step episode sums (the reset-read-trap pattern: read BEFORE this step's env.step())
        pre_sums = {t: rm._episode_sums[t].clone() for t in term_names} if hasattr(rm, "_episode_sums") else None
        act = policy(obs["policy"]).clamp(-1, 1)
        obs, rew, term, trunc, extras = env.step(act)
        done = (term | trunc) & (outcome < 0)
        if done.any():
            off_b = tm.get_term("block_a_off_b") & done
            c_drop = tm.get_term("block_c_dropped") & done & ~off_b
            b_drop = tm.get_term("block_b_dropped") & done & ~off_b & ~c_drop
            tout = tm.get_term("time_out") & done & ~off_b & ~c_drop & ~b_drop
            outcome[off_b] = 0
            outcome[c_drop] = 2
            outcome[b_drop] = 3
            outcome[tout] = 1
            outcome[done & (outcome < 0)] = 4  # unexpected
            ep_len[done] = step + 1
            if pre_sums is not None:
                for t in term_names:
                    per_term_sum[t][done] = pre_sums[t][done]
        if bool((outcome >= 0).all()):
            break

    names = {0: "block_a_off_b", 1: "timeout", 2: "block_c_dropped", 3: "block_b_dropped", 4: "other"}
    print(f"\n{'outcome':>16} {'n':>5} {'mean_ep_len':>12} " + " ".join(f"{t[:14]:>15}" for t in term_names))
    for code, name in names.items():
        mask = outcome == code
        n = int(mask.sum())
        if n == 0:
            continue
        row = f"{name:>16} {n:>5} {float(ep_len[mask].mean()):>12.1f} "
        row += " ".join(f"{float(per_term_sum[t][mask].mean()):>15.4f}" for t in term_names)
        print(row)

simulation_app.close()
