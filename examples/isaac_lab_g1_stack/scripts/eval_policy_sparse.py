"""Evaluate a KAN-49 from-scratch sparse-reward G1 checkpoint (milestone_stack_reward_g1), with its
deterministic actor, on a genuinely COLD start -- the expert-start snapshot mix is forced to 0 here
regardless of the task cfg's own default, since this script's whole purpose is to measure real,
unassisted end-to-end competence, not performance inflated by the training-time cold-start allowance.

Same pre-reset-trap discipline as eval_policy.py: Isaac Lab auto-resets an env inside the very
step() call whose term/trunc comes back true, so state read on any later iteration already belongs
to the next episode. Every per-step judging read happens before that iteration's own env.step().
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-Sparse-S1-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--noise", type=float, default=0.0, help="gaussian action noise, to mimic PPO's sampling")
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
env_cfg.scene.robot_pov_cam = None  # XR-teleop leftover, crashes sensor init unless enable_cameras is set
# Force a genuinely cold evaluation -- no expert-start freebies, regardless of the task cfg's
# own training-time default (KAN-49's 15% minimal allowance is a training-only concession).
if hasattr(env_cfg.events, "reset_from_snapshots"):
    env_cfg.events.reset_from_snapshots.params["prob"] = 0.0
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

NAMES = ("reach", "grasp", "lift", "over", "placed")

with torch.inference_mode():
    obs, _ = env.reset()
    dev, N = env.device, env.num_envs
    env.episode_length_buf[:] = 0
    policy = load_actor(args_cli.checkpoint, dev)
    stack_term = env.reward_manager.get_term_cfg("stack").func
    ever_complete = torch.zeros(N, dtype=torch.bool, device=dev)
    # Full milestone ladder, not just the final "placed" flag -- the expert-start snapshot mix
    # (forced to 0 above for this eval, but its effect on the TRAINING logs needs separating out)
    # places the block already held and roughly aligned, so "over" itself could be just as
    # free-ridden during training as "placed" was found to be. Track the whole chain cold so that
    # doesn't get missed a second time.
    ever_milestone = torch.zeros(N, len(NAMES), dtype=torch.bool, device=dev)

    outcome = torch.full((N,), -1, dtype=torch.long, device=dev)  # -1 running, 0 placed, 1 other
    final_placed = torch.zeros(N, dtype=torch.bool, device=dev)
    ep_len = torch.zeros(N, device=dev)

    for step in range(env.max_episode_length + 5):
        pre_placed = stack_term._complete_now.clone()
        pre_done_milestones = stack_term._done.clone()
        act = policy(obs["policy"])
        if args_cli.noise > 0:
            act = act + args_cli.noise * torch.randn_like(act)
        act = act.clamp(-1, 1)
        obs, rew, term, trunc, extras = env.step(act)
        ever_complete |= pre_placed
        ever_milestone |= pre_done_milestones
        done = (term | trunc) & (outcome < 0)
        final_placed = torch.where(done, pre_placed, final_placed)
        if done.any():
            outcome[done] = 0
            ep_len[done] = step + 1
        if bool((outcome >= 0).all()):
            break

print("COLD-START milestone-ever rates (full ladder, no expert-start assistance):")
for i, name in enumerate(NAMES):
    print(f"  {name:>6}_ever: {int(ever_milestone[:, i].sum())}/{N} ({ever_milestone[:, i].float().mean()*100:.2f}%)")
print(f"COLD-START FULL-EPISODE end state: placed & standing at episode end: {int(final_placed.sum())}/{N}")
print(f"ACHIEVED-AT-ANY-POINT (same completion definition, same def as training): {int(ever_complete.sum())}/{N}")
print(
    f"EVAL {args_cli.checkpoint}: success {int(final_placed.sum())}/{N} "
    f"({final_placed.float().mean()*100:.2f}%)  mean success ep len "
    f"{ep_len[final_placed].mean() if final_placed.any() else float('nan'):.0f} steps"
)

simulation_app.close()
