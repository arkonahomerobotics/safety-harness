"""Diagnose why block_a_off_b holds ~91% after 150 iterations of BC warm-start training. Runs the
BC checkpoint full-episode on Isaac-Stack-Blocks-G1-RL-Stage2-RC-v0 (the real snapshot-mix reset),
either deterministically or at a given sampling std, and reports:
  - success / block_a_off_b / C-placed rates split by start type (expert-carry / handoff / default)
  - for handoff starts: wrist-to-A distance at step 0 (reset)
  - step histogram of block_a_off_b terminations, plus whether C had ever been lifted by then

Start-type classification (same scene, no privileged info needed): expert-carry = C well above the
table at reset (still held); among the rest, "touched" = robot joints differ meaningfully from the
architecture default (handoff snapshots overwrite joints with a recorded policy pose) vs "default"
= joints are the plain reset default (reset_stage2_mixed_snapshots never touched this env)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-Stage2-RC-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--std", type=float, default=0.0, help="0 = deterministic (mean action), >0 = sample with this std")
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
    widx = robot.data.body_names.index("left_wrist_yaw_link")
    tm = env.termination_manager

    # -- classify start type at reset --
    pc0 = block_c.data.root_pos_w.torch - o_
    expert_like = pc0[:, 2] > 0.74
    joint_diff = (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch).abs().sum(dim=1)
    touched = joint_diff > 0.05
    handoff_like = touched & ~expert_like
    default_like = ~touched & ~expert_like
    print(f"start types: expert-carry={int(expert_like.sum())} handoff={int(handoff_like.sum())} default={int(default_like.sum())}")

    # -- handoff hand-to-A distance at reset --
    pa0 = object_.data.root_pos_w.torch - o_
    wrist0 = robot.data.body_pos_w.torch[:, widx] - o_
    hand_to_a0 = torch.linalg.norm(pa0 - wrist0, dim=1)
    if handoff_like.any():
        d = hand_to_a0[handoff_like]
        print(f"handoff hand-to-A distance at reset: min={d.min():.3f} max={d.max():.3f} mean={d.mean():.3f} median={d.median():.3f}")
        print(f"  <5cm: {int((d < 0.05).sum())}/{d.numel()}  <10cm: {int((d < 0.10).sum())}/{d.numel()}")

    ever_lifted = torch.zeros(N, dtype=torch.bool, device=dev)
    off_step = torch.full((N,), -1, dtype=torch.long, device=dev)
    off_lifted_at_term = torch.zeros(N, dtype=torch.bool, device=dev)
    outcome = torch.full((N,), -1, dtype=torch.long, device=dev)  # -1 running, 0 success, 1 off_b, 2 other

    for step in range(env.max_episode_length + 5):
        obs_t = obs["policy"]
        mean_act = policy(obs_t)
        if args_cli.std > 0:
            act = (mean_act + args_cli.std * torch.randn_like(mean_act)).clamp(-1, 1)
        else:
            act = mean_act.clamp(-1, 1)
        obs, rew, term, trunc, extras = env.step(act)
        pc = block_c.data.root_pos_w.torch - o_
        grasped_proxy = torch.linalg.norm((robot.data.body_pos_w.torch[:, widx] - o_) - pc, dim=1) < 0.14
        ever_lifted |= grasped_proxy & (pc[:, 2] > 0.75)
        off_fired = tm.get_term("block_a_off_b") & (off_step < 0)
        off_step[off_fired] = step
        off_lifted_at_term[off_fired] = ever_lifted[off_fired]
        done = (term | trunc) & (outcome < 0)
        if done.any():
            succ_flag = getattr(env.reward_manager.get_term_cfg("staged_achievements_3stack").func, "_last_success", torch.zeros(N, dtype=torch.bool, device=dev))
            off_b = tm.get_term("block_a_off_b") & done
            outcome[done & succ_flag] = 0
            outcome[off_b & (outcome < 0)] = 1
            outcome[done & (outcome < 0)] = 2
        if bool((outcome >= 0).all()):
            break

    def rate(mask_outer, mask_outcome):
        n = int(mask_outer.sum())
        if n == 0:
            return "n/a"
        return f"{int((mask_outer & mask_outcome).sum())}/{n} ({100*(mask_outer & mask_outcome).float().sum()/max(n,1):.1f}%)"

    off_b_mask = outcome == 1
    succ_mask = outcome == 0
    for name, mask in [("expert-carry", expert_like), ("handoff", handoff_like), ("default", default_like)]:
        print(f"{name}: success={rate(mask, succ_mask)} block_a_off_b={rate(mask, off_b_mask)}")

    tripped = off_step >= 0
    print(f"\nTOTAL block_a_off_b trips: {int(tripped.sum())}/{N}")
    if tripped.any():
        steps = off_step[tripped]
        bins = [(0, 10), (10, 30), (30, 100), (100, 300), (300, 500)]
        for lo, hi in bins:
            m = (steps >= lo) & (steps < hi)
            print(f"  step [{lo},{hi}): {int(m.sum())}  (of which C ever lifted first: {int((off_lifted_at_term[tripped] & m).sum())})")

simulation_app.close()
