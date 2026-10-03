"""Capture real Stage-1 handoff states: run model_5797 (the best Stage-1 checkpoint) on the base
2-block task, and the moment each env's success streak first reaches success_hold_steps AND the
left hand is genuinely clear of block A (see HAND_CLEAR_MARGIN below), snapshot that env's robot
joint state + block_a/block_b poses.

Added hand-clearance filter (2026-10-03): Stage 1's own staged_achievements "hand clear" check
measures WRIST distance only. A diagnostic on the first batch of handoff snapshots found the hand
extends ~9-10cm past the wrist, so wrist-clear (>11cm) states routinely had a FINGERTIP
(left_hand_middle_1_link) only 4-5cm from block A -- Stage-2 training from these snapshots then saw
block_a_off_b fire on ~97-100% of handoff-start episodes within the first 30 steps, well before any
grasp of block C, as the policy's early waist rotation swept the already-close hand through the
tower. This filter only gates WHICH successful moments get saved here; it deliberately does not
touch staged_achievements' own hand_clear_dist (Stage 1's live training/success definition is
unaffected, per instruction).

Captured live, mid-episode, specifically to avoid the reset-read trap eval_policy.py's own fix
addressed: reading a reward term's counters (or, as in diagnose_failures_scan.py's first version,
reading them after a batch-wide reset) gives the POST-reset zeroed values, not the real ones. Here
there's no reset involved at all -- every env is snapshotted at the exact step its live streak
value crosses the threshold (and stays below it to keep checking on later steps if the hand isn't
clear enough yet, trading snapshot yield for quality), same format `rl_events_g1.reset_from_snapshots`
already reads for Stage 1's own reverse-curriculum: {"joint_pos": (S, n_dof), "cube_pose": (S, 2, 7)},
cube_names ("object", "block_b"), poses stored relative to each env's own origin."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--out", type=str, default="/workspace/isaaclab/g1_stage1_handoff_snapshots.pt")
parser.add_argument("--hand_clear_margin", type=float, default=0.10, help="min distance from ANY left-hand link (palm+fingers) to block A")
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
env_cfg.scene.robot_pov_cam = None
env_cfg.terminations.success = None
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

with torch.inference_mode():
    obs, _ = env.reset()
    dev, N = env.device, env.num_envs
    env.episode_length_buf[:] = 0
    policy = load_actor(args_cli.checkpoint, dev)
    robot = env.scene["robot"]
    object_, block_b = env.scene["object"], env.scene["block_b"]
    o_ = env.scene.env_origins
    staged_term = env.reward_manager.get_term_cfg("staged_achievements").func
    threshold = 15  # success_hold_steps, see g1_block_stack_rl_env_cfg.py's _STACK

    LEFT_HAND_LINKS = [
        "left_hand_palm_link", "left_hand_index_0_link", "left_hand_middle_0_link", "left_hand_thumb_0_link",
        "left_hand_index_1_link", "left_hand_middle_1_link", "left_hand_thumb_1_link", "left_hand_thumb_2_link",
    ]
    hand_link_ids = [robot.data.body_names.index(n) for n in LEFT_HAND_LINKS]

    captured = torch.zeros(N, dtype=torch.bool, device=dev)
    snap_joint = torch.zeros(N, robot.data.joint_pos.torch.shape[1], device=dev)
    snap_cube = torch.zeros(N, 2, 7, device=dev)
    streak_satisfied_steps = torch.zeros(N, dtype=torch.long, device=dev)  # how many steps streak>=threshold but hand not clear enough
    best_clear_while_streak_ok = torch.zeros(N, device=dev)  # diagnostic: best hand clearance seen while streak was satisfied

    for step in range(env.max_episode_length + 5):
        act = policy(obs["policy"]).clamp(-1, 1)
        obs, rew, term, trunc, extras = env.step(act)
        streak = staged_term._success_streak
        streak_ok = streak >= threshold

        pa = object_.data.root_pos_w.torch - o_
        hand_pos = robot.data.body_pos_w.torch[:, hand_link_ids] - o_.unsqueeze(1)  # (N, H, 3)
        hand_min_dist = torch.linalg.norm(hand_pos - pa.unsqueeze(1), dim=2).min(dim=1).values
        hand_clear = hand_min_dist > args_cli.hand_clear_margin

        streak_satisfied_steps[streak_ok & ~hand_clear] += 1  # diagnostic: how often streak fires without real clearance
        best_clear_while_streak_ok = torch.where(streak_ok, torch.maximum(best_clear_while_streak_ok, hand_min_dist), best_clear_while_streak_ok)

        newly = streak_ok & hand_clear & (~captured)
        if newly.any():
            snap_joint[newly] = robot.data.joint_pos.torch[newly]
            pb = block_b.data.root_pos_w.torch - o_
            snap_cube[newly, 0] = torch.cat([pa, object_.data.root_quat_w.torch], dim=1)[newly]
            snap_cube[newly, 1] = torch.cat([pb, block_b.data.root_quat_w.torch], dim=1)[newly]
            captured |= newly
        if step % 50 == 0:
            print(f"step={step} captured={int(captured.sum())}/{N}  (streak-ok-but-hand-not-clear so far: {int((streak_satisfied_steps > 0).sum())})")
        if bool(captured.all()):
            break

    keep = captured
    torch.save({"joint_pos": snap_joint[keep].cpu(), "cube_pose": snap_cube[keep].cpu()}, args_cli.out)
    print(f"CAPTURED {int(keep.sum())}/{N} Stage-1 handoff states (hand_clear_margin={args_cli.hand_clear_margin}) -> {args_cli.out}")
    print(f"envs that reached success_hold_steps but never had a genuinely clear hand within the episode: "
          f"{int(((streak_satisfied_steps > 0) & ~captured).sum())}/{N}")
    reached = streak_satisfied_steps > 0
    if reached.any():
        b = best_clear_while_streak_ok[reached]
        print(f"best hand-link clearance ever seen while streak was satisfied, over envs that reached it: "
              f"min={b.min():.3f} p25={b.quantile(0.25):.3f} median={b.median():.3f} p75={b.quantile(0.75):.3f} max={b.max():.3f}")

simulation_app.close()
