"""Check that snapshot resets reproduce the recorded scene (Isaac Lab 3.x port of the Brev test_snapshot_reset.py).

Stage 1 (default: Robosuite-Snap-Sparse + expert_carry_onto_blue.pt): right after reset, is red in the gripper, above
blue, with both finger contact sensors firing? Then hold (gripper closed, zero arm delta) for --hold_steps, then open
the gripper for --release_steps and report robosuite "stacked" (red lifted, touching blue, not grasped).
Stage 2 (--stage2, TowerOnly/Stage2Skill tasks + a stage-2 snapshot file): is red still seated on blue after reset?
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-RL-Robosuite-Snap-Sparse-v0")
parser.add_argument("--snapshot_path", type=str, default="/workspace/isaaclab/snapshots/expert_carry_onto_blue.pt")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--hold_steps", type=int, default=10)
parser.add_argument("--release_steps", type=int, default=30)
parser.add_argument("--stage2", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.contrib.stack.mdp.robosuite_rewards import _in_contact  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

CUBE_H, LIFT_Z, THR = 0.0468, 0.0403, 0.01

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.events.reset_from_snapshot.params = {"snapshot_path": args_cli.snapshot_path, "prob": 1.0}
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
print("reset event order:", env.event_manager.active_terms.get("reset"))

snaps = torch.load(args_cli.snapshot_path, weights_only=True)
cp = snaps["cube_pose"]
sdz = cp[:, 1, 2] - cp[:, 0, 2]
sxy = torch.linalg.norm(cp[:, 1, :2] - cp[:, 0, :2], dim=1)
print(f"in snapshot file: n={len(cp)} red-blue dz median={sdz.median() * 100:.2f}cm  xy median={sxy.median() * 100:.2f}cm  "
      f"red quat[0] (xyzw)={[round(v, 3) for v in cp[0, 1, 3:].tolist()]}")


def report(tag):
    o = env.scene.env_origins
    blue = env.scene["cube_1"].data.root_pos_w.torch - o
    red = env.scene["cube_2"].data.root_pos_w.torch - o
    ee = env.scene["ee_frame"].data.target_pos_w.torch[:, 0, :] - o
    dz = (red[:, 2] - blue[:, 2]) * 100
    dxy = torch.linalg.norm(red[:, :2] - blue[:, :2], dim=1) * 100
    ee_red = torch.linalg.norm(ee - red, dim=1) * 100
    lf, rf = _in_contact(env, "left_finger_red_contact", THR), _in_contact(env, "right_finger_red_contact", THR)
    touching = _in_contact(env, "red_blue_contact", THR)
    grasping = lf & rf
    stacked = (red[:, 2] > LIFT_Z) & touching & ~grasping
    seated = ((dz - CUBE_H * 100).abs() < 0.6) & (dxy < 2.0)
    print(f"{tag:22s} red-blue dz med={dz.median():.2f}cm dxy med={dxy.median():.2f}cm | ee-red med={ee_red.median():.2f}cm | "
          f"both fingers on red={grasping.float().mean():.2f} red-blue contact={touching.float().mean():.2f} "
          f"robosuite stacked={stacked.float().mean():.2f} red seated on blue={seated.float().mean():.2f}", flush=True)


act = torch.zeros(env.num_envs, env.action_space.shape[1], device=env.device)
act[:, -1] = -1.0  # keep the gripper closed
with torch.inference_mode():
    env.reset()
    report("right after reset")
    for k in range(1, args_cli.hold_steps + 1):
        env.step(act)
        if k in (1, 5, args_cli.hold_steps):
            report(f"hold {k} steps")
    if not args_cli.stage2:
        act[:, -1] = 1.0  # open: red should settle onto blue
        for k in range(1, args_cli.release_steps + 1):
            env.step(act)
            if k in (5, 15, args_cli.release_steps):
                report(f"released {k} steps")
env.close()
simulation_app.close()
