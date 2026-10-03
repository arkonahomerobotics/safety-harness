"""Record "red cube grasped and carried onto the blue cube" states from the scripted expert (expert_stack.py).

Uses the expert's phase logic unchanged on the camera-free IK-Rel stack task (same robot/cubes/table as the
RL tasks, and the action scale the expert was written for). Records states while it holds the red cube in
the "move" (aligned hover) and "lower" (descending onto blue) phases of the first stacking step.
Saves {"joint_pos": (M, num_joints), "cube_pose": (M, 3, 7), "phase": (M,)} with cube positions
relative to the env origin; phase 0 = move, 1 = lower.

Isaac Lab 3.x port: default task id updated, --seed added, cube quaternions are xyzw (root_link_pose_w), and the
height-gap report uses 4.68 cm. The phase machine and recording rule are unchanged.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
# Isaac Lab 3.x id of the old camera-free "Isaac-Stack-Cube-Franka-IK-Rel-v0" (same scene, IK-Rel scale 0.5)
parser.add_argument("--task", type=str, default="IsaacContrib-Stack-Cube-Franka-IK-Rel")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--max_snapshots", type=int, default=60000)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import os

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

CUBE_H = 0.0468
HOVER = 0.10
KP = 1.0
AMAX = 0.05
OPEN, CLOSE = 1.0, -1.0


class Expert:
    """expert_stack.py's first stacking step (pick cube_2 -> place on cube_1), unchanged."""

    def __init__(self):
        self.phase = "above"
        self.phase_steps = 0
        self.done = False

    def set_phase(self, phase):
        self.phase = phase
        self.phase_steps = 0

    def act(self, eef, obj, dest, finger):
        self.phase_steps += 1
        if self.done:
            return self._to(eef, eef, OPEN)
        grip = OPEN
        target = eef.copy()
        xy_err_obj = np.linalg.norm((obj - eef)[:2])
        holding = np.linalg.norm(obj - eef) < 0.03 and finger < 0.035
        if self.phase == "above":
            target = obj + np.array([0, 0, HOVER])
            if xy_err_obj < 0.006 and abs(target[2] - eef[2]) < 0.015:
                self.set_phase("descend")
        elif self.phase == "descend":
            target = obj.copy()
            if xy_err_obj > 0.02:
                self.set_phase("above")
            elif np.linalg.norm(target - eef) < 0.008 or self.phase_steps > 60:
                self.set_phase("grasp")
            else:
                return self._to(eef, target, grip, z_max=0.02)
        elif self.phase == "grasp":
            target = obj.copy()
            grip = CLOSE
            if self.phase_steps > 12:
                self.set_phase("lift")
        elif self.phase == "lift":
            grip = CLOSE
            target = np.array([eef[0], eef[1], dest[2] + CUBE_H + HOVER])
            if not holding and self.phase_steps > 5:
                self.set_phase("above")
            elif abs(target[2] - eef[2]) < 0.015:
                self.set_phase("move")
        elif self.phase == "move":
            grip = CLOSE
            target = dest + np.array([0, 0, CUBE_H + HOVER])
            if not holding:
                self.set_phase("above")
            elif np.linalg.norm((target - eef)[:2]) < 0.006 and abs(target[2] - eef[2]) < 0.015:
                self.set_phase("lower")
        elif self.phase == "lower":
            grip = CLOSE
            target = dest + np.array([0, 0, CUBE_H + 0.004])
            if not holding:
                self.set_phase("above")
            elif np.linalg.norm((dest - eef)[:2]) > 0.02:
                self.set_phase("move")
            elif np.linalg.norm(target - eef) < 0.006 or self.phase_steps > 60:
                self.set_phase("release")
            else:
                return self._to(eef, target, grip, z_max=0.02)
        elif self.phase == "release":
            self.done = True
            return self._to(eef, eef, OPEN)
        return self._to(eef, target, grip)

    def holding(self, eef, obj, finger):
        return np.linalg.norm(obj - eef) < 0.03 and finger < 0.035

    @staticmethod
    def _to(eef, target, grip, z_max=AMAX):
        pos = np.clip(KP * (target - eef), -AMAX, AMAX)
        pos[2] = np.clip(pos[2], -z_max, z_max)
        return np.array([*pos, 0.0, 0.0, 0.0, grip], dtype=np.float32)


env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
n = env.num_envs
robot = env.scene["robot"]
finger_ids, _ = robot.find_joints("panda_finger_joint.*")
experts = [Expert() for _ in range(n)]
joint_list, cube_list, phase_list = [], [], []

with torch.inference_mode():
    env.reset()
    for step in range(args_cli.steps):
        origins = env.scene.env_origins
        eef = (env.scene["ee_frame"].data.target_pos_w.torch[:, 0, :] - origins).cpu().numpy().astype(np.float64)
        red = (env.scene["cube_2"].data.root_pos_w.torch - origins).cpu().numpy().astype(np.float64)
        blue = (env.scene["cube_1"].data.root_pos_w.torch - origins).cpu().numpy().astype(np.float64)
        finger = robot.data.joint_pos.torch[:, finger_ids].mean(dim=1).cpu().numpy()

        record = [i for i in range(n) if experts[i].phase in ("move", "lower") and experts[i].holding(eef[i], red[i], finger[i])]
        if record:
            ids = torch.tensor(record, device=env.device)
            joint_list.append(robot.data.joint_pos.torch[ids].cpu())
            poses = []
            for name in ("cube_1", "cube_2", "cube_3"):
                p = env.scene[name].data.root_link_pose_w.torch[ids].clone()
                p[:, :3] -= origins[ids]
                poses.append(p.cpu())
            cube_list.append(torch.stack(poses, dim=1))
            phase_list.append(torch.tensor([0 if experts[i].phase == "move" else 1 for i in record]))

        actions = np.stack([experts[i].act(eef[i], red[i], blue[i], float(finger[i])) for i in range(n)])
        _, _, terminated, truncated, _ = env.step(torch.tensor(actions, device=env.device))
        for i in torch.nonzero(terminated | truncated).flatten().tolist():
            experts[i] = Expert()

joint_pos, cube_pose, phase = torch.cat(joint_list), torch.cat(cube_list), torch.cat(phase_list)
if len(joint_pos) > args_cli.max_snapshots:
    sel = torch.randperm(len(joint_pos))[: args_cli.max_snapshots]
    joint_pos, cube_pose, phase = joint_pos[sel], cube_pose[sel], phase[sel]
os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
torch.save({"joint_pos": joint_pos, "cube_pose": cube_pose, "phase": phase}, args_cli.out)
# reporting only: seated red center = blue center + 4.68 cm (the Brev script still had the stale 4.06 here)
gap = (cube_pose[:, 1, 2] - cube_pose[:, 0, 2] - CUBE_H) * 100
xy = torch.linalg.norm(cube_pose[:, 1, :2] - cube_pose[:, 0, :2], dim=1) * 100
print(f"SNAPSHOTS saved={len(joint_pos)} (move={int((phase == 0).sum())}, lower={int((phase == 1).sum())}) -> {args_cli.out}")
print(f"SNAPSHOTS height gap above seated (cm): median={gap.median():.2f} p10={gap.quantile(0.1):.2f} p90={gap.quantile(0.9):.2f}")
print(f"SNAPSHOTS sideways offset (cm): median={xy.median():.2f} p90={xy.quantile(0.9):.2f}")
env.close()
simulation_app.close()
