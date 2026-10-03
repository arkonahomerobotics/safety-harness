"""Record "green cube carried onto the red-on-blue stack" states from the scripted expert (expert_stack.py).

The expert's two-step phase logic, unchanged, on the camera-free IK-Rel stack task. With --all_phases it records every
phase of the second step before release (above/descend/grasp/lift/move/lower), whenever red is seated on
blue; otherwise only while it holds cube_3 (green) in "move"/"lower". Same file format as capture_expert_snapshots.py.

Isaac Lab 3.x port: default task id updated, --seed added, cube quaternions are xyzw. Logic unchanged.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
# Isaac Lab 3.x id of the old camera-free "Isaac-Stack-Cube-Franka-IK-Rel-v0" (same scene, IK-Rel scale 0.5)
parser.add_argument("--task", type=str, default="IsaacContrib-Stack-Cube-Franka-IK-Rel")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=700)
parser.add_argument("--max_snapshots", type=int, default=60000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--all_phases", action="store_true")
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
    """expert_stack.py's Expert, unchanged: cube_2 -> cube_1, then cube_3 -> cube_2."""

    def __init__(self):
        self.tasks = [("cube_2", "cube_1"), ("cube_3", "cube_2")]
        self.task_idx = 0
        self.phase = "above"
        self.phase_steps = 0
        self.done = False

    def set_phase(self, phase):
        self.phase = phase
        self.phase_steps = 0

    def act(self, eef, cubes, finger):
        self.phase_steps += 1
        if self.done:
            return self._to(eef, eef + np.array([0, 0, 0.0]), OPEN)
        obj_name, dest_name = self.tasks[self.task_idx]
        obj, dest = cubes[obj_name], cubes[dest_name]
        grip = OPEN
        target = eef.copy()
        xy_err_obj = np.linalg.norm((obj - eef)[:2])
        holding = np.linalg.norm(obj - eef) < 0.03 and finger < 0.035

        if self.phase == "above":
            target = obj + np.array([0, 0, HOVER])
            if xy_err_obj < 0.006 and abs(target[2] - eef[2]) < 0.015:
                self.set_phase("descend")
        elif self.phase == "descend":
            target = obj + np.array([0, 0, 0.0])
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
            grip = OPEN
            target = eef.copy()
            if self.phase_steps > 10:
                self.set_phase("retreat")
        elif self.phase == "retreat":
            grip = OPEN
            target = np.array([eef[0], eef[1], dest[2] + CUBE_H + HOVER])
            if abs(target[2] - eef[2]) < 0.02:
                self.task_idx += 1
                if self.task_idx >= len(self.tasks):
                    self.done = True
                else:
                    self.set_phase("above")
        return self._to(eef, target, grip)

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
        cubes = {k: (env.scene[k].data.root_pos_w.torch - origins).cpu().numpy().astype(np.float64) for k in ("cube_1", "cube_2", "cube_3")}
        finger = robot.data.joint_pos.torch[:, finger_ids].mean(dim=1).cpu().numpy()

        if args_cli.all_phases:
            seated = (np.abs(cubes["cube_2"][:, 2] - cubes["cube_1"][:, 2] - CUBE_H) < 0.004) & (
                np.linalg.norm(cubes["cube_2"][:, :2] - cubes["cube_1"][:, :2], axis=1) < 0.01)
            record = [
                i for i in range(n)
                if experts[i].task_idx == 1 and not experts[i].done and seated[i]
                and experts[i].phase in ("above", "descend", "grasp", "lift", "move", "lower")
            ]
        else:
            record = [
                i for i in range(n)
                if experts[i].task_idx == 1 and experts[i].phase in ("move", "lower")
                and np.linalg.norm(cubes["cube_3"][i] - eef[i]) < 0.03 and finger[i] < 0.035
            ]
        if record:
            ids = torch.tensor(record, device=env.device)
            joint_list.append(robot.data.joint_pos.torch[ids].cpu())
            poses = []
            for name in ("cube_1", "cube_2", "cube_3"):
                p = env.scene[name].data.root_link_pose_w.torch[ids].clone()
                p[:, :3] -= origins[ids]
                poses.append(p.cpu())
            cube_list.append(torch.stack(poses, dim=1))
            PH = ["above", "descend", "grasp", "lift", "move", "lower"]
            phase_list.append(torch.tensor([PH.index(experts[i].phase) - 4 if not args_cli.all_phases else PH.index(experts[i].phase) for i in record]))

        actions = np.stack([experts[i].act(eef[i], {k: v[i] for k, v in cubes.items()}, float(finger[i])) for i in range(n)])
        _, _, terminated, truncated, _ = env.step(torch.tensor(actions, device=env.device))
        for i in torch.nonzero(terminated | truncated).flatten().tolist():
            experts[i] = Expert()

joint_pos, cube_pose, phase = torch.cat(joint_list), torch.cat(cube_list), torch.cat(phase_list)
if len(joint_pos) > args_cli.max_snapshots:
    sel = torch.randperm(len(joint_pos))[: args_cli.max_snapshots]
    joint_pos, cube_pose, phase = joint_pos[sel], cube_pose[sel], phase[sel]
os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
torch.save({"joint_pos": joint_pos, "cube_pose": cube_pose, "phase": phase}, args_cli.out)
red_on_blue = torch.linalg.norm(cube_pose[:, 1, :2] - cube_pose[:, 0, :2], dim=1) * 100
gap = (cube_pose[:, 2, 2] - cube_pose[:, 1, 2] - 0.0468) * 100
xy = torch.linalg.norm(cube_pose[:, 2, :2] - cube_pose[:, 1, :2], dim=1) * 100
print(f"SNAPSHOTS saved={len(joint_pos)} phase counts={torch.bincount(phase).tolist()} -> {args_cli.out}")
print(f"SNAPSHOTS red-on-blue sideways offset (cm): median={red_on_blue.median():.2f} p90={red_on_blue.quantile(0.9):.2f}")
print(f"SNAPSHOTS green height gap above seated on red (cm): median={gap.median():.2f} p10={gap.quantile(0.1):.2f}")
print(f"SNAPSHOTS green sideways offset from red (cm): median={xy.median():.2f} p90={xy.quantile(0.9):.2f}")
env.close()
simulation_app.close()
