# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event that starts a fraction of episodes from recorded "red cube held above blue" states.

This is the start-state (reverse-curriculum) technique for hard exploration: under robosuite's Stack
reward the policy settled into holding the red cube over the blue one (~1.5 per step) and almost never
tried releasing (2.0 per step once stacked), because from normal starts a release is rare. Starting
some episodes right at the held-above-blue moment makes release attempts, and their outcomes, common.

Snapshots come from the scripted expert carrying the red cube onto the blue one
(``capture_expert_snapshots.py``): the robot's joint positions and the poses of all three cubes, with
positions relative to the env origin. (Snapshots from the RL policy itself held the cube ~12 cm high,
so releasing there was a drop rather than a placement.)

Isaac Lab 3.x port note: ``cube_pose`` is ``[x, y, z, qx, qy, qz, qw]`` (xyzw, as ``root_link_pose_w`` returns it
now). Snapshot files are written and read by the same Isaac Lab version, so the order is never converted -- but a
snapshot file from the old Brev box would have been wxyz and must not be reused here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv

_SNAPSHOTS: dict[str, dict[str, torch.Tensor]] = {}


def reset_from_snapshots(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    snapshot_path: str,
    prob: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_names: tuple[str, ...] = ("cube_1", "cube_2", "cube_3"),
):
    if env_ids is None or isinstance(env_ids, slice):  # full reset may pass None / slice(None)
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    snaps = _SNAPSHOTS.get(snapshot_path)
    if snaps is None:
        snaps = {k: v.to(env.device) for k, v in torch.load(snapshot_path, map_location="cpu", weights_only=True).items()}
        _SNAPSHOTS[snapshot_path] = snaps

    env_ids = torch.as_tensor(env_ids, device=env.device)
    chosen = env_ids[torch.rand(len(env_ids), device=env.device) < prob]
    if len(chosen) == 0:
        return
    idx = torch.randint(0, snaps["joint_pos"].shape[0], (len(chosen),), device=env.device)

    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = snaps["joint_pos"][idx].clone()
    joint_vel = torch.zeros_like(joint_pos)
    # Isaac Lab 3.x: Articulation.set_joint_*_target_index is deprecated (warns every call) in favour of the
    # actuator collection's target command; same buffers, same effect.
    robot.actuators.target_command.set_position_index(value=joint_pos, env_ids=chosen)
    robot.actuators.target_command.set_velocity_index(value=joint_vel, env_ids=chosen)
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=chosen)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=chosen)

    zero_vel = torch.zeros(len(chosen), 6, device=env.device)
    origins = env.scene.env_origins[chosen]
    for i, name in enumerate(cube_names):
        pose = snaps["cube_pose"][idx, i].clone()
        pose[:, :3] += origins
        cube: RigidObject = env.scene[name]
        cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=chosen)
        cube.write_root_velocity_to_sim_index(root_velocity=zero_vel, env_ids=chosen)
