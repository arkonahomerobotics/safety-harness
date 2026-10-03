# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events specific to the G1 two-block stacking task.

Reconstructed 2026-10-03: the reverse-curriculum env cfg imported this from
``isaaclab_tasks.manager_based.manipulation.stack.mdp``, a path that does not exist in this
IsaacLab checkout -- another Brev-only file, never committed (same failure mode as
``overlay.py``). The snapshot FILE FORMAT below is read directly off
``scripts/stack_expert_rl.py``'s ``--snapshots`` writer, not guessed: it saves
``{"joint_pos": robot.data.joint_pos.torch[kept], "cube_pose": stacked_pos_quat[kept]}``, where
``cube_pose`` is shaped ``(S, 2, 7)`` -- one ``[x, y, z, qw/qx, qy, qz, qw]``-style pose per cube
(whatever quaternion order ``root_quat_w`` natively returns, reused unchanged, never reordered),
index 0 for the first ``cube_names`` entry and index 1 for the second -- and both block positions
are stored **relative to each env's own origin** (``blk.data.root_pos_w - origins`` in the writer),
so restoring them has to add ``env.scene.env_origins`` back before writing into the sim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Snapshot files are small (one stack_expert_rl.py run's worth of kept episodes) and reused across
# every reset call for the lifetime of the process -- cache by path+device instead of re-reading
# the file from disk on every single reset.
_SNAPSHOT_CACHE: dict[tuple[str, str], dict[str, torch.Tensor]] = {}


def _load_snapshots(snapshot_path: str, device: str) -> dict[str, torch.Tensor]:
    key = (snapshot_path, str(device))
    cached = _SNAPSHOT_CACHE.get(key)
    if cached is None:
        data = torch.load(snapshot_path, map_location=device, weights_only=True)
        cached = {"joint_pos": data["joint_pos"].to(device), "cube_pose": data["cube_pose"].to(device)}
        _SNAPSHOT_CACHE[key] = cached
    return cached


def reset_from_snapshots(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    snapshot_path: str,
    prob: float,
    cube_names: tuple[str, str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """With probability ``prob`` per env being reset, overwrite that env's robot joint state and
    both cubes' poses from a randomly-chosen saved snapshot (block already held and aligned above
    the destination block) instead of leaving whatever a normal reset already produced.

    Defined last in the event list on purpose (see the env cfg) so it runs after -- and overrides,
    for only the envs it selects -- the ordinary block-position randomization.
    """
    robot = env.scene[robot_cfg.name]
    device = robot.device
    snaps = _load_snapshots(snapshot_path, device)
    n_snap = snaps["joint_pos"].shape[0]

    # env_ids may be slice(None) for a full reset -- len()/indexing-for-randint needs real ids.
    ids = torch.arange(env.num_envs, device=device) if isinstance(env_ids, slice) else env_ids
    if ids.numel() == 0 or n_snap == 0:
        return

    use_snapshot = torch.rand(ids.numel(), device=device) < prob
    sel_ids = ids[use_snapshot]
    if sel_ids.numel() == 0:
        return

    snap_idx = torch.randint(0, n_snap, (sel_ids.numel(),), device=device)

    # robot: full joint state (all DOFs -- the writer saved the whole `robot.data.joint_pos`, not
    # just the arm), zero velocity (the captured moment is a held, settling grasp, not mid-motion).
    joint_pos = snaps["joint_pos"][snap_idx]
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_position_to_sim_index(position=joint_pos, joint_ids=slice(None), env_ids=sel_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=slice(None), env_ids=sel_ids)

    # both cubes: stored position is relative to the env's own origin -- add it back for the sim.
    origins = env.scene.env_origins[sel_ids]
    zero_vel = torch.zeros(sel_ids.numel(), 6, device=device)
    for cube_idx, cube_name in enumerate(cube_names):
        cube = env.scene[cube_name]
        pose = snaps["cube_pose"][snap_idx, cube_idx].clone()
        pose[:, 0:3] = pose[:, 0:3] + origins
        cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=sel_ids)
        cube.write_root_velocity_to_sim_index(root_velocity=zero_vel, env_ids=sel_ids)


def reset_stage2_mixed_snapshots(
    env,
    env_ids,
    handoff_snapshot_path: str,
    expert_snapshot_path: str,
    handoff_prob: float,
    expert_prob: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Stage-2 reverse curriculum: each env gets EXACTLY ONE of three mutually exclusive starts --
    a real Stage-1 handoff state (A/B + robot joints overwritten, C keeps its own just-randomized
    table position), an expert mid-carry-of-C state (C + robot joints overwritten, A/B keep their
    normal pre-seated-on-B reset), or the plain default reset (no override at all). Two independent
    reset_from_snapshots calls at the same probs would instead let ~(handoff_prob * expert_prob) of
    envs get BOTH, with the second write clobbering the first snapshot's robot joints -- leaving
    the hand in one snapshot's (e.g. resting) pose while a block from the OTHER snapshot sits
    wherever it was recorded (e.g. an expert mid-carry block floating mid-air with nothing holding
    it). Drawing one bucket per env instead guarantees that never happens.

    remainder = 1 - handoff_prob - expert_prob is the default-reset fraction, kept intentionally
    nonzero -- same reasoning as Stage 1's own prob=0.5 reverse curriculum leaving half the envs
    on the plain reset rather than always starting from a snapshot.
    """
    robot = env.scene[robot_cfg.name]
    device = robot.device
    ids = torch.arange(env.num_envs, device=device) if isinstance(env_ids, slice) else env_ids
    if ids.numel() == 0:
        return

    draw = torch.rand(ids.numel(), device=device)
    handoff_ids = ids[draw < handoff_prob]
    expert_ids = ids[(draw >= handoff_prob) & (draw < handoff_prob + expert_prob)]
    if handoff_ids.numel() > 0:
        reset_from_snapshots(env, handoff_ids, handoff_snapshot_path, 1.0, ("object", "block_b"), robot_cfg)
    if expert_ids.numel() > 0:
        reset_from_snapshots(env, expert_ids, expert_snapshot_path, 1.0, ("block_c",), robot_cfg)

