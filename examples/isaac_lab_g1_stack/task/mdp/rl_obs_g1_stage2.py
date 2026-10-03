"""Privileged-state observation for Stage 2 of the G1 three-block tower task: a fresh policy, a
fresh observation layout (no need to share dims with Stage 1 -- the chained evaluator swaps the
whole actor, not just part of one)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.locomanip_pick_place.mdp.rl_obs_g1 import LEFT_ARM_JOINTS, LEFT_HAND_JOINTS

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def stack_state_stage2(
    env: ManagerBasedRLEnv,
    block_height: float = 0.045,
    wrist_link_name: str = "left_wrist_yaw_link",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    block_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    block_b_cfg: SceneEntityCfg = SceneEntityCfg("block_b"),
    block_c_cfg: SceneEntityCfg = SceneEntityCfg("block_c"),
) -> torch.Tensor:
    """Wrist pose, all three block poses (B position only -- its orientation never matters, it's
    the fixed base of the tower), the two task vectors for C (C - wrist, seat-on-A - C), left arm
    joint pos/vel relative to default, and left hand joint pos -- all env-relative. Mirrors Stage
    1's `stack_state` layout, extended for the third body."""
    robot: Articulation = env.scene[robot_cfg.name]
    a: RigidObject = env.scene[block_a_cfg.name]
    b: RigidObject = env.scene[block_b_cfg.name]
    c: RigidObject = env.scene[block_c_cfg.name]
    o = env.scene.env_origins
    widx = robot.data.body_names.index(wrist_link_name)
    wpos = robot.data.body_pos_w.torch[:, widx] - o
    wquat = robot.data.body_quat_w.torch[:, widx]
    pa = a.data.root_pos_w.torch - o
    pb = b.data.root_pos_w.torch - o
    pc = c.data.root_pos_w.torch - o
    seat_c = pa.clone()
    seat_c[:, 2] += block_height
    if not hasattr(env, "_g1_stack_stage2_obs_ids"):
        arm_ids, _ = robot.find_joints(LEFT_ARM_JOINTS, preserve_order=True)
        hand_ids, _ = robot.find_joints(LEFT_HAND_JOINTS, preserve_order=True)
        env._g1_stack_stage2_obs_ids = (arm_ids, hand_ids)
    arm_ids, hand_ids = env._g1_stack_stage2_obs_ids
    jp = robot.data.joint_pos.torch
    jv = robot.data.joint_vel.torch
    jd = robot.data.default_joint_pos.torch
    return torch.cat(
        [
            wpos,
            wquat,
            pa,
            a.data.root_quat_w.torch,
            pb,
            pc,
            c.data.root_quat_w.torch,
            pc - wpos,
            seat_c - pc,
            jp[:, arm_ids] - jd[:, arm_ids],
            jv[:, arm_ids] * 0.1,
            jp[:, hand_ids],
        ],
        dim=1,
    )
