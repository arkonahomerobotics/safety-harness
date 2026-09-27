"""Privileged-state observations for the G1 two-block stacking RL task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

LEFT_ARM_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
LEFT_HAND_JOINTS = [
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
]


def stack_state(
    env: ManagerBasedRLEnv,
    block_height: float = 0.045,
    wrist_link_name: str = "left_wrist_yaw_link",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    block_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    block_b_cfg: SceneEntityCfg = SceneEntityCfg("block_b"),
) -> torch.Tensor:
    """Wrist pose, both block poses, the two task vectors (block A - wrist, stack seat - block A),
    left arm joint pos/vel relative to default, and left hand joint pos -- all env-relative."""
    robot: Articulation = env.scene[robot_cfg.name]
    a: RigidObject = env.scene[block_a_cfg.name]
    b: RigidObject = env.scene[block_b_cfg.name]
    o = env.scene.env_origins
    widx = robot.data.body_names.index(wrist_link_name)
    wpos = robot.data.body_pos_w.torch[:, widx] - o
    wquat = robot.data.body_quat_w.torch[:, widx]
    pa = a.data.root_pos_w.torch - o
    pb = b.data.root_pos_w.torch - o
    seat = pb.clone()
    seat[:, 2] += block_height
    if not hasattr(env, "_g1_stack_obs_ids"):
        arm_ids, _ = robot.find_joints(LEFT_ARM_JOINTS, preserve_order=True)
        hand_ids, _ = robot.find_joints(LEFT_HAND_JOINTS, preserve_order=True)
        env._g1_stack_obs_ids = (arm_ids, hand_ids)
    arm_ids, hand_ids = env._g1_stack_obs_ids
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
            pa - wpos,
            seat - pa,
            jp[:, arm_ids] - jd[:, arm_ids],
            jv[:, arm_ids] * 0.1,
            jp[:, hand_ids],
        ],
        dim=1,
    )
