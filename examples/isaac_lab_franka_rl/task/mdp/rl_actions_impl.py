# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of the distance-adaptive IK-Rel arm action -- see ``rl_actions.py`` for the
config and design rationale.

Kept separate from the config on purpose: this module imports ``task_space_actions``, which
loads USD (``pxr``). Config modules are resolved by ``train.py`` (via Hydra) *before* the Kit app
is constructed, and loading ``pxr`` that early makes Kit initialize USD a second time and crash
natively (``free(): invalid pointer`` in ``TfRegistryManager``). The config references this class
through a lazy ``{DIR}`` string instead, so it's only imported once the app is up -- the same
split Isaac Lab uses for its own ``actions_cfg.py`` / ``task_space_actions.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

from . import observations as stack_obs

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import FrameTransformer

    from .rl_actions import DistanceScaledIKRelActionCfg


class DistanceScaledIKRelAction(DifferentialInverseKinematicsAction):
    """Same as ``DifferentialInverseKinematicsAction``, but ``scale`` shrinks from ``cfg.max_scale``
    (at or beyond ``cfg.distance_cap``) toward a per-phase minimum as the target nears: the reach
    ramp (EE to target cube) until the cube is lifted, then the place ramp (held cube to its
    destination). Recomputed every step.
    """

    cfg: DistanceScaledIKRelActionCfg

    def process_actions(self, actions: torch.Tensor):
        env: ManagerBasedEnv = self._env
        cube_1: RigidObject = env.scene[self.cfg.cube_1_cfg.name]
        cube_2: RigidObject = env.scene[self.cfg.cube_2_cfg.name]
        cube_3: RigidObject = env.scene[self.cfg.cube_3_cfg.name]
        ee_frame: FrameTransformer = env.scene[self.cfg.ee_frame_cfg.name]
        ee_pos = ee_frame.data.target_pos_w.torch[:, 0, :]

        stacked_1 = stack_obs.object_stacked(env, self.cfg.robot_cfg, self.cfg.cube_2_cfg, self.cfg.cube_1_cfg)
        grasped_1 = stack_obs.object_grasped(env, self.cfg.robot_cfg, self.cfg.ee_frame_cfg, self.cfg.cube_2_cfg)
        grasped_2 = stack_obs.object_grasped(env, self.cfg.robot_cfg, self.cfg.ee_frame_cfg, self.cfg.cube_3_cfg)
        stage1_done = stacked_1 & ~grasped_1

        cube_1_pos = cube_1.data.root_pos_w.torch
        cube_2_pos = cube_2.data.root_pos_w.torch
        cube_3_pos = cube_3.data.root_pos_w.torch

        offset = torch.zeros_like(cube_1_pos)
        offset[:, 2] = self.cfg.height_diff

        # -- same "current target" logic as the reward's staged_achievements, so the action term
        # and the reward always agree on what's being approached --
        grasped_cur = torch.where(stage1_done, grasped_2, grasped_1)
        held_pos_cur = torch.where(stage1_done.unsqueeze(1), cube_3_pos, cube_2_pos)
        lifted_cur = grasped_cur & (held_pos_cur[:, 2] > self.cfg.minimal_lift_height)

        dist_to_2 = torch.linalg.norm(cube_2_pos - ee_pos, dim=1)
        dist_to_3 = torch.linalg.norm(cube_3_pos - ee_pos, dim=1)
        dist_reach_cur = torch.where(stage1_done, dist_to_3, dist_to_2)

        dist_dest_1 = torch.linalg.norm(cube_2_pos - (cube_1_pos + offset), dim=1)
        dist_dest_2 = torch.linalg.norm(cube_3_pos - (cube_2_pos + offset), dim=1)
        dist_align_cur = torch.where(stage1_done, dist_dest_2, dist_dest_1)

        reach_frac = (dist_reach_cur / self.cfg.distance_cap).clamp(0.0, 1.0) ** self.cfg.reach_ramp_exponent
        reach_scale = self.cfg.reach_min_scale + (self.cfg.max_scale - self.cfg.reach_min_scale) * reach_frac
        place_frac = (dist_align_cur / self.cfg.distance_cap).clamp(0.0, 1.0) ** self.cfg.place_ramp_exponent
        place_scale = self.cfg.place_min_scale + (self.cfg.max_scale - self.cfg.place_min_scale) * place_frac

        dynamic_scale = torch.where(lifted_cur, place_scale, reach_scale)
        self._scale[:] = dynamic_scale.unsqueeze(1)

        super().process_actions(actions)
