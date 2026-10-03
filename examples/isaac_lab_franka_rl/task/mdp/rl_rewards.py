# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the privileged-state RL variant of the Franka cube-stack task.

Design history: earlier versions used continuous, per-step dense/shaping rewards (distance-based
"reaching"/"aligning" terms, flat "grasping"/"lifting" bonuses paid every step held). Real training
runs showed these are exploitable by dwelling near whatever pays out -- first "hold the cube
anywhere forever", then (after a partial fix) "hover near the destination forever" -- because a
policy can out-earn a one-time completion bonus just by farming an uncapped per-step reward for
long enough. This version replaces all of that with pure one-time achievements: each milestone
(grasp the target cube, lift it, align it above its destination) pays its point value exactly
once per stage, the first time it's reached; re-achieving it (e.g. after a bad release forces a
retry) pays nothing further. The one thing evaluated every time it happens, not just once, is
the release outcome -- since we always want to reward a well-placed release and penalize a
careless one, however many times it occurs.

The one exception to "one-time" is ``reaching``: a small dense distance-based reward toward the
current target cube, kept purely as exploration scaffolding for finding the first grasp. Unlike
the old dense terms, it's paid out of a fixed per-stage budget (default 0.5 total) rather than
per step forever -- so it can never be farmed by lingering or oscillating in and out of range,
it just tapers off once its budget is spent.

``align`` uses a related but distinct mechanic: a running best-proximity tracker, gated on
``lifted`` (must actually be holding the cube, not just have it airborne from an incidental
collision -- the same bug we found and fixed for ``lifted`` itself). Reward only grows when the
held cube gets closer to its destination than it has ever been this stage, capped at 1.0 exactly
when it reaches the destination; standing still (even very close) or moving away earns nothing
further, and releasing stops accumulation entirely (though whatever was already earned is kept).
This -- not a flat one-time flag -- is what closes the "hover near the destination forever"
exploit properly: a fixed reward the first time you're "close enough" can still be dwelled next
to indefinitely, but a reward that only pays for genuine improvement cannot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as stack_obs
from .terminations import cubes_stacked

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer

def _as_env_ids(env, env_ids) -> torch.Tensor:
    """Isaac Lab 3.x managers pass ``slice(None)`` (or None) for a full reset; the per-episode logging needs ids."""
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids



def _ee_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w.torch[:, 0, :]


def _tanh_dist(distance: torch.Tensor, std: float) -> torch.Tensor:
    return 1.0 - torch.tanh(distance / std)


class staged_achievements(ManagerTermBase):
    """One-time achievement rewards for each stage of the stacking task (cube_2 on cube_1, then
    cube_3 on cube_2), plus an every-time release-outcome reward/penalty.

    Point values (defaults match the agreed design): grasp=+1, lift=+1, release_aligned=+1,
    release_misaligned=-3 (0 in the Phase 2 config -- see env cfg), stacked=+5 (paid once per
    stage, so up to 2x across the full task -- stage-1 completion and the final success). Also
    includes two budget-capped dense terms, ``reaching`` (0.5 total) and ``align`` (1.0 total,
    gated on ``lifted``) -- see module docstring for why these are budgeted rather than flat
    one-time flags or free per-step rewards.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, d = env.num_envs, env.device
        # reaching's spent-budget tracker: one column per stage (0 = cube_2->cube_1, 1 = cube_3->cube_2)
        self._reach_budget_used = torch.zeros(n, 2, dtype=torch.float32, device=d)
        # milestone-achieved flags: one column per stage (0 = cube_2->cube_1, 1 = cube_3->cube_2)
        self._grasped_done = torch.zeros(n, 2, dtype=torch.bool, device=d)
        self._lifted_done = torch.zeros(n, 2, dtype=torch.bool, device=d)
        # align's running best-proximity-while-lifted tracker (0..1), one column per stage
        self._align_best = torch.zeros(n, 2, dtype=torch.float32, device=d)
        self._stack1_done = torch.zeros(n, dtype=torch.bool, device=d)
        self._success_done = torch.zeros(n, dtype=torch.bool, device=d)
        # first aligned release of cube_2 onto cube_1 this episode (one-time placement reward)
        self._placed_done = torch.zeros(n, dtype=torch.bool, device=d)
        # release-event bookkeeping, tracked w.r.t. whichever stage/cube was active last step
        self._was_holding = torch.zeros(n, dtype=torch.bool, device=d)
        self._prev_stage1_done = torch.zeros(n, dtype=torch.bool, device=d)
        # release-outcome counters, purely for logging (not used in the reward itself)
        self._release_aligned_count = torch.zeros(n, dtype=torch.float32, device=d)
        self._release_misaligned_count = torch.zeros(n, dtype=torch.float32, device=d)

    def reset(self, env_ids: torch.Tensor):
        env_ids = _as_env_ids(self._env, env_ids)
        if len(env_ids) > 0:
            log = self._env.extras.setdefault("log", {})
            log["Achievements/reach_budget_used_stage1"] = self._reach_budget_used[env_ids, 0].mean().item()
            log["Achievements/grasped_stage1_rate"] = self._grasped_done[env_ids, 0].float().mean().item()
            log["Achievements/lifted_stage1_rate"] = self._lifted_done[env_ids, 0].float().mean().item()
            log["Achievements/align_best_stage1"] = self._align_best[env_ids, 0].mean().item()
            log["Achievements/stack1_rate"] = self._stack1_done[env_ids].float().mean().item()
            log["Achievements/success_rate"] = self._success_done[env_ids].float().mean().item()
            log["Achievements/placed_rate"] = self._placed_done[env_ids].float().mean().item()
            log["Achievements/release_aligned_count"] = self._release_aligned_count[env_ids].mean().item()
            log["Achievements/release_misaligned_count"] = self._release_misaligned_count[env_ids].mean().item()

        self._reach_budget_used[env_ids] = 0.0
        self._grasped_done[env_ids] = False
        self._lifted_done[env_ids] = False
        self._align_best[env_ids] = 0.0
        self._stack1_done[env_ids] = False
        self._success_done[env_ids] = False
        self._placed_done[env_ids] = False
        self._was_holding[env_ids] = False
        self._prev_stage1_done[env_ids] = False
        self._release_aligned_count[env_ids] = 0.0
        self._release_misaligned_count[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reach_std: float = 0.1,
        max_reach_reward: float = 0.5,
        grasp_reward: float = 1.0,
        lift_reward: float = 1.0,
        align_std: float = 0.1,
        max_align_reward: float = 1.0,
        release_aligned_reward: float = 1.0,
        release_misaligned_penalty: float = -3.0,
        stack_reward: float = 5.0,
        stage2_reward_enabled: bool = True,
        placement_reward: float = 0.0,
        minimal_lift_height: float = 0.03,
        align_xy_threshold: float = 0.05,
        align_height_threshold: float = 0.005,
        height_diff: float = 0.0468,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
        cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
        cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ) -> torch.Tensor:
        cube_1: RigidObject = env.scene[cube_1_cfg.name]
        cube_2: RigidObject = env.scene[cube_2_cfg.name]
        cube_3: RigidObject = env.scene[cube_3_cfg.name]
        ee_pos = _ee_pos(env, ee_frame_cfg)

        stacked_1 = stack_obs.object_stacked(env, robot_cfg, cube_2_cfg, cube_1_cfg)
        grasped_1 = stack_obs.object_grasped(env, robot_cfg, ee_frame_cfg, cube_2_cfg)
        grasped_2 = stack_obs.object_grasped(env, robot_cfg, ee_frame_cfg, cube_3_cfg)
        # stage 1 is done only once cube_2 is in the stacked window *and* let go -- object_stacked
        # is position-only, so without the release check the stage flipped the moment the
        # still-held cube entered the window, freezing stage-1 align at <=0.95 (typically ~0.86)
        stage1_done = stacked_1 & ~grasped_1

        cube_1_pos = cube_1.data.root_pos_w.torch
        cube_2_pos = cube_2.data.root_pos_w.torch
        cube_3_pos = cube_3.data.root_pos_w.torch

        offset = torch.zeros_like(cube_1_pos)
        offset[:, 2] = height_diff

        def _aligned_bool(held_pos: torch.Tensor, dest_pos: torch.Tensor) -> torch.Tensor:
            # purely positional -- used only for the release-outcome check below, which by
            # definition fires the same step the gripper (and so "grasped") just opened, so it
            # must not require currently holding the cube
            diff = held_pos - (dest_pos + offset)
            xy_dist = torch.linalg.norm(diff[:, :2], dim=1)
            height_err = torch.abs(diff[:, 2])
            return (xy_dist < align_xy_threshold) & (height_err < align_height_threshold)

        aligned_1 = _aligned_bool(cube_2_pos, cube_1_pos)
        aligned_2 = _aligned_bool(cube_3_pos, cube_2_pos)

        dist_dest_1 = torch.linalg.norm(cube_2_pos - (cube_1_pos + offset), dim=1)
        dist_dest_2 = torch.linalg.norm(cube_3_pos - (cube_2_pos + offset), dim=1)

        dist_to_2 = torch.linalg.norm(cube_2_pos - ee_pos, dim=1)
        dist_to_3 = torch.linalg.norm(cube_3_pos - ee_pos, dim=1)

        # require actually holding the cube, not just it being airborne -- otherwise an
        # incidental arm/cube collision that knocks it upward pays the lift achievement for
        # free (confirmed from a real run: lifted_stage1_rate was 82% against a 4% grasp rate)
        lifted_1 = (cube_2_pos[:, 2] > minimal_lift_height) & grasped_1
        lifted_2 = (cube_3_pos[:, 2] > minimal_lift_height) & grasped_2

        # -- current-stage milestone achievements, each paid once (first time true) per stage --
        cur_stage = stage1_done.long()  # 0 while working on cube_2->cube_1, 1 once that's done
        idx = cur_stage.unsqueeze(1)

        grasped_cur = torch.where(stage1_done, grasped_2, grasped_1)
        lifted_cur = torch.where(stage1_done, lifted_2, lifted_1)

        reward = torch.zeros(env.num_envs, device=env.device)
        # with stage-2 rewards disabled (Phase 2a), finishing stage 1 unlocks nothing, so the only
        # way to score is refining stage-1 alignment while still holding the cube
        stage_weight = torch.ones_like(reward) if stage2_reward_enabled else (~stage1_done).float()

        # -- reaching: dense, but paid out of a fixed per-stage budget so it can't be farmed by
        # lingering or oscillating near the cube -- it just tapers off once budget is spent --
        dist_cur = torch.where(stage1_done, dist_to_3, dist_to_2)
        reach_potential = _tanh_dist(dist_cur, reach_std)
        reach_used = self._reach_budget_used.gather(1, idx).squeeze(1)
        reach_remaining = (max_reach_reward - reach_used).clamp(min=0.0)
        reach_paid = torch.minimum(reach_potential, reach_remaining)
        reward = reward + reach_paid * stage_weight
        self._reach_budget_used.scatter_(1, idx, (reach_used + reach_paid).unsqueeze(1))

        for done_buf, cur_flag, points in (
            (self._grasped_done, grasped_cur, grasp_reward),
            (self._lifted_done, lifted_cur, lift_reward),
        ):
            already = done_buf.gather(1, idx).squeeze(1)
            newly = cur_flag & (~already)
            reward = reward + newly.float() * points * stage_weight
            done_buf.scatter_(1, idx, (already | cur_flag).unsqueeze(1))

        # -- align: running best-proximity-while-lifted, only grows on genuine improvement, capped
        # at max_align_reward (reached exactly when the held cube arrives at its destination) --
        align_dist_cur = torch.where(stage1_done, dist_dest_2, dist_dest_1)
        align_potential = _tanh_dist(align_dist_cur, align_std) * max_align_reward * lifted_cur.float()
        align_best = self._align_best.gather(1, idx).squeeze(1)
        align_new_best = torch.maximum(align_best, align_potential)
        reward = reward + (align_new_best - align_best) * stage_weight
        self._align_best.scatter_(1, idx, align_new_best.unsqueeze(1))

        # -- release event: attributed to whichever cube/stage was actually held last step, so a
        # release that also completes stage 1 (switching the current stage this same step) still
        # gets evaluated against the cube that was just released, not the new stage's target --
        holding_prev_target = torch.where(self._prev_stage1_done, grasped_2, grasped_1)
        released = self._was_holding & (~holding_prev_target)
        aligned_prev_target = torch.where(self._prev_stage1_done, aligned_2, aligned_1)

        release_reward = torch.where(
            released,
            torch.where(aligned_prev_target, release_aligned_reward, release_misaligned_penalty),
            torch.zeros_like(reward),
        )
        reward = reward + release_reward
        self._release_aligned_count += (released & aligned_prev_target).float()
        self._release_misaligned_count += (released & ~aligned_prev_target).float()

        # -- one-time placement: first aligned release of cube_2 onto cube_1 this episode. The ratchet
        # align reward pays nothing after the closest approach, so without this there's no reason to
        # finish (lower the last few mm and let go) rather than hover. Not scaled by stage_weight:
        # the release itself usually completes stage 1 in the same step. --
        newly_placed = released & aligned_prev_target & ~self._prev_stage1_done & ~self._placed_done
        reward = reward + newly_placed.float() * placement_reward
        self._placed_done = self._placed_done | newly_placed

        # -- one-time stack completion bonuses --
        success = cubes_stacked(env, robot_cfg, cube_1_cfg, cube_2_cfg, cube_3_cfg)
        newly_stack1 = stage1_done & (~self._stack1_done)
        newly_success = success & (~self._success_done)
        reward = reward + newly_stack1.float() * stack_reward + newly_success.float() * stack_reward * stage_weight
        self._stack1_done = self._stack1_done | stage1_done
        self._success_done = self._success_done | success

        # -- state for next step's release detection --
        self._was_holding = grasped_cur
        self._prev_stage1_done = stage1_done

        return reward


def gripper_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize rapid changes in the gripper action channel (assumed last action dim).

    Directly targets the open/close jitter observed from the imitation-learned GR00T policy --
    a decisive, infrequently-changing gripper command should score better than one that
    flickers open/closed every few steps.
    """
    return torch.square(env.action_manager.action[:, -1] - env.action_manager.prev_action[:, -1])
