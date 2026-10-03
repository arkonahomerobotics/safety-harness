"""Reward term for Stage 2 of the G1 three-block tower task: place block C on top of the
already-stacked A-on-B pair.

Deliberately a SIBLING of `rl_rewards_g1.staged_achievements`, not an extension of it: that class
is what Run D/E train against for the 2-block task, and this task needs a second, dynamic level of
"is the thing I'm placing onto actually still correctly placed itself" that the original's
`place_asset_cfg` mechanism (one base, checked against its own static cfg rest height) can't
express -- here the immediate base (A) is only "at rest" while it is itself seated on B's LIVE
pose, not a fixed table height. Reuses the shared helpers and the one-time-achievement /
farmable-release fixes from that module rather than re-deriving them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from isaaclab_tasks.contrib.locomanip_pick_place.mdp.rl_rewards_g1 import (
    LEFT_HAND_JOINT_NAMES,
    _finger_closed_frac,
    _left_wrist_pos,
    _tanh_dist,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


class staged_achievements_3stack(ManagerTermBase):
    """Stage-2 analogue of `staged_achievements`: grasp/lift/align/release/success for block C,
    with success and the hold reward gated on the WHOLE tower being grounded -- A seated on B
    (live, not static, pose) AND B still resting on the table -- so there is no loophole where C
    looks "placed" while the thing underneath it has been knocked loose."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, d = env.num_envs, env.device
        self._grasped_done = torch.zeros(n, dtype=torch.bool, device=d)
        self._lifted_done = torch.zeros(n, dtype=torch.bool, device=d)
        self._reach_budget_used = torch.zeros(n, dtype=torch.float32, device=d)
        self._align_best = torch.zeros(n, dtype=torch.float32, device=d)
        self._success_done = torch.zeros(n, dtype=torch.bool, device=d)
        self._was_holding = torch.zeros(n, dtype=torch.bool, device=d)
        self._release_aligned_count = torch.zeros(n, dtype=torch.float32, device=d)
        self._release_misaligned_count = torch.zeros(n, dtype=torch.float32, device=d)
        self._success_streak = torch.zeros(n, dtype=torch.long, device=d)
        self._last_success = torch.zeros(n, dtype=torch.bool, device=d)
        self._aligned_release_done = torch.zeros(n, dtype=torch.bool, device=d)

        robot: Articulation = env.scene["robot"]
        joint_ids, _ = robot.find_joints(list(LEFT_HAND_JOINT_NAMES))
        self._joint_ids = joint_ids
        limits = robot.data.soft_joint_pos_limits.torch[0, joint_ids]
        lo, hi = limits[:, 0], limits[:, 1]
        self._open_vals = torch.clamp(torch.zeros_like(lo), lo, hi)
        self._closed_vals = torch.where(lo.abs() > hi.abs(), lo, hi)

    def reset(self, env_ids: torch.Tensor):
        if isinstance(env_ids, slice) or len(env_ids) > 0:
            log = self._env.extras.setdefault("log", {})
            log["Achievements3/reach_budget_used"] = self._reach_budget_used[env_ids].mean().item()
            log["Achievements3/grasped_rate"] = self._grasped_done[env_ids].float().mean().item()
            log["Achievements3/lifted_rate"] = self._lifted_done[env_ids].float().mean().item()
            log["Achievements3/align_best"] = self._align_best[env_ids].mean().item()
            log["Achievements3/success_rate"] = self._success_done[env_ids].float().mean().item()
            log["Achievements3/release_aligned_count"] = self._release_aligned_count[env_ids].mean().item()
            log["Achievements3/release_misaligned_count"] = self._release_misaligned_count[env_ids].mean().item()

        self._grasped_done[env_ids] = False
        self._lifted_done[env_ids] = False
        self._reach_budget_used[env_ids] = 0.0
        self._align_best[env_ids] = 0.0
        self._success_done[env_ids] = False
        self._was_holding[env_ids] = False
        self._release_aligned_count[env_ids] = 0.0
        self._release_misaligned_count[env_ids] = 0.0
        self._success_streak[env_ids] = 0
        self._aligned_release_done[env_ids] = False
        self._last_success[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reach_std: float = 0.10,
        max_reach_reward: float = 0.5,
        grasp_reward: float = 1.0,
        lift_reward: float = 1.0,
        align_std: float = 0.10,
        max_align_reward: float = 1.0,
        release_aligned_reward: float = 1.0,
        release_misaligned_penalty: float = 0.0,
        success_reward: float = 5.0,
        minimal_lift_height: float = 0.03,
        grasp_diff_threshold: float = 0.14,
        finger_closed_threshold: float = 0.5,
        wrist_link_name: str = "left_wrist_yaw_link",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        block_c_cfg: SceneEntityCfg = SceneEntityCfg("block_c"),
        block_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        block_b_cfg: SceneEntityCfg = SceneEntityCfg("block_b"),
        place_height: float = 0.045,
        stack_xy_tol: float = 0.025,  # Kaoru-approved 2026-10-03, same rationale as Stage 1 -- see g1_block_stack_rl_env_cfg.py's _STACK
        stack_z_tol: float = 0.012,
        success_hold_steps: int = 15,
        success_max_speed: float = 0.03,
        hand_clear_dist: float = 0.11,
        hold_reward: float = 0.05,
        base_rest_z_tol: float = 0.01,
    ) -> torch.Tensor:
        block_c: RigidObject = env.scene[block_c_cfg.name]
        block_a: RigidObject = env.scene[block_a_cfg.name]
        block_b: RigidObject = env.scene[block_b_cfg.name]
        c_pos = block_c.data.root_pos_w.torch - env.scene.env_origins
        a_pos = block_a.data.root_pos_w.torch - env.scene.env_origins
        b_pos = block_b.data.root_pos_w.torch - env.scene.env_origins
        wrist_pos = _left_wrist_pos(env, robot_cfg, wrist_link_name)

        # level 1: is B itself still resting on the table (same static-rest-height check Stage 1 uses)
        b_rest_z = env.cfg.scene.block_b.init_state.pos[2]
        b_on_table = (b_pos[:, 2] - b_rest_z).abs() < base_rest_z_tol

        # level 2: is A correctly seated on B's LIVE pose -- not a static rest height, since A's
        # only "resting" position in this task is wherever B currently is.
        a_seat = b_pos.clone()
        a_seat[:, 2] += place_height
        a_off = a_pos - a_seat
        a_on_b = (torch.linalg.norm(a_off[:, :2], dim=1) < stack_xy_tol) & (a_off[:, 2].abs() < stack_z_tol)
        a_settled = torch.linalg.norm(block_a.data.root_vel_w.torch[:, :3], dim=1) < success_max_speed
        tower_ok = b_on_table & a_on_b & a_settled  # both levels of the base must hold

        # C's target: directly on top of A's live pose -- only meaningful while the base is sound
        place_target = a_pos.clone()
        place_target[:, 2] += place_height

        dist_to_obj = torch.linalg.norm(c_pos - wrist_pos, dim=1)
        finger_frac = _finger_closed_frac(env, robot_cfg, self._joint_ids, self._open_vals, self._closed_vals)
        grasped = (dist_to_obj < grasp_diff_threshold) & (finger_frac > finger_closed_threshold)
        rest_z = env.cfg.scene.block_c.init_state.pos[2]
        lifted = grasped & (c_pos[:, 2] > rest_z + minimal_lift_height)

        dist_dest = torch.linalg.norm(c_pos - place_target, dim=1)
        off = c_pos - place_target
        aligned = (torch.linalg.norm(off[:, :2], dim=1) < stack_xy_tol) & (off[:, 2].abs() < stack_z_tol) & tower_ok

        reward = torch.zeros(env.num_envs, device=env.device)

        reach_potential = _tanh_dist(dist_to_obj, reach_std)
        reach_remaining = (max_reach_reward - self._reach_budget_used).clamp(min=0.0)
        reach_paid = torch.minimum(reach_potential, reach_remaining)
        reward = reward + reach_paid
        self._reach_budget_used = self._reach_budget_used + reach_paid

        newly_grasped = grasped & (~self._grasped_done)
        reward = reward + newly_grasped.float() * grasp_reward
        self._grasped_done = self._grasped_done | grasped

        newly_lifted = lifted & (~self._lifted_done)
        reward = reward + newly_lifted.float() * lift_reward
        self._lifted_done = self._lifted_done | lifted

        align_potential = _tanh_dist(dist_dest, align_std) * max_align_reward * lifted.float()
        align_new_best = torch.maximum(self._align_best, align_potential)
        reward = reward + (align_new_best - self._align_best)
        self._align_best = align_new_best

        # one-time aligned-release bonus, same farmable-loop fix as Stage 1
        released = self._was_holding & (~grasped)
        aligned_release_first = released & aligned & (~self._aligned_release_done)
        self._aligned_release_done = self._aligned_release_done | (released & aligned)
        release_reward = torch.where(
            aligned_release_first,
            torch.full_like(reward, release_aligned_reward),
            torch.where(released & ~aligned, release_misaligned_penalty, 0.0),
        )
        reward = reward + release_reward
        self._release_aligned_count += (released & aligned).float()
        self._release_misaligned_count += (released & ~aligned).float()
        self._was_holding = grasped

        settled = torch.linalg.norm(block_c.data.root_vel_w.torch[:, :3], dim=1) < success_max_speed
        clear = dist_to_obj > hand_clear_dist
        # full-tower success: C on A, hand clear of C, AND the base (A-on-B-on-table) still sound
        cond = aligned & settled & (~grasped) & clear & (finger_frac < finger_closed_threshold) & tower_ok
        self._success_streak = torch.where(cond, self._success_streak + 1, torch.zeros_like(self._success_streak))
        success = self._success_streak >= success_hold_steps
        reward = reward + hold_reward * cond.float()
        newly_success = success & (~self._success_done)
        reward = reward + newly_success.float() * success_reward
        self._success_done = self._success_done | success

        self._last_success = success
        return reward


def stack_success_3stack(env: ManagerBasedRLEnv, term_name: str = "staged_achievements_3stack") -> torch.Tensor:
    """Termination: the staged_achievements_3stack term's own success flag this step."""
    term = env.reward_manager.get_term_cfg(term_name).func
    return getattr(term, "_last_success", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))


def block_a_off_b(
    env: ManagerBasedRLEnv,
    block_a_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    block_b_cfg: SceneEntityCfg = SceneEntityCfg("block_b"),
    place_height: float = 0.045,
    margin: float = 0.02,
    grace_steps: int = 5,
) -> torch.Tensor:
    """Termination: A has dropped more than `margin` below its expected seat on B's live pose --
    the base of the tower was knocked loose (onto the table, or off to the side with it). Checked
    directly against B's current height (not a static rest value), same reasoning as `tower_ok`
    above: being "on the table" is not itself a failure for A in this task, being OFF B is.

    `grace_steps` skips the check for the first few steps after reset: a zero-action sanity check
    (256 envs, 20 steps) found this firing in 7.1% of expert-mid-carry-snapshot starts even with no
    policy action -- settling physics from the snapshot restore, since that snapshot only stores C
    + robot joints, not A/B, so the robot's captured arm pose can slightly overlap a freshly
    randomized A/B placement. Not the dominant driver of the live ~93% termination rate (0 of
    handoff/default starts tripped), but free to rule out."""
    block_a: RigidObject = env.scene[block_a_cfg.name]
    block_b: RigidObject = env.scene[block_b_cfg.name]
    a_z = block_a.data.root_pos_w.torch[:, 2]
    b_z = block_b.data.root_pos_w.torch[:, 2]
    off = (a_z - b_z) < (place_height - margin)
    return off & (env.episode_length_buf >= grace_steps)
