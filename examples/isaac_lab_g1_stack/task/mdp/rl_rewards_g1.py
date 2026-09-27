"""Reward terms for the privileged-state RL variant of the G1 single-arm block pick-and-place task.

Ports the design the Franka cube-stack RL work on this box already discovered the hard way (see
that task's own stack_ik_rel_rl_env_cfg.py module docstring for the full history) rather than
re-discovering it: a pure one-time-achievement reward stalled after the capped `reaching` budget
was spent, and a misaligned-release penalty actively taught the policy to avoid grasping at all,
since an early lucky grasp followed by an (inevitable, before placement is learned) bad release
scored net negative. The fix there was a two-phase curriculum -- Phase 1 rewards only
grasp/lift/reaching (align/release/stack pay nothing, so there is nothing to build a
"grasping is bad" association from), then Phase 2 resumes that checkpoint and adds
align/release/stack with the misaligned-release penalty removed entirely. This file's
`staged_achievements` is built to support that same phase split for G1 from the start via the
`*_reward` params being independently zeroable per phase in the env cfg, rather than needing to
discover the stall firsthand.

The other real adaptation is grasp detection itself. Franka's own `object_grasped` checks two
gripper-finger joints against a fixed open value. G1 has no such simple 2-DOF signal -- grasp
detection here is a distance-plus-finger-curl proxy: wrist-to-object distance under a threshold AND
the left-hand curl joints closed past a threshold fraction of their real, measured range. That
proxy alone is NOT evidence of a grip (an earlier scripted-expert version reported "holding" while
the object sat still on the table -- see grid_real_grasp.py), so `lifted` additionally requires the
object's real height to rise, which only happens if contact physics is actually carrying it. This
reward never applies a kinematic attach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

# Curl joints only -- thumb_0 is a symmetric yaw joint with no single "closed" direction, so it's
# excluded from the closure fraction rather than guessed at.
LEFT_HAND_JOINT_NAMES = (
    "left_hand_index_0_joint", "left_hand_middle_0_joint",
    "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
)


def _tanh_dist(distance: torch.Tensor, std: float) -> torch.Tensor:
    return 1.0 - torch.tanh(distance / std)


def _left_wrist_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, wrist_link_name: str) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    idx = robot.data.body_names.index(wrist_link_name)
    return robot.data.body_pos_w.torch[:, idx] - env.scene.env_origins


def _finger_closed_frac(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, joint_ids: list[int], open_vals: torch.Tensor, closed_vals: torch.Tensor
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    pos = robot.data.joint_pos.torch[:, joint_ids]
    frac = (pos - open_vals) / (closed_vals - open_vals + 1e-9)
    return frac.mean(dim=1)


class staged_achievements(ManagerTermBase):
    """One-time achievement rewards (grasp/lift/align) plus an every-time release-outcome
    reward/penalty, for a single object and a single fixed place target -- the G1 single-arm
    analogue of the Franka task's own reward of the same name. See module docstring for the
    grasp-detection adaptation and the phase-curriculum rationale (both ported from that task's
    real training history, not re-derived here)."""

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

        robot: Articulation = env.scene["robot"]
        joint_ids, _ = robot.find_joints(list(LEFT_HAND_JOINT_NAMES))
        self._joint_ids = joint_ids
        limits = robot.data.soft_joint_pos_limits.torch[0, joint_ids]
        lo, hi = limits[:, 0], limits[:, 1]
        # Open = the rest pose (every hand joint reads 0.0 at reset) clamped into the soft limits;
        # closed = the soft limit farthest from zero. An earlier version used open=lo/closed=hi
        # uniformly, which is inverted for the left index/middle joints (soft limits ~[-1.49,
        # -0.08]: 0 is open, -1.49 is curled) -- found by grid_real_grasp.py's diagnosis.
        self._open_vals = torch.clamp(torch.zeros_like(lo), lo, hi)
        self._closed_vals = torch.where(lo.abs() > hi.abs(), lo, hi)

    def reset(self, env_ids: torch.Tensor):
        if len(env_ids) > 0:
            log = self._env.extras.setdefault("log", {})
            log["Achievements/reach_budget_used"] = self._reach_budget_used[env_ids].mean().item()
            log["Achievements/grasped_rate"] = self._grasped_done[env_ids].float().mean().item()
            log["Achievements/lifted_rate"] = self._lifted_done[env_ids].float().mean().item()
            log["Achievements/align_best"] = self._align_best[env_ids].mean().item()
            log["Achievements/success_rate"] = self._success_done[env_ids].float().mean().item()
            log["Achievements/release_aligned_count"] = self._release_aligned_count[env_ids].mean().item()
            log["Achievements/release_misaligned_count"] = self._release_misaligned_count[env_ids].mean().item()

        self._grasped_done[env_ids] = False
        self._lifted_done[env_ids] = False
        self._reach_budget_used[env_ids] = 0.0
        self._align_best[env_ids] = 0.0
        self._success_done[env_ids] = False
        self._was_holding[env_ids] = False
        self._release_aligned_count[env_ids] = 0.0
        self._release_misaligned_count[env_ids] = 0.0
        self._success_streak[env_ids] = 0
        # else the success termination (read one step later) would see the PREVIOUS episode's flag
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
        align_xy_threshold: float = 0.06,
        align_height_threshold: float = 0.03,
        wrist_link_name: str = "left_wrist_yaw_link",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        place_asset_cfg: SceneEntityCfg | None = None,
        place_height: float = 0.045,
        stack_xy_tol: float = 0.015,
        stack_z_tol: float = 0.012,
        success_hold_steps: int = 1,
        success_max_speed: float = 0.2,
        hand_clear_dist: float = 0.0,
        hold_reward: float = 0.0,
    ) -> torch.Tensor:
        object_: RigidObject = env.scene[object_cfg.name]
        object_pos = object_.data.root_pos_w.torch - env.scene.env_origins
        wrist_pos = _left_wrist_pos(env, robot_cfg, wrist_link_name)
        if place_asset_cfg is not None:
            # stacking: the target is the live seat on top of another block (it is randomized per reset)
            base: RigidObject = env.scene[place_asset_cfg.name]
            place_target = base.data.root_pos_w.torch - env.scene.env_origins
            place_target = place_target + torch.tensor([0.0, 0.0, place_height], device=env.device)
        else:
            place_target = env.place_target  # set by the env cfg's own init event -- see g1_block_rl_env_cfg.py

        dist_to_obj = torch.linalg.norm(object_pos - wrist_pos, dim=1)
        finger_frac = _finger_closed_frac(env, robot_cfg, self._joint_ids, self._open_vals, self._closed_vals)
        grasped = (dist_to_obj < grasp_diff_threshold) & (finger_frac > finger_closed_threshold)
        # Height ABOVE the block's resting height, not absolute -- Franka's version compares absolute
        # z against 0.03 because its table surface sits at z~0; this G1 table is at z~0.70, so a raw
        # absolute check is always true and would pay the lift reward for merely touching the block.
        rest_z = env.cfg.scene.object.init_state.pos[2]
        lifted = grasped & (object_pos[:, 2] > rest_z + minimal_lift_height)

        dist_dest = torch.linalg.norm(object_pos - place_target, dim=1)
        if place_asset_cfg is not None:
            off = object_pos - place_target
            aligned = (torch.linalg.norm(off[:, :2], dim=1) < stack_xy_tol) & (off[:, 2].abs() < stack_z_tol)
        else:
            aligned = dist_dest < align_xy_threshold  # 3D distance -- no separate surface to be "on top of" here

        reward = torch.zeros(env.num_envs, device=env.device)

        # -- reaching: dense, budget-capped so it can't be farmed by lingering near the object --
        reach_potential = _tanh_dist(dist_to_obj, reach_std)
        reach_remaining = (max_reach_reward - self._reach_budget_used).clamp(min=0.0)
        reach_paid = torch.minimum(reach_potential, reach_remaining)
        reward = reward + reach_paid
        self._reach_budget_used = self._reach_budget_used + reach_paid

        # -- grasp / lift: one-time achievements --
        newly_grasped = grasped & (~self._grasped_done)
        reward = reward + newly_grasped.float() * grasp_reward
        self._grasped_done = self._grasped_done | grasped

        newly_lifted = lifted & (~self._lifted_done)
        reward = reward + newly_lifted.float() * lift_reward
        self._lifted_done = self._lifted_done | lifted

        # -- align: running best-proximity-while-lifted, only grows on genuine improvement --
        align_potential = _tanh_dist(dist_dest, align_std) * max_align_reward * lifted.float()
        align_new_best = torch.maximum(self._align_best, align_potential)
        reward = reward + (align_new_best - self._align_best)
        self._align_best = align_new_best

        # -- release event, evaluated against whatever was held the step before release --
        released = self._was_holding & (~grasped)
        release_reward = torch.where(
            released,
            torch.where(aligned, release_aligned_reward, release_misaligned_penalty),
            torch.zeros_like(reward),
        )
        reward = reward + release_reward
        self._release_aligned_count += (released & aligned).float()
        self._release_misaligned_count += (released & ~aligned).float()
        self._was_holding = grasped

        # -- one-time success: released, aligned, and settled (mirrors the scripted expert's own
        # success check, not this task's built-in bimanual-handoff criterion) --
        # Stacking success must be a real, released placement: seated, at rest, fingers open AND the
        # hand clear of the block, sustained for success_hold_steps. A one-step "seated, fingers
        # below 30% closure, < 0.2 m/s" test was satisfied transiently while the block was still
        # tipping off the red one -- PPO learned to trigger it (85% "success", mostly not stacked).
        settled = torch.linalg.norm(object_.data.root_vel_w.torch[:, :3], dim=1) < success_max_speed
        clear = dist_to_obj > hand_clear_dist
        cond = aligned & settled & (~grasped) & clear & (finger_frac < finger_closed_threshold)
        self._success_streak = torch.where(cond, self._success_streak + 1, torch.zeros_like(self._success_streak))
        success = self._success_streak >= success_hold_steps
        # Per-step reward for every step the finished stack stands with the hand clear. Episodes no
        # longer end at success: ending there meant the policy never saw post-success states, and left
        # running it knocked its own tower down (1/1024 still standing at t=10s).
        reward = reward + hold_reward * cond.float()
        newly_success = success & (~self._success_done)
        reward = reward + newly_success.float() * success_reward
        self._success_done = self._success_done | success

        self._last_success = success
        return reward


def stack_success(env: ManagerBasedRLEnv, term_name: str = "staged_achievements") -> torch.Tensor:
    """Termination: the staged_achievements reward term's own success flag this step (seated,
    settled, released) -- so success is judged by exactly one definition."""
    term = env.reward_manager.get_term_cfg(term_name).func
    return getattr(term, "_last_success", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))


def hand_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize rapid changes in the 14 hand-joint action channels (indices 14:28 of the 28-dim
    action) -- the G1 analogue of the Franka task's own gripper_action_rate_l2, same rationale:
    discourage the kind of open/close jitter an imitation-learned policy tends toward."""
    return torch.sum(
        torch.square(env.action_manager.action[:, 14:28] - env.action_manager.prev_action[:, 14:28]), dim=1
    )


def grip_action_rate_l2(env: ManagerBasedRLEnv, grip_index: int = -1) -> torch.Tensor:
    """Penalize grip-command chatter for the scalar-grip RL action space (last action dim)."""
    return torch.square(env.action_manager.action[:, grip_index] - env.action_manager.prev_action[:, grip_index])
