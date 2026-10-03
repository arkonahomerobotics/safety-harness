# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""robosuite's Stack shaped reward, ported to the Isaac Lab Franka cube-stack task.

Source: robosuite/environments/manipulation/stack.py (``Stack.reward`` / ``Stack.staged_rewards``).
Every step the reward is the max of three stage rewards, so the policy is paid for the highest stage
it currently occupies, and the finished state (red resting on blue, released) pays the most for as
long as it lasts:

    r_reach = 0.25 * (1 - tanh(10 * |ee - red|)), + 0.25 while grasping red        (<= 0.5)
    r_lift  = 1.0 if red is lifted, + 0.5 * (1 - tanh(horizontal |red - blue|))   (<= 1.5)
    r_stack = 2.0 if red is lifted, touching blue, and not grasped                 (= 2.0)
    reward  = max(r_reach, r_lift, r_stack) * reward_scale / 2      (shaped mode)
    reward  = r_stack * reward_scale / 2                              (sparse mode)

``align_mode="seated3d"`` swaps the horizontal align term for 0.5 * (1 - tanh(10 * |red - seated spot|))
(seated spot = blue center + ``seated_height``); see the comment in ``__call__``.

Mapping: robosuite cubeA -> cube_2 (red), cubeB -> cube_1 (blue); cube_3 (green) is ignored.
"Grasping" follows robosuite's ``_check_grasp`` (both finger pads touching the cube) and "touching"
its ``check_contact``, both from filtered contact sensors. robosuite's lift test is the cube center
2 cm above its resting height (table + 0.04 with 2 cm half-size cubes); ``lift_z`` expresses the same
margin for these cubes, which rest with their center 2.03 cm above the table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer

def _as_env_ids(env, env_ids) -> torch.Tensor:
    """Isaac Lab 3.x managers pass ``slice(None)`` (or None) for a full reset; the per-episode logging needs ids."""
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids



def _in_contact(env: ManagerBasedRLEnv, sensor_name: str, threshold: float) -> torch.Tensor:
    # Isaac Lab 3.x: PhysX only reports the normal part of the filtered force matrix. ``force_matrix_w`` is now
    # an alias that returns ``normal_force_matrix_w`` and emits a UserWarning on every access, so read the
    # normal matrix directly (same numbers as the Brev box's PhysX ``force_matrix_w``, without the warning).
    force = env.scene.sensors[sensor_name].data.normal_force_matrix_w.torch.reshape(env.num_envs, -1, 3)
    return (torch.linalg.norm(force, dim=-1) > threshold).any(dim=1)


class robosuite_stack_reward(ManagerTermBase):
    """robosuite Stack staged reward (see module docstring), with per-episode stage logging."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, d = env.num_envs, env.device
        self._steps = torch.zeros(n, device=d)
        # steps in which reach / lift / stack was the paying (max) stage
        self._stage_steps = torch.zeros(n, 3, device=d)
        self._grasped_ever = torch.zeros(n, dtype=torch.bool, device=d)
        self._lifted_ever = torch.zeros(n, dtype=torch.bool, device=d)
        self._stacked_ever = torch.zeros(n, dtype=torch.bool, device=d)
        self._stacked_now = torch.zeros(n, dtype=torch.bool, device=d)

    def reset(self, env_ids: torch.Tensor):
        env_ids = _as_env_ids(self._env, env_ids)
        if len(env_ids) > 0:
            log = self._env.extras.setdefault("log", {})
            steps = self._steps[env_ids].clamp(min=1)
            log["Robosuite/success_at_end"] = self._stacked_now[env_ids].float().mean().item()
            log["Robosuite/success_ever"] = self._stacked_ever[env_ids].float().mean().item()
            log["Robosuite/grasped_ever"] = self._grasped_ever[env_ids].float().mean().item()
            log["Robosuite/lifted_ever"] = self._lifted_ever[env_ids].float().mean().item()
            for i, name in enumerate(("reach", "lift", "stack")):
                log[f"Robosuite/frac_steps_{name}"] = (self._stage_steps[env_ids, i] / steps).mean().item()
        self._steps[env_ids] = 0.0
        self._stage_steps[env_ids] = 0.0
        self._grasped_ever[env_ids] = False
        self._lifted_ever[env_ids] = False
        self._stacked_ever[env_ids] = False
        self._stacked_now[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_z: float = 0.0403,
        contact_threshold: float = 0.01,
        reward_scale: float = 1.0,
        reward_shaping: bool = True,
        align_mode: str = "horizontal",
        seated_height: float = 0.0406,
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        red_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
        blue_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
        red_blue_sensor: str = "red_blue_contact",
        finger_sensors: tuple[str, ...] = ("left_finger_red_contact", "right_finger_red_contact"),
    ) -> torch.Tensor:
        red: RigidObject = env.scene[red_cfg.name]
        blue: RigidObject = env.scene[blue_cfg.name]
        ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
        red_pos = red.data.root_pos_w.torch
        blue_pos = blue.data.root_pos_w.torch
        ee_pos = ee_frame.data.target_pos_w.torch[:, 0, :]

        grasping = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for name in finger_sensors:
            grasping &= _in_contact(env, name, contact_threshold)
        touching = _in_contact(env, red_blue_sensor, contact_threshold)
        lifted = (red_pos[:, 2] - env.scene.env_origins[:, 2]) > lift_z

        r_reach = (1.0 - torch.tanh(10.0 * torch.linalg.norm(ee_pos - red_pos, dim=1))) * 0.25
        r_reach = r_reach + 0.25 * grasping.float()
        if align_mode == "seated3d":
            # 3D distance to where red sits on blue, in robosuite's reach-term form (tanh(10 d)): unlike the
            # horizontal-only term, hovering high earns little, so coming down onto blue before releasing pays
            seated = blue_pos.clone()
            seated[:, 2] += seated_height
            align = 1.0 - torch.tanh(10.0 * torch.linalg.norm(red_pos - seated, dim=1))
        else:
            align = 1.0 - torch.tanh(torch.linalg.norm(red_pos[:, :2] - blue_pos[:, :2], dim=1))
        r_lift = lifted.float() * (1.0 + 0.5 * align)
        stacked = lifted & touching & ~grasping
        r_stack = 2.0 * stacked.float()

        reward, stage = torch.stack((r_reach, r_lift, r_stack), dim=1).max(dim=1)

        self._steps += 1.0
        self._stage_steps.scatter_add_(1, stage.unsqueeze(1), torch.ones_like(reward).unsqueeze(1))
        self._grasped_ever |= grasping
        self._lifted_ever |= lifted
        self._stacked_ever |= stacked
        self._stacked_now = stacked

        if not reward_shaping:
            # robosuite's sparse mode: 2.0 only while stacked
            reward = r_stack
        return reward * reward_scale / 2.0


class full_stack_sparse_reward(ManagerTermBase):
    """Two-level extension of robosuite's sparse Stack reward to the three-cube tower.

    Per step: 2.0 while red (cube_2) is stacked on blue (cube_1), 4.0 while green (cube_3) is also stacked on
    red; times ``reward_scale / 2`` like robosuite. "Stacked" is robosuite's definition for each pair: the upper
    cube lifted, touching the one below, and not grasped by both fingers. The stage-1 value per step matches
    ``robosuite_stack_reward(reward_shaping=False)``, so a stage-1 policy resumes into this unchanged.

    ``stage1_reward=0`` makes it tower-only: with red-on-blue paying on its own, the policy learned to avoid
    risking that safe reward by working with green next to the stack (the same trap the shaped reward's
    holding term caused in stage 1).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, d = env.num_envs, env.device
        self._steps = torch.zeros(n, device=d)
        self._tower_steps = torch.zeros(n, device=d)
        self._stage1_now = torch.zeros(n, dtype=torch.bool, device=d)
        self._tower_now = torch.zeros(n, dtype=torch.bool, device=d)
        self._stage1_ever = torch.zeros(n, dtype=torch.bool, device=d)
        self._tower_ever = torch.zeros(n, dtype=torch.bool, device=d)

    def reset(self, env_ids: torch.Tensor):
        env_ids = _as_env_ids(self._env, env_ids)
        if len(env_ids) > 0:
            log = self._env.extras.setdefault("log", {})
            log["FullStack/red_on_blue_at_end"] = self._stage1_now[env_ids].float().mean().item()
            log["FullStack/tower_at_end"] = self._tower_now[env_ids].float().mean().item()
            log["FullStack/red_on_blue_ever"] = self._stage1_ever[env_ids].float().mean().item()
            log["FullStack/tower_ever"] = self._tower_ever[env_ids].float().mean().item()
            log["FullStack/frac_steps_tower"] = (self._tower_steps[env_ids] / self._steps[env_ids].clamp(min=1)).mean().item()
        self._steps[env_ids] = 0.0
        self._tower_steps[env_ids] = 0.0
        self._stage1_now[env_ids] = False
        self._tower_now[env_ids] = False
        self._stage1_ever[env_ids] = False
        self._tower_ever[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_z: float = 0.0403,
        contact_threshold: float = 0.01,
        reward_scale: float = 1.0,
        red_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
        green_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
        stage1_reward: float = 2.0,
        tower_reward: float = 2.0,
        red_blue_sensor: str = "red_blue_contact",
        green_red_sensor: str = "green_red_contact",
        red_finger_sensors: tuple[str, ...] = ("left_finger_red_contact", "right_finger_red_contact"),
        green_finger_sensors: tuple[str, ...] = ("left_finger_green_contact", "right_finger_green_contact"),
    ) -> torch.Tensor:
        origin_z = env.scene.env_origins[:, 2]
        red_z = env.scene[red_cfg.name].data.root_pos_w.torch[:, 2] - origin_z
        green_z = env.scene[green_cfg.name].data.root_pos_w.torch[:, 2] - origin_z

        grasp_red = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for name in red_finger_sensors:
            grasp_red &= _in_contact(env, name, contact_threshold)
        grasp_green = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for name in green_finger_sensors:
            grasp_green &= _in_contact(env, name, contact_threshold)

        stage1 = (red_z > lift_z) & _in_contact(env, red_blue_sensor, contact_threshold) & ~grasp_red
        tower = stage1 & (green_z > lift_z) & _in_contact(env, green_red_sensor, contact_threshold) & ~grasp_green

        self._steps += 1.0
        self._tower_steps += tower.float()
        self._stage1_now, self._tower_now = stage1, tower
        self._stage1_ever |= stage1
        self._tower_ever |= tower
        return (stage1_reward * stage1.float() + tower_reward * tower.float()) * reward_scale / 2.0


class milestone_stack_reward(ManagerTermBase):
    """One-time milestone bonuses on the way to a stack, plus a per-step completion reward.

    For each stacking level (level 1: red onto blue; level 2: green onto red) five real, discrete sub-goals pay
    ``milestone_values`` exactly once per episode, the first time each is reached:

        reach   the gripper centre is within ``reach_dist`` of the cube
        grasp   both finger pads touch the cube
        lift    grasped and the cube centre is above ``lift_z``
        over    grasped, lifted, and within ``over_xy`` horizontally of the cube below
        placed  robosuite's "stacked": lifted, touching the cube below, not grasped (the release)

    Nothing is paid per step for being in a milestone state, so there is no standing incentive to linger in a
    partial state (the failure of the earlier continuous/shaped rewards). The default values sum to 2.0, one
    step of the completion reward, so finishing and keeping the stack standing always out-earns any partial
    state. Completion pays per step while it stands: 2.0 for the level-1 stack (``mode="stage1"``) or 4.0 for the
    full tower (``mode="stage2"``: level 2 only, from a start with red already on blue; ``mode="full"``: both
    levels' milestones, completion only for the tower). Level-2 milestones count only while level 1 holds.

    Per-episode diagnostics are logged under ``Milestone/`` so gaming can be seen directly: the ever-reached rate
    of each milestone, the conversion rate from each milestone to the next, the fraction of steps spent holding the
    cube lifted, and the rate of episodes that reached "over" but never completed.
    """

    NAMES = ("reach", "grasp", "lift", "over", "placed")

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, d = env.num_envs, env.device
        self._steps = torch.zeros(n, device=d)
        self._held_lifted_steps = torch.zeros(n, device=d)
        self._done = torch.zeros(n, 2, 5, dtype=torch.bool, device=d)
        self._complete_ever = torch.zeros(n, dtype=torch.bool, device=d)
        self._complete_now = torch.zeros(n, dtype=torch.bool, device=d)
        self._phi_prev = torch.zeros(n, device=d)
        self._have_prev = torch.zeros(n, dtype=torch.bool, device=d)

    def reset(self, env_ids: torch.Tensor):
        env_ids = _as_env_ids(self._env, env_ids)
        self._have_prev[env_ids] = False
        if len(env_ids) > 0:
            log = self._env.extras.setdefault("log", {})
            for lvl in (0, 1):
                ever = self._done[env_ids, lvl].float()
                if ever.sum() == 0 and lvl == 1:
                    continue
                for i, name in enumerate(self.NAMES):
                    log[f"Milestone/L{lvl + 1}_{name}_ever"] = ever[:, i].mean().item()
                    if i > 0:
                        log[f"Milestone/L{lvl + 1}_{name}_given_prev"] = (
                            (ever[:, i].sum() / ever[:, i - 1].sum().clamp(min=1.0)).item()
                        )
            log["Milestone/complete_ever"] = self._complete_ever[env_ids].float().mean().item()
            log["Milestone/complete_at_end"] = self._complete_now[env_ids].float().mean().item()
            log["Milestone/held_lifted_frac_steps"] = (
                self._held_lifted_steps[env_ids] / self._steps[env_ids].clamp(min=1)
            ).mean().item()
            stuck = self._done[env_ids, :, 3].any(dim=1) & ~self._complete_ever[env_ids]
            log["Milestone/reached_over_never_completed"] = stuck.float().mean().item()
        self._steps[env_ids] = 0.0
        self._held_lifted_steps[env_ids] = 0.0
        self._done[env_ids] = False
        self._complete_ever[env_ids] = False
        self._complete_now[env_ids] = False

    def _level(self, env, upper, lower, contact_sensor, finger_sensors, lift_z, over_xy, thr, reach_dist, rest_z=0.0203, reach_kernel=0.4, over_kernel=0.2):
        up = env.scene[upper].data.root_pos_w.torch
        ee = env.scene["ee_frame"].data.target_pos_w.torch[:, 0, :]
        reach = torch.linalg.norm(ee - up, dim=1) < reach_dist
        lo = env.scene[lower].data.root_pos_w.torch
        grasp = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for name in finger_sensors:
            grasp &= _in_contact(env, name, thr)
        lifted = (up[:, 2] - env.scene.env_origins[:, 2]) > lift_z
        touching = _in_contact(env, contact_sensor, thr)
        near = torch.linalg.norm(up[:, :2] - lo[:, :2], dim=1) < over_xy
        lift = grasp & lifted
        over = lift & near
        placed = lifted & touching & ~grasp & near
        # bounded potential in [0, 1]: nearness of gripper to cube, grasp, grasped height, grasped nearness to target
        zrel = (up[:, 2] - env.scene.env_origins[:, 2] - rest_z).clamp(0.0, 0.06) / 0.06
        phi = (
            0.2 * (1.0 - torch.tanh(torch.linalg.norm(ee - up, dim=1) / reach_kernel))
            + 0.2 * grasp.float()
            + 0.3 * grasp.float() * zrel
            + 0.3 * lift.float() * (1.0 - torch.tanh(torch.linalg.norm(up[:, :2] - lo[:, :2], dim=1) / over_kernel))
        )
        return torch.stack((reach, grasp, lift, over, placed), dim=1), phi

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        mode: str = "stage1",
        lift_z: float = 0.0403,
        over_xy: float = 0.03,
        reach_dist: float = 0.04,
        contact_threshold: float = 0.01,
        reward_scale: float = 1.0,
        milestone_values: tuple[float, float, float, float, float] = (0.1, 0.2, 0.4, 0.5, 0.8),
        completion_reward: float = 2.0,
        potential_scale: float = 0.0,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        thr = contact_threshold
        l1, phi1 = self._level(
            env, "cube_2", "cube_1", "red_blue_contact", ("left_finger_red_contact", "right_finger_red_contact"),
            lift_z, over_xy, thr, reach_dist,
        )
        vals = torch.tensor(milestone_values, device=env.device)
        reward = torch.zeros(env.num_envs, device=env.device)
        held_lifted = l1[:, 2]

        if mode in ("stage1", "full"):
            new = l1 & ~self._done[:, 0]
            reward += (new.float() * vals).sum(dim=1)
            self._done[:, 0] |= l1
        if mode == "stage1":
            complete = l1[:, 4]
        else:
            l2, phi2 = self._level(
                env, "cube_3", "cube_2", "green_red_contact",
                ("left_finger_green_contact", "right_finger_green_contact"), lift_z, over_xy, thr, reach_dist,
            )
            l2 = l2 & l1[:, 4:5]  # level-2 milestones count only while red stands on blue
            new = l2 & ~self._done[:, 1]
            reward += (new.float() * vals).sum(dim=1)
            self._done[:, 1] |= l2
            complete = l2[:, 4]
            held_lifted = l2[:, 2] if mode == "stage2" else (l1[:, 2] | l2[:, 2])

        if potential_scale > 0.0:
            # Ng/Harada/Russell potential-based shaping, gamma*Phi(s') - Phi(s): telescopes along any trajectory, so it
            # cannot change the optimal policy, and standing still in a high-potential state nets a small negative.
            if mode == "stage1":
                phi = torch.where(l1[:, 4], torch.ones_like(phi1), phi1)
            elif mode == "stage2":
                phi = torch.where(l1[:, 4], torch.where(l2[:, 4], torch.ones_like(phi2), phi2), torch.zeros_like(phi2))
            else:
                phi2_total = torch.where(l2[:, 4], torch.ones_like(phi2), phi2)
                phi = torch.where(l1[:, 4], 1.0 + phi2_total, phi1)
            shaping = gamma * phi - self._phi_prev
            reward += potential_scale * torch.where(self._have_prev, shaping, torch.zeros_like(shaping))
            self._phi_prev = phi
            self._have_prev[:] = True
        reward += completion_reward * (2.0 if mode != "stage1" else 1.0) * complete.float()
        self._steps += 1.0
        self._held_lifted_steps += held_lifted.float()
        self._complete_now = complete
        self._complete_ever |= complete
        return reward * reward_scale / 2.0


def red_knocked_off_blue(env: ManagerBasedRLEnv, max_xy: float = 0.03, min_red_z: float = 0.045) -> torch.Tensor:
    """Termination for the stage-2 skill: red (cube_2) is no longer on blue (cube_1) -- fallen below
    ``min_red_z`` (seated red sits at ~0.067 above the env origin) or shifted more than ``max_xy``
    sideways off blue. The skill's precondition is gone, so the episode can't earn anything more.
    """
    blue = env.scene["cube_1"].data.root_pos_w.torch
    red = env.scene["cube_2"].data.root_pos_w.torch
    red_z = red[:, 2] - env.scene.env_origins[:, 2]
    return (red_z < min_red_z) | (torch.linalg.norm(red[:, :2] - blue[:, :2], dim=1) > max_xy)


def robot_stack_contact(env: ManagerBasedRLEnv, sensor_names: list[str], threshold: float = 0.01) -> torch.Tensor:
    """Undesired-contact indicator (as Isaac Lab's ``undesired_contacts``, but on filtered contact
    sensors): 1.0 while any listed robot-vs-stack sensor reports contact, else 0.0.
    """
    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in sensor_names:
        hit |= _in_contact(env, name, threshold)
    return hit.float()
