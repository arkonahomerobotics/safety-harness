"""Implementation of the scalar Dex3 grip action -- see ``rl_actions_g1.py`` for the config."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .rl_actions_g1 import G1GripActionCfg


class G1GripAction(ActionTerm):
    """Maps a scalar grip command to the left hand's curl joints (and a fixed thumb yaw).

    Open is the rest pose (every hand joint reads 0.0 at reset) clamped into the soft limits;
    fully curled is the soft limit farthest from zero -- the mapping hand_fk.py/grid_pocket.py
    validated (the naive open=lo/closed=hi is inverted for the index/middle joints)."""

    cfg: G1GripActionCfg
    _asset: Articulation

    def __init__(self, cfg: G1GripActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        curl_ids, _ = self._asset.find_joints(cfg.curl_joint_names, preserve_order=True)
        yaw_ids, _ = self._asset.find_joints([cfg.thumb_yaw_joint_name])
        self._joint_ids = list(curl_ids) + list(yaw_ids)
        lim = self._asset.data.soft_joint_pos_limits.torch[0, curl_ids]
        lo, hi = lim[:, 0], lim[:, 1]
        self._open = torch.clamp(torch.zeros_like(lo), lo, hi)
        full = torch.where(lo.abs() > hi.abs(), lo, hi)
        self._closed = self._open + cfg.close_frac * (full - self._open)
        self._raw_actions = -torch.ones(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._processed_actions[:, : len(curl_ids)] = self._open
        self._processed_actions[:, -1] = cfg.thumb_yaw

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        frac = (actions[:, :1].clamp(-1.0, 1.0) + 1.0) / 2.0
        self._processed_actions[:, :-1] = self._open + frac * (self._closed - self._open)
        self._processed_actions[:, -1] = self.cfg.thumb_yaw

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = -1.0
