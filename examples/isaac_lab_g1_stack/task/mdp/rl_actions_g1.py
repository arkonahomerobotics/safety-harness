"""Action configs for the GPU-fast RL variant of the G1 two-block stacking task.

The stock task drives the upper body through Pink IK, which solves on the CPU once per env per step
(~85 env-steps/s measured on this box) -- far too slow for PPO. The RL variant instead drives only
the left arm with Isaac Lab's batched differential IK (relative pose deltas, as the Franka RL
variant does) and the left Dex3 hand with a single scalar grip command.

Config-only on purpose: train.py resolves env configs before the Kit app exists, so the
implementation lives in ``rl_actions_g1_impl.py`` behind a lazy ``{DIR}`` class_type (same split as
the Franka ``rl_actions.py`` / ``rl_actions_impl.py``).
"""

from __future__ import annotations

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

LEFT_HAND_CURL_JOINTS = [
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
]


@configclass
class G1GripActionCfg(ActionTermCfg):
    """One scalar in [-1, 1] -> left Dex3 curl joints between open (-1) and ``close_frac`` of the way
    to fully curled (+1); thumb yaw held at ``thumb_yaw``. Values from grid_pocket.py's most robust
    real-grip region (close 0.8, thumb yaw -0.4)."""

    class_type: type | str = "{DIR}.rl_actions_g1_impl:G1GripAction"
    asset_name: str = "robot"
    curl_joint_names: list[str] = LEFT_HAND_CURL_JOINTS
    thumb_yaw_joint_name: str = "left_hand_thumb_0_joint"
    thumb_yaw: float = -0.4
    close_frac: float = 0.8
