# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distance-adaptive IK-Rel arm action config for the privileged-state RL variant of the Franka
cube-stack task.

Even after cutting the base ``DifferentialInverseKinematicsActionCfg``'s ``scale`` from 0.5 to
0.1 (see ``stack_ik_rel_rl_env_cfg.py``'s Phase 2a docstring), ``align_best`` plateaued a second
time -- now around 0.80-0.81 instead of 1.0 -- while reach/grasp/lift stayed perfect. A single
fixed scale is a compromise between two conflicting needs: large enough to cover the ~30-50cm
initial approach in a reasonable number of steps, small enough to hold a sub-centimeter final
position without a single noisy action sample kicking the held cube back out of tolerance. No
fixed value serves both well. This action term resolves that by making ``scale`` a function of
the current guidance distance -- the same distance the reward's ``reaching``/``align`` terms are
already computed from (end-effector-to-target-cube while not yet holding it, held-cube-to-
destination once lifted) -- so steps are coarse far from the target and shrink smoothly to
``min_scale`` as it closes in, rather than using one compromise value everywhere.

This module must stay importable before the Kit app exists (``train.py`` resolves env configs
via Hydra before ``launch_simulation()``), so it only imports config-level modules; the
implementation lives in ``rl_actions_impl.py`` and is referenced lazily via ``class_type``.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass


@configclass
class DistanceScaledIKRelActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for :class:`~.rl_actions_impl.DistanceScaledIKRelAction`."""

    class_type: type | str = "{DIR}.rl_actions_impl:DistanceScaledIKRelAction"

    max_scale: float = 0.1
    """Scale used at or beyond ``distance_cap`` -- matches the fixed scale that already proved
    sufficient for reach/grasp/lift, so the far-field behavior is unchanged."""
    distance_cap: float = 0.15
    """Guidance distance (meters) beyond which scale is clamped to ``max_scale``."""

    # Separate ramps for the two phases: a quadratic ramp everywhere made the final placement finer
    # but also shrank steps around the grasp, which had been working, and grasping degraded.
    reach_min_scale: float = 0.02
    """Scale at the target cube while reaching/grasping (not yet lifted)."""
    reach_ramp_exponent: float = 1.0
    """Ramp shape while reaching/grasping: 1 is linear."""
    place_min_scale: float = 0.005
    """Scale at the destination while placing (cube lifted)."""
    place_ramp_exponent: float = 2.0
    """Ramp shape while placing: 2 (quadratic) shrinks steps much faster within the last few cm."""
    minimal_lift_height: float = 0.03
    height_diff: float = 0.0468
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2")
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3")
