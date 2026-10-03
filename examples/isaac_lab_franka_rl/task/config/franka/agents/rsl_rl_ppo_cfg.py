# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config shared by every Franka cube-stack RL task (``Isaac-Stack-Cube-Franka-IK-Rel-RL-*``).

Values are the final ones from the Brev box (the cfg G and K7 were trained with): fixed LR 5e-5, desired_kl 0.01
(inert under the fixed schedule), entropy 0.006, [256, 128, 64] ELU actor/critic with observation normalization,
init_std 1.0, 24 steps/env, 5 epochs, 4 mini-batches. Field names map 1:1 onto rsl-rl-lib 5.5.1 (``actor`` /
``critic`` model cfgs and ``distribution_cfg`` already existed on the Brev install).

Port note (rsl-rl-lib 5.5.1): ``obs_groups`` is now set explicitly; it resolves to the same thing rsl_rl's fallback
picked before ("policy" for actor and critic), but keeps extra observation groups (e.g. eval_chain.py's
``policy_s2``) from ever being fed to the networks.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class StackCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "franka_stack_rl_expert"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
