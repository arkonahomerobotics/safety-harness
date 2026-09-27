"""RSL-RL PPO config for the G1 block pick-and-place RL task -- hyperparameters ported directly
from the Franka cube-stack task's own StackCubePPORunnerCfg (same box, same algorithm), as a
starting point: nothing about PPO's own hyperparameters is Franka-specific, and that config is
the product of real tuning on a task of comparable shape (privileged-state, sparse-ish
achievement rewards, single-arm manipulation). Expect this to need its own iteration once real
training curves come in, the same way the Franka config did -- not treated as final.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class G1BlockPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "g1_block_pickplace_rl"
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


@configclass
class G1BlockStackPPORunnerCfg(G1BlockPickPlacePPORunnerCfg):
    """Two-block stacking with the GPU diff-IK action space; same Franka-derived hyperparameters."""

    experiment_name = "g1_block_stack_rl"
    max_iterations = 4000
    clip_actions = 1.0


@configclass
class G1BlockStackBCPPORunnerCfg(G1BlockStackPPORunnerCfg):
    """PPO fine-tuning FROM a behavior-cloned actor (bc_train.py writes the starting checkpoint).
    Bigger net for the multi-phase expert; small action std and a gentle learning rate / KL target so
    the randomly initialized critic's early, noisy advantages can't wreck the cloned behavior."""

    experiment_name = "g1_block_stack_bc"
    max_iterations = 2000
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.15),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
