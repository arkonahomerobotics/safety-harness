# -- G1 two-block stacking, GPU-fast RL variant (diff-IK left arm + scalar Dex3 grip) --
gym.register(
    id="Isaac-Stack-Blocks-G1-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_block_stack_rl_env_cfg:G1BlockStackRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_g1_block:G1BlockStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Blocks-G1-RL-Phase1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_block_stack_rl_env_cfg:G1BlockStackRLPhase1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_g1_block:G1BlockStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Blocks-G1-RL-RC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_block_stack_rl_env_cfg:G1BlockStackRLReverseCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_g1_block:G1BlockStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

# BC-initialized PPO fine-tuning config, selectable with train.py --agent rsl_rl_bc_cfg_entry_point
for _id in ("Isaac-Stack-Blocks-G1-RL-v0", "Isaac-Stack-Blocks-G1-RL-RC-v0"):
    gym.spec(_id).kwargs["rsl_rl_bc_cfg_entry_point"] = (
        f"{agents.__name__}.rsl_rl_ppo_cfg_g1_block:G1BlockStackBCPPORunnerCfg"
    )

# -- Stage 2: place block C on the already-stacked A-on-B pair (chained after a Stage-1 checkpoint) --
gym.register(
    id="Isaac-Stack-Blocks-G1-RL-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_block_stack_3stage2_rl_env_cfg:G1BlockStack3Stage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_g1_block:G1BlockStackPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Stage 2 + reverse curriculum: the real training distribution (snapshot-mix resets -- see
# g1_block_stack_3stage2_rl_env_cfg.py's G1BlockStack3Stage2ReverseCurriculumEnvCfg)
gym.register(
    id="Isaac-Stack-Blocks-G1-RL-Stage2-RC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_block_stack_3stage2_rl_env_cfg:G1BlockStack3Stage2ReverseCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_g1_block:G1BlockStackPPORunnerCfg",
    },
    disable_env_checker=True,
)
# BC-initialized PPO fine-tuning config -- Run F/G (fresh init, std 1.0) never saw a success,
# block_a_off_b terminating ~93-97% of episodes at ~17/500 steps, same failure Stage 1's own
# Franka history had before a BC warm start fixed it. See G1BlockStack3Stage2BCPPORunnerCfg.
gym.spec("Isaac-Stack-Blocks-G1-RL-Stage2-RC-v0").kwargs["rsl_rl_bc_cfg_entry_point"] = (
    f"{agents.__name__}.rsl_rl_ppo_cfg_g1_block:G1BlockStack3Stage2BCPPORunnerCfg"
)

# -- KAN-49: from-scratch E2E RL, no BC/warm-start, milestone + potential-shaping sparse reward --
# see g1_block_stack_rl_sparse_env_cfg.py's own module docstring. Plain (non-BC) PPO runner cfg
# on purpose: there is no pretrained checkpoint to fine-tune from.
gym.register(
    id="Isaac-Stack-Blocks-G1-RL-Sparse-S1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_block_stack_rl_sparse_env_cfg:G1BlockStackRLSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_g1_block:G1BlockStackPPORunnerCfg",
    },
    disable_env_checker=True,
)
