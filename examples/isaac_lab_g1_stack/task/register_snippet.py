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
