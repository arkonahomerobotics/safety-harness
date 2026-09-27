"""GPU-fast, RL-trainable G1 two-block stacking task: pick the blue block, stack it on the red one.

Scene: the fixed-base G1 at the packing table with two 4.5cm blocks, both inside the left wrist's
measured reachable band (reach_map.py -- the stock object spawn at (-0.35, 0.45) is outside it),
positions randomized +/-2cm per reset. Grasping is real contact physics; nothing is attached.

Actions (7): waist + left-arm relative-pose differential IK (6, batched on the GPU) + a scalar Dex3 grip
(1). The stock Pink IK action solves on the CPU per env (~85 env-steps/s here) -- unusable for PPO.

Rewards: the Franka-derived staged_achievements in stacking mode (the target is the live seat on the
red block), with the same two-phase curriculum: Phase 1 pays only reach/grasp/lift, Phase 2 adds
align/release/success with no misaligned-release penalty. An optional reverse-curriculum reset
starts a fraction of episodes from scripted-expert snapshots (block held above the red block).
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp import action_rate_l2, joint_vel_l2
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
    FixedBaseUpperBodyIKG1EnvCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp import rl_obs_g1, rl_rewards_g1
from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp.rl_actions_g1 import G1GripActionCfg

BLOCK = 0.045
A_POS = (-0.26, 0.36, 0.72)
B_POS = (-0.14, 0.36, 0.72)


def _block(prim_path: str, pos, color) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(0.0, 0.0, 0.0, 1.0)),  # xyzw
        spawn=sim_utils.CuboidCfg(
            size=(BLOCK, BLOCK, BLOCK),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


@configclass
class ActionsCfg:
    left_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        # waist included, as in the stock Pink IK action -- arm-only IK can't reach the grasp pose
        # without twisting the wrist into its limits (measured: blocks knocked, orientation drifts)
        joint_names=["waist_.*_joint", "left_shoulder_.*", "left_elbow_joint", "left_wrist_.*"],
        body_name="left_wrist_yaw_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=(0.02, 0.02, 0.02, 0.05, 0.05, 0.05),  # per 50Hz step: 2cm / ~3 deg at |a|=1
    )
    left_grip = G1GripActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(func=rl_obs_g1.stack_state, params={"block_height": BLOCK})
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


_STACK = {
    "place_asset_cfg": SceneEntityCfg("block_b"),
    "place_height": BLOCK,
    "finger_closed_threshold": 0.3,
    # real released placement only -- see staged_achievements' success comment
    "success_hold_steps": 15,
    "success_max_speed": 0.03,
    "hand_clear_dist": 0.11,
    "hold_reward": 0.05,  # per step the released stack stands (up to ~20 over an episode)
}


@configclass
class RewardsCfg:
    """Phase 2 (full task)."""

    staged_achievements = RewTerm(
        func=rl_rewards_g1.staged_achievements, weight=1.0, params={**_STACK, "release_misaligned_penalty": 0.0}
    )
    action_rate = RewTerm(func=action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})
    grip_smoothness = RewTerm(func=rl_rewards_g1.grip_action_rate_l2, weight=-1e-2)


@configclass
class Phase1RewardsCfg(RewardsCfg):
    """Phase 1: reach/grasp/lift only -- nothing yet to build a "grasping is bad" association from."""

    staged_achievements = RewTerm(
        func=rl_rewards_g1.staged_achievements,
        weight=1.0,
        params={
            **_STACK,
            "max_align_reward": 0.0,
            "release_aligned_reward": 0.0,
            "release_misaligned_penalty": 0.0,
            "success_reward": 0.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    block_a_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )
    block_b_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("block_b")}
    )
    # No success termination: the episode runs to time-out so the policy learns to leave a finished
    # stack alone (see staged_achievements' hold_reward). Evaluate by the end state at t=10s.


@configclass
class G1BlockStackRLEnvCfg(FixedBaseUpperBodyIKG1EnvCfg):
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # Deliberately not calling the base __post_init__: all it adds beyond these timing settings is
        # the Pink IK URDF path and the XR teleop pipeline, neither of which this task uses.
        self.decimation = 4
        self.sim.dt = 1 / 200  # 200Hz physics, 50Hz control -- same as the base task
        self.sim.render_interval = 2
        self.episode_length_s = 10.0  # 500 control steps; the scripted expert finishes in ~310
        self.scene.object = _block("{ENV_REGEX_NS}/Object", A_POS, (0.15, 0.4, 0.85))
        self.scene.block_b = _block("{ENV_REGEX_NS}/BlockB", B_POS, (0.85, 0.2, 0.15))

        rng = {"x": (-0.02, 0.02), "y": (-0.02, 0.02)}
        self.events.reset_block_a = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": rng, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object")},
        )
        self.events.reset_block_b = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": rng, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block_b")},
        )


@configclass
class G1BlockStackRLPhase1EnvCfg(G1BlockStackRLEnvCfg):
    rewards: Phase1RewardsCfg = Phase1RewardsCfg()


SNAPSHOT_PATH = "/workspace/isaaclab/g1_stack_snapshots.pt"  # from stack_expert_rl.py --snapshots


@configclass
class G1BlockStackRLReverseCurriculumEnvCfg(G1BlockStackRLEnvCfg):
    """Phase 2 + reverse curriculum: half the episodes start from a scripted-expert snapshot with the
    blue block held and aligned above the red one, so release attempts (and their outcomes) are
    common from the start -- the technique that unstuck the Franka stacking policy."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab_tasks.manager_based.manipulation.stack.mdp import rl_events

        # defined last, so it overrides the block randomization for the envs it picks
        self.events.reset_from_snapshots = EventTerm(
            func=rl_events.reset_from_snapshots,
            mode="reset",
            params={"snapshot_path": SNAPSHOT_PATH, "prob": 0.5, "cube_names": ("object", "block_b")},
        )
