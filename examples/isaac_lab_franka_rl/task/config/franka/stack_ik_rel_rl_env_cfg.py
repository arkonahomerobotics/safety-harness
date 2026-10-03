# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged-state, RL-trainable variant of the IK-Rel Franka cube-stack task.

Reward design (v5, curriculum-split): a pure one-time-achievement reward (grasp/lift/align/
release/stack) turned out to still stall on discovering the first grasp -- once the capped
``reaching`` budget is spent there's no further gradient near the cube, and worse, a lucky early
grasp followed by an (inevitable, since placement hasn't been learned yet) misaligned release
netted *negative* reward under the original -3 penalty, actively teaching the policy to avoid
grasping rather than reinforcing it. Fixed by splitting into two curriculum phases that share the
same ``staged_achievements`` reward function with different point values, and are meant to be
trained in sequence (resume the Phase 1 checkpoint into Phase 2 -- adding new goals on top of an
already-good grasp/lift policy is the case where resuming makes sense, unlike our earlier reward
changes which were reversing an already-reinforced bad habit):

  Phase 1 (``-Phase1-v0``): only grasp/lift/reaching pay anything -- align, release, and stack are
  all zeroed out -- so the only thing being learned is reliably grasping and lifting the cube,
  with nothing to build a "grasping is bad" association out of.

  Phase 2 (``-v0``, the full task): once grasp/lift is reliable, resume into this config, which
  adds align/release/stack -- and removes the misaligned-release penalty entirely (was -3), so an
  attempted-but-imprecise placement is never worse than not having tried.

  Phase 2a (``-Phase2a-v0``): after Phase 2 training plateaued with ``align_best`` sitting around
  0.55-0.58 (well short of the 1.0 max), the working theory was that ``stack1``'s large, rare,
  high-variance +5 reward was adding enough noise to PPO's value estimates to drown out the
  smaller, smoother ``align_best`` gradient underneath it -- not a genuine incentive conflict
  (more precision only ever helps stack1), just a training-dynamics one. Phase 2a zeroes out
  ``release_aligned``/``release_misaligned``/``stack_reward``, leaving only reach/grasp/lift/align,
  to let precision refine cleanly; resume Phase 2a's result back into Phase 2 afterward. Safe to
  resume into (and out of) without restarting, since it only removes incentive rather than
  reversing anything the policy already relies on (release stays unpenalized either way).

That policy is then rolled out (same way ``expert_stack.py`` was) to generate a better GR00T
training dataset.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.config.franka.stack_ik_rel_env_cfg import FrankaCubeStackEnvCfg
from isaaclab_tasks.contrib.stack.mdp import rl_actions, rl_events, rl_rewards, robosuite_rewards

# Isaac Lab 3.x / Isaac Sim 6.x Franka asset: the links are nested under Robot/Geometry/panda_link0/.../panda_hand
# (verified with franka_rl/inspect_prims.py), not directly under Robot/ as on the old Brev box. The 3.x PhysX
# ContactSensor splits ``prim_path`` into a parent expression and a leaf-name regex and then searches the parent's
# whole subtree for contact-reporting bodies with that name, so "Robot/panda_leftfinger" still resolves to the nested
# link (a regex group such as "(Geometry/.*/)?" in the path is NOT accepted: the path is split on "/" first).
# Filter expressions are NOT subtree-searched, which is fine: only the cubes are used as filters, and the block USDs
# still carry their rigid body on the child mesh "Cube". smoke_test_rl_env.py prints the resolved sensor bodies.
_ROBOT_LINK = "{ENV_REGEX_NS}/Robot/"
_HAND = _ROBOT_LINK + "panda_hand"
_LEFT_FINGER = _ROBOT_LINK + "panda_leftfinger"
_RIGHT_FINGER = _ROBOT_LINK + "panda_rightfinger"
_BLUE = "{ENV_REGEX_NS}/Cube_1/Cube"
_RED = "{ENV_REGEX_NS}/Cube_2/Cube"
_GREEN = "{ENV_REGEX_NS}/Cube_3/Cube"

SNAPSHOT_DIR = "/workspace/isaaclab/snapshots"


@configclass
class RewardsCfg:
    """Phase 2 (full task) reward: grasp=+1, lift=+1, release_aligned=+1, release_misaligned=0
    (no penalty -- see module docstring), stacked=+5, reaching capped at +0.5 total per stage,
    align capped at +1.0 total per stage (running best-proximity-while-lifted, see
    ``staged_achievements``'s docstring).
    """

    staged_achievements = RewTerm(
        func=rl_rewards.staged_achievements,
        weight=1.0,
        params={"release_misaligned_penalty": 0.0},
    )

    # penalties -- small relative to the achievement terms above
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # targets the gripper open/close jitter observed from the imitation-learned policy
    gripper_smoothness = RewTerm(func=rl_rewards.gripper_action_rate_l2, weight=-1e-2)


@configclass
class Phase1RewardsCfg:
    """Phase 1 (grasp+lift only) reward: align/release/stack all zeroed out, so grasping and
    lifting are the only things that pay anything -- see module docstring for why.
    """

    staged_achievements = RewTerm(
        func=rl_rewards.staged_achievements,
        weight=1.0,
        params={
            "max_align_reward": 0.0,
            "release_aligned_reward": 0.0,
            "release_misaligned_penalty": 0.0,
            "stack_reward": 0.0,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    gripper_smoothness = RewTerm(func=rl_rewards.gripper_action_rate_l2, weight=-1e-2)


@configclass
class Phase2aRewardsCfg:
    """Phase 2a (reach+grasp+lift+align only) reward: release_aligned/release_misaligned/stack_reward
    all zeroed out, so nothing distracts from refining ``align_best`` -- see module docstring for why.

    Stage-2 rewards are also disabled: otherwise finishing stage 1 unlocks up to 3.5 points of
    stage-2 reach/grasp/lift/align, which outweighs the last ~0.15 of stage-1 align, so the policy
    learned to release as soon as the cube was in the (5cm) stacking window instead of refining.

    With stage 2 off, nothing rewarded finishing: the align ratchet pays only for new closest
    approaches, so a rollout showed every env hovering above cube_1 without settling. A one-time
    ``placement_reward`` (+1, first aligned release per episode: xy < 5cm, height within 5mm) gives
    a precise, released placement as the way to finish.
    """

    staged_achievements = RewTerm(
        func=rl_rewards.staged_achievements,
        weight=1.0,
        params={
            "release_aligned_reward": 0.0,
            "release_misaligned_penalty": 0.0,
            "stack_reward": 0.0,
            "stage2_reward_enabled": False,
            "placement_reward": 1.0,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    gripper_smoothness = RewTerm(func=rl_rewards.gripper_action_rate_l2, weight=-1e-2)


@configclass
class FrankaCubeStackRLEnvCfg(FrankaCubeStackEnvCfg):
    """State-based, RL-trainable Franka cube-stack environment (IK-Rel actions), Phase 2 (full task)."""

    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # RL rollouts don't need the (unpopulated) rgb_camera obs group or subtask_terms group
        # that the imitation-only base config carries -- drop them to save a little overhead.
        del self.observations.rgb_camera
        del self.observations.subtask_terms
        # the imitation pipeline wants a named dict of observations (for HDF5 export); RSL-RL's
        # default MLP actor/critic wants a single flat Box, so concatenate for this variant.
        self.observations.policy.concatenate_terms = True


@configclass
class FrankaCubeStackRLEnvCfg_PLAY(FrankaCubeStackRLEnvCfg):
    """Smaller-scale variant for interactive play/eval."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeStackRLPhase1EnvCfg(FrankaCubeStackEnvCfg):
    """Phase 1: grasp+lift only. Same scene/actions/observations as Phase 2 -- only the reward
    differs -- so a checkpoint trained here loads cleanly into Phase 2 for continued training.
    """

    rewards: Phase1RewardsCfg = Phase1RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        del self.observations.rgb_camera
        del self.observations.subtask_terms
        self.observations.policy.concatenate_terms = True


@configclass
class FrankaCubeStackRLPhase2aEnvCfg(FrankaCubeStackEnvCfg):
    """Phase 2a: reach+grasp+lift+align only, no release/stack incentive. Same observations as
    Phase 2 -- only the reward and (see below) the arm action differ.

    ``align_best`` plateaued around 0.53-0.55 (right at the tanh-potential value for the 5cm
    alignment boundary -- ``1 - tanh(0.05/0.1) = 0.538``) instead of climbing toward 1.0, while
    the policy's action std stayed flat/slightly rising (~0.80) rather than shrinking. Root cause:
    the base arm action (``DifferentialInverseKinematicsActionCfg``, relative-pose mode) used a
    single fixed ``scale=0.5`` -- a raw action of 1.0 commands up to a 0.5m EE delta per control
    step -- so with action std ~0.8 in those same units, a single sampled step could swing by
    roughly 0.4m, two orders of magnitude coarser than the ~5mm/5cm precision the task needs for
    the final approach. Cutting the fixed scale to 0.1 unblocked most of it (``align_best`` jumped
    to ~0.80), but then plateaued a *second* time at that finer scale -- a single fixed value is
    a compromise between "small enough to hold a precise final position" and "large enough to
    cover the initial approach in a reasonable number of steps," and no fixed value serves both.

    So the arm action here is ``DistanceScaledIKRelActionCfg`` instead: ``scale`` is recomputed
    every step from the same guidance distance the reward's own reaching/align terms use (EE to
    target cube while not yet holding it, held cube to destination once lifted) -- ``max_scale``
    (0.1, matching what already proved sufficient for reach/grasp/lift, so far-field behavior is
    unchanged) at or beyond ``distance_cap``, shrinking smoothly to ``min_scale`` right at the
    target. Reach/grasp and placement use separate ramps: linear down to 0.02 while reaching (which
    already grasped reliably) and quadratic down to 0.005 once the cube is lifted. A linear ramp
    everywhere left the final placement too coarse (2-3cm steps inside the last 2cm); a quadratic
    ramp everywhere fixed that but also shrank the grasp-approach steps, and grasping degraded. Scoped to Phase 2a only (not Phase 1/Phase 2/the base imitation task); this changes
    the action-space dynamics, not just the reward, so a checkpoint trained under the old action
    isn't guaranteed to transfer cleanly -- verify with a cheap probe before a full run.
    """

    rewards: Phase2aRewardsCfg = Phase2aRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        del self.observations.rgb_camera
        del self.observations.subtask_terms
        self.observations.policy.concatenate_terms = True
        old_arm_action = self.actions.arm_action
        self.actions.arm_action = rl_actions.DistanceScaledIKRelActionCfg(
            asset_name=old_arm_action.asset_name,
            joint_names=old_arm_action.joint_names,
            body_name=old_arm_action.body_name,
            controller=old_arm_action.controller,
            body_offset=old_arm_action.body_offset,
            scale=old_arm_action.scale,
            max_scale=0.1,
            distance_cap=0.15,
            reach_min_scale=0.02,
            reach_ramp_exponent=1.0,
            place_min_scale=0.005,
            place_ramp_exponent=2.0,
        )
        self.actions.arm_action.scale = 0.1


@configclass
class RobosuiteStackRewardsCfg:
    """robosuite's Stack shaped reward (``mdp/robosuite_rewards.py``) in place of the staged
    achievements, plus the same small smoothness penalties as the other RL phases.
    """

    stack = RewTerm(func=robosuite_rewards.robosuite_stack_reward, weight=1.0)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    gripper_smoothness = RewTerm(func=rl_rewards.gripper_action_rate_l2, weight=-1e-2)


@configclass
class FrankaCubeStackRLRobosuiteEnvCfg(FrankaCubeStackRLPhase2aEnvCfg):
    """Same scene, actions and observations as Phase 2a; only the reward changes, to robosuite's
    Stack reward. Adds the contact sensors it needs: red cube vs blue cube, and each finger vs red.
    """

    rewards: RobosuiteStackRewardsCfg = RobosuiteStackRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn = self.scene.robot.spawn.replace(activate_contact_sensors=True)
        self.scene.cube_2.spawn = self.scene.cube_2.spawn.replace(activate_contact_sensors=True)
        # the block USDs carry their rigid body on the child mesh prim "Cube", not the Cube_N Xform
        self.scene.red_blue_contact = ContactSensorCfg(prim_path=_RED, filter_prim_paths_expr=[_BLUE])
        self.scene.left_finger_red_contact = ContactSensorCfg(prim_path=_LEFT_FINGER, filter_prim_paths_expr=[_RED])
        self.scene.right_finger_red_contact = ContactSensorCfg(prim_path=_RIGHT_FINGER, filter_prim_paths_expr=[_RED])


@configclass
class FrankaCubeStackRLRobosuiteSnapEnvCfg(FrankaCubeStackRLRobosuiteEnvCfg):
    """robosuite reward plus start-state resets: half the episodes begin from recorded "red cube held
    above the blue cube" states (see ``mdp/rl_events.py``), so release attempts are common.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_snapshot = EventTerm(
            func=rl_events.reset_from_snapshots,
            mode="reset",
            params={"snapshot_path": f"{SNAPSHOT_DIR}/expert_carry_onto_blue.pt", "prob": 0.5},
        )


@configclass
class RobosuiteStackSparseRewardsCfg(RobosuiteStackRewardsCfg):
    """robosuite's sparse Stack reward (2.0 only while stacked): holding the cube pays nothing, which
    removes the "hold it over the blue cube" optimum the shaped reward settled into.
    """

    stack = RewTerm(func=robosuite_rewards.robosuite_stack_reward, weight=1.0, params={"reward_shaping": False})


@configclass
class FrankaCubeStackRLRobosuiteSnapSparseEnvCfg(FrankaCubeStackRLRobosuiteSnapEnvCfg):
    """Sparse robosuite reward plus expert start states (half the episodes begin with the red cube
    carried onto the blue one), which is what makes the sparse reward reachable.
    """

    rewards: RobosuiteStackSparseRewardsCfg = RobosuiteStackSparseRewardsCfg()


@configclass
class RobosuiteStackSeatedRewardsCfg(RobosuiteStackRewardsCfg):
    """robosuite's shaped Stack reward with the align term measured as 3D distance to the seated spot
    on top of the blue cube (``align_mode="seated3d"``). With robosuite's horizontal-only align, holding
    the cube ~12 cm above blue scored the same as holding it on blue, so the policy hovered high and its
    releases were drops; this makes coming down pay, after which releasing is safe and pays 2.0.
    """

    stack = RewTerm(func=robosuite_rewards.robosuite_stack_reward, weight=1.0, params={"align_mode": "seated3d"})


@configclass
class FrankaCubeStackRLRobosuiteSnapSeatedEnvCfg(FrankaCubeStackRLRobosuiteSnapEnvCfg):
    """Seated-3D align variant of the robosuite reward, plus expert start states."""

    rewards: RobosuiteStackSeatedRewardsCfg = RobosuiteStackSeatedRewardsCfg()


@configclass
class FullStackSparseRewardsCfg:
    """Two-level sparse reward for the three-cube tower (``robosuite_rewards.full_stack_sparse_reward``),
    plus the same small smoothness penalties as the other RL phases.
    """

    stack = RewTerm(func=robosuite_rewards.full_stack_sparse_reward, weight=1.0)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    gripper_smoothness = RewTerm(func=rl_rewards.gripper_action_rate_l2, weight=-1e-2)


@configclass
class FrankaCubeStackRLFullSparseSnapEnvCfg(FrankaCubeStackRLRobosuiteSnapEnvCfg):
    """Stage 2, the full tower: red on blue, then green on red. Same recipe that solved stage 1 (sparse
    reward + expert start states), with start states mixing both placements, and contact sensors for green.
    """

    rewards: FullStackSparseRewardsCfg = FullStackSparseRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.cube_3.spawn = self.scene.cube_3.spawn.replace(activate_contact_sensors=True)
        self.scene.green_red_contact = ContactSensorCfg(prim_path=_GREEN, filter_prim_paths_expr=[_RED])
        self.scene.left_finger_green_contact = ContactSensorCfg(prim_path=_LEFT_FINGER, filter_prim_paths_expr=[_GREEN])
        self.scene.right_finger_green_contact = ContactSensorCfg(
            prim_path=_RIGHT_FINGER, filter_prim_paths_expr=[_GREEN]
        )
        # the reward pays every step the tower stands; ending the episode at success would cut that off
        self.terminations.success = None
        self.events.reset_from_snapshot.params = {
            "snapshot_path": f"{SNAPSHOT_DIR}/expert_stage1_stage2_mix.pt",
            "prob": 0.5,
        }


@configclass
class TowerOnlySparseRewardsCfg(FullStackSparseRewardsCfg):
    """Tower-only sparse reward: nothing for red-on-blue alone, 2.0 per step while the full tower stands.
    With red-on-blue paying on its own, the policy protected that safe reward and learned to avoid green.
    """

    stack = RewTerm(
        func=robosuite_rewards.full_stack_sparse_reward, weight=1.0, params={"stage1_reward": 0.0, "tower_reward": 4.0}
    )


@configclass
class FrankaCubeStackRLTowerOnlySnapEnvCfg(FrankaCubeStackRLFullSparseSnapEnvCfg):
    """Full tower with the tower-only sparse reward, plus mixed stage-1/stage-2 expert start states."""

    rewards: TowerOnlySparseRewardsCfg = TowerOnlySparseRewardsCfg()


@configclass
class FrankaCubeStackRLStage2SkillEnvCfg(FrankaCubeStackRLTowerOnlySnapEnvCfg):
    """Stage-2 skill for skill chaining: put green on the red-on-blue stack. Run after the stage-1 policy
    has placed red, starting from the stage-1 weights.

    The policy observations rebind the cube roles (cube_1 := red, cube_2 := green, cube_3 := blue), so
    the stage-1 "pick cube_2, place it on cube_1, release, withdraw" skill applies to green-onto-red as-is
    (goal/object-parameterized skill reuse). The IK-Rel step-size ramp already switches to green-to-red
    once red is stacked. Every episode starts from an expert stage-2 state (red seated on blue), the reward
    is tower-only sparse, and the episode ends if red is knocked off blue.
    """

    def __post_init__(self):
        super().__post_init__()
        roles = {
            "cube_1_cfg": SceneEntityCfg("cube_2"),
            "cube_2_cfg": SceneEntityCfg("cube_3"),
            "cube_3_cfg": SceneEntityCfg("cube_1"),
        }
        for term in ("object", "cube_positions", "cube_orientations"):
            getattr(self.observations.policy, term).params.update(roles)
        self.terminations.red_off_blue = DoneTerm(func=robosuite_rewards.red_knocked_off_blue)
        self.events.reset_from_snapshot.params = {
            "snapshot_path": f"{SNAPSHOT_DIR}/stage2_allphases_neargoal.pt",
            "prob": 1.0,
        }


@configclass
class Stage2SkillContactRewardsCfg(TowerOnlySparseRewardsCfg):
    """Tower-only sparse reward plus an undesired-contact penalty: the hand or fingers touching the red-on-blue
    stack (hand-red, hand-blue, fingers-red, fingers-blue). The hand body hitting red was the top cause of
    knocking the stack over while grasping green next to it.
    """

    stack_contact = RewTerm(
        func=robosuite_rewards.robot_stack_contact,
        weight=-1.0,
        params={
            "sensor_names": [
                "hand_red_contact",
                "hand_blue_contact",
                "left_finger_red_contact",
                "right_finger_red_contact",
                "left_finger_blue_contact",
                "right_finger_blue_contact",
            ]
        },
    )


@configclass
class FrankaCubeStackRLStage2SkillContactEnvCfg(FrankaCubeStackRLStage2SkillEnvCfg):
    """Stage-2 skill with the robot-vs-stack contact penalty (extra contact sensors on blue and the hand)."""

    rewards: Stage2SkillContactRewardsCfg = Stage2SkillContactRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.cube_1.spawn = self.scene.cube_1.spawn.replace(activate_contact_sensors=True)
        self.scene.hand_red_contact = ContactSensorCfg(prim_path=_HAND, filter_prim_paths_expr=[_RED])
        self.scene.hand_blue_contact = ContactSensorCfg(prim_path=_HAND, filter_prim_paths_expr=[_BLUE])
        self.scene.left_finger_blue_contact = ContactSensorCfg(prim_path=_LEFT_FINGER, filter_prim_paths_expr=[_BLUE])
        self.scene.right_finger_blue_contact = ContactSensorCfg(prim_path=_RIGHT_FINGER, filter_prim_paths_expr=[_BLUE])


##
# PLAY variants (Isaac Lab 3.x port). Small scene, no observation corruption. ``play.py`` additionally calls
# ``play_mode()`` (caps num_envs at 50). The stage-1 PLAY variant drops the expert start states so a rollout shows the
# policy from normal starts, as eval_stacking.py measures it. The stage-2 skill PLAY variants keep their snapshot
# starts, because the skill's precondition (red already seated on blue) does not exist at a normal start.
##


def _play(cfg) -> None:
    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = False


@configclass
class FrankaCubeStackRLRobosuiteSnapSparseEnvCfg_PLAY(FrankaCubeStackRLRobosuiteSnapSparseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
        self.events.reset_from_snapshot = None


@configclass
class FrankaCubeStackRLStage2SkillEnvCfg_PLAY(FrankaCubeStackRLStage2SkillEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class FrankaCubeStackRLStage2SkillContactEnvCfg_PLAY(FrankaCubeStackRLStage2SkillContactEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


# --- Strict sparse-only variants: reward is paid only for the true completed state. -----------------
# No smoothness penalties, no shaping, no pretrained/BC weights. The step-size ramp in the action term and
# the optional expert start states are environment design, not reward; each variant says which it uses.


def _strict_sparse(cfg) -> None:
    for name in ("action_rate", "joint_vel", "gripper_smoothness"):
        setattr(cfg.rewards, name, None)


@configclass
class FrankaCubeStackRLStrictS1SnapEnvCfg(FrankaCubeStackRLRobosuiteSnapSparseEnvCfg):
    """Stage 1, sparse-only, 50% expert 'red carried onto blue' start states."""

    def __post_init__(self):
        super().__post_init__()
        _strict_sparse(self)


@configclass
class FrankaCubeStackRLStrictS1EnvCfg(FrankaCubeStackRLStrictS1SnapEnvCfg):
    """Stage 1, sparse-only, ordinary starts only (no expert-sampled start states)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_snapshot = None


@configclass
class FrankaCubeStackRLStrictStage2SkillEnvCfg(FrankaCubeStackRLStage2SkillEnvCfg):
    """Stage-2 skill (green onto red-on-blue), tower-only sparse, expert stage-2 start states."""

    def __post_init__(self):
        super().__post_init__()
        _strict_sparse(self)


@configclass
class FrankaCubeStackRLStrictFullSnapEnvCfg(FrankaCubeStackRLTowerOnlySnapEnvCfg):
    """Single end-to-end policy for the whole 3-cube tower, tower-only sparse, stage-1/2 expert start-state mix."""

    def __post_init__(self):
        super().__post_init__()
        _strict_sparse(self)


@configclass
class FrankaCubeStackRLStrictFullEnvCfg(FrankaCubeStackRLStrictFullSnapEnvCfg):
    """Single end-to-end policy for the whole 3-cube tower, tower-only sparse, ordinary starts only."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_snapshot = None


# --- Milestone variants: one-time sparse sub-goal bonuses (grasp, lift, over-target, placed) plus the per-step
# completion reward (``robosuite_rewards.milestone_stack_reward``). No continuous shaping, no smoothness
# penalties, no pretrained/BC weights.


@configclass
class MilestoneRewardsCfg:
    stack = RewTerm(func=robosuite_rewards.milestone_stack_reward, weight=1.0, params={"mode": "stage1"})


@configclass
class MilestoneStage2RewardsCfg:
    stack = RewTerm(func=robosuite_rewards.milestone_stack_reward, weight=1.0, params={"mode": "stage2"})


@configclass
class MilestoneFullRewardsCfg:
    stack = RewTerm(func=robosuite_rewards.milestone_stack_reward, weight=1.0, params={"mode": "full"})


@configclass
class FrankaCubeStackRLMilestoneS1SnapEnvCfg(FrankaCubeStackRLRobosuiteSnapSparseEnvCfg):
    """Stage 1, milestone reward, expert 'red carried onto blue' starts (set the fraction with ``prob``)."""

    rewards: MilestoneRewardsCfg = MilestoneRewardsCfg()



@configclass
class FrankaCubeStackRLMilestoneS1EnvCfg(FrankaCubeStackRLMilestoneS1SnapEnvCfg):
    """Stage 1, milestone reward, ordinary starts only."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_snapshot = None


@configclass
class FrankaCubeStackRLMilestoneStage2SkillEnvCfg(FrankaCubeStackRLStage2SkillEnvCfg):
    """Stage-2 skill (green onto red-on-blue), milestone reward, expert stage-2 start states."""

    rewards: MilestoneStage2RewardsCfg = MilestoneStage2RewardsCfg()



@configclass
class FrankaCubeStackRLMilestoneFullSnapEnvCfg(FrankaCubeStackRLTowerOnlySnapEnvCfg):
    """One end-to-end policy for the whole tower, milestone reward for both levels, expert start-state mix."""

    rewards: MilestoneFullRewardsCfg = MilestoneFullRewardsCfg()



@configclass
class FrankaCubeStackRLMilestoneFullEnvCfg(FrankaCubeStackRLMilestoneFullSnapEnvCfg):
    """One end-to-end policy for the whole tower, milestone reward, ordinary starts only."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_snapshot = None


# --- Milestone + potential-based shaping variants (policy-invariant by Ng et al. 1999; see milestone_stack_reward).


@configclass
class MilestonePotentialRewardsCfg:
    stack = RewTerm(
        func=robosuite_rewards.milestone_stack_reward, weight=1.0, params={"mode": "stage1", "potential_scale": 10.0}
    )


@configclass
class MilestonePotentialStage2RewardsCfg:
    stack = RewTerm(
        func=robosuite_rewards.milestone_stack_reward, weight=1.0, params={"mode": "stage2", "potential_scale": 10.0}
    )


@configclass
class MilestonePotentialFullRewardsCfg:
    stack = RewTerm(
        func=robosuite_rewards.milestone_stack_reward, weight=1.0, params={"mode": "full", "potential_scale": 10.0}
    )


@configclass
class FrankaCubeStackRLMilestonePotentialS1SnapEnvCfg(FrankaCubeStackRLMilestoneS1SnapEnvCfg):
    rewards: MilestonePotentialRewardsCfg = MilestonePotentialRewardsCfg()


@configclass
class FrankaCubeStackRLMilestonePotentialS1EnvCfg(FrankaCubeStackRLMilestoneS1EnvCfg):
    rewards: MilestonePotentialRewardsCfg = MilestonePotentialRewardsCfg()


@configclass
class FrankaCubeStackRLMilestonePotentialStage2SkillEnvCfg(FrankaCubeStackRLMilestoneStage2SkillEnvCfg):
    rewards: MilestonePotentialStage2RewardsCfg = MilestonePotentialStage2RewardsCfg()


@configclass
class FrankaCubeStackRLMilestonePotentialFullSnapEnvCfg(FrankaCubeStackRLMilestoneFullSnapEnvCfg):
    rewards: MilestonePotentialFullRewardsCfg = MilestonePotentialFullRewardsCfg()


@configclass
class FrankaCubeStackRLMilestonePotentialFullEnvCfg(FrankaCubeStackRLMilestoneFullEnvCfg):
    rewards: MilestonePotentialFullRewardsCfg = MilestonePotentialFullRewardsCfg()
