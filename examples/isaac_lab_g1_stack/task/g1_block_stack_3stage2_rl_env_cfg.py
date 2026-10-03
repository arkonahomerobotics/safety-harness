"""Stage 2 of the G1 three-block tower task: place block C on top of the already-stacked A-on-B
pair. A fresh policy and fresh observation (see mdp/rl_obs_g1_stage2.py) chained after a Stage-1
checkpoint by the chained evaluator -- this cfg's own normal reset is NOT the real training
distribution (that's a snapshot mix of expert mid-carry-of-C states and real Stage-1 handoff
states, wired in separately once Run D/E has a strong checkpoint to capture snapshots from); it
places A pre-seated on B as a standalone-trainable fallback so the task is well-posed even before
the snapshot mix exists.

Scene, actions and the two existing blocks (A, B) are otherwise identical to the 2-block task (see
g1_block_stack_rl_env_cfg.py's own docstring for the reachability/IK rationale) -- only the reward,
observation and termination layer changes, plus the new block C."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp import action_rate_l2, joint_vel_l2
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.locomanip_pick_place.g1_block_stack_rl_env_cfg import ActionsCfg, A_POS, B_POS, BLOCK, _block
from isaaclab_tasks.contrib.locomanip_pick_place.mdp import rl_obs_g1_stage2, rl_rewards_g1_stage2
from isaaclab_tasks.contrib.locomanip_pick_place.mdp.rl_rewards_g1 import grip_action_rate_l2

from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
    FixedBaseUpperBodyIKG1EnvCfg,
)

# C starts >=20cm center-to-center from B_POS (the A-on-B tower sits at x in [-0.26,-0.14]) so even
# a straight-line grasp approach has real clearance -- the first candidate (0.0, 0.35) was only
# 14cm from B_POS and the scripted expert's approach knocked the tower over reaching for it (see
# stack_expert_rl_stage2.py's own module docstring for the fix: rise-to-safe-height before any
# lateral move). reach_check_cpos.py confirmed this cell: 0.76-0.86cm pos error at roll 0-20deg,
# 21.96cm from B_POS.
C_POS = (0.05, 0.25, 0.72)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(func=rl_obs_g1_stage2.stack_state_stage2, params={"block_height": BLOCK})
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


_STACK3 = {
    "block_a_cfg": SceneEntityCfg("object"),
    "block_b_cfg": SceneEntityCfg("block_b"),
    "block_c_cfg": SceneEntityCfg("block_c"),
    "place_height": BLOCK,
    "finger_closed_threshold": 0.3,
    "success_hold_steps": 15,
    "success_max_speed": 0.03,
    "hand_clear_dist": 0.11,
    "hold_reward": 0.05,
}


@configclass
class RewardsCfg:
    staged_achievements_3stack = RewTerm(func=rl_rewards_g1_stage2.staged_achievements_3stack, weight=1.0, params=_STACK3)
    action_rate = RewTerm(func=action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})
    grip_smoothness = RewTerm(func=grip_action_rate_l2, weight=-1e-2)
    # Added 2026-10-03 (KAN-36): Run H/I's episode length shrank toward ~20 steps with block_a_off_b
    # at ~99% -- a reward-by-termination diagnostic found no per-step penalty was large enough to
    # explain it; the real driver was REWARD RATE, since the achievement budget above is one-time
    # and farmable again every fresh episode, and block_a_off_b (no penalty before this) was cheap
    # to trigger. These two terms close that gap -- see each one's own docstring for the reasoning.
    block_a_off_b_penalty = RewTerm(func=rl_rewards_g1_stage2.block_a_off_b_penalty, weight=1.0)
    tower_intact_reward = RewTerm(func=rl_rewards_g1_stage2.tower_intact_reward, weight=1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    block_c_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("block_c")}
    )
    block_b_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("block_b")}
    )
    # the small-scale failure (A knocked off B, not a catastrophic table-clearing drop) -- see
    # rl_rewards_g1_stage2.block_a_off_b's docstring for why this needs B's LIVE pose, not a static
    # rest height.
    block_a_off_b = DoneTerm(func=rl_rewards_g1_stage2.block_a_off_b, params={"place_height": BLOCK})
    # No success termination, same reasoning as Stage 1: the policy must learn to leave a finished
    # tower alone, not just reach it once -- see RewardsCfg's hold_reward and eval_policy.py's
    # --full_episode end-state judging.


@configclass
class G1BlockStack3Stage2EnvCfg(FixedBaseUpperBodyIKG1EnvCfg):
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # see g1_block_stack_rl_env_cfg.py's own __post_init__ for why the base class's
        # __post_init__ is deliberately not called.
        self.scene.left_hand_contact = None
        self.scene.right_hand_contact = None
        self.scene.robot_pov_cam = None
        self.decimation = 4
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2
        self.episode_length_s = 10.0

        self.scene.block_b = _block("{ENV_REGEX_NS}/BlockB", B_POS, (0.85, 0.2, 0.15))
        # A's "rest" position for Stage 2 is seated on B, not the table -- see module docstring.
        a_seated_pos = (B_POS[0], B_POS[1], B_POS[2] + BLOCK)
        self.scene.object = _block("{ENV_REGEX_NS}/Object", a_seated_pos, (0.15, 0.4, 0.85))
        self.scene.block_c = _block("{ENV_REGEX_NS}/BlockC", C_POS, (0.9, 0.9, 0.2))

        self.events.reset_block_b = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block_b")},
        )
        # small range -- A is meant to already be precisely seated on B at reset, not randomized
        # across the whole table like Stage 1's blocks.
        self.events.reset_block_a = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object")},
        )
        self.events.reset_block_c = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("block_c")},
        )


EXPERT_SNAPSHOT_PATH = "/workspace/isaaclab/g1_stage2_expert_snapshots.pt"  # stack_expert_rl_stage2.py --snapshots
HANDOFF_SNAPSHOT_PATH = "/workspace/isaaclab/g1_stage1_handoff_snapshots.pt"  # capture_stage1_handoff.py


@configclass
class G1BlockStack3Stage2ReverseCurriculumEnvCfg(G1BlockStack3Stage2EnvCfg):
    """Phase 2 + reverse curriculum: the real training distribution. Same reasoning as Stage 1's
    own reverse curriculum (the technique that unstuck the Franka/G1 stacking policies) but now
    mixing two different kinds of "mid-task" start: real Stage-1 handoff states (a genuine
    model_5797 2-block success, A really seated on B, hand clear) and expert mid-carry-of-C states
    (already holding C, approaching the tower). A single event (rl_events_g1.reset_stage2_mixed_snapshots)
    draws ONE bucket per env -- handoff / expert / plain default -- rather than two independent
    reset_from_snapshots calls, which would let ~(handoff_prob*expert_prob) of envs get both, the
    second overwriting the first's robot joints (see that function's own docstring for exactly what
    breaks). Defined last so it overrides whatever the ordinary per-block reset events above
    already produced, for the envs it picks."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab_tasks.contrib.locomanip_pick_place.mdp import rl_events_g1 as rl_events

        self.events.reset_stage2_mixed_snapshots = EventTerm(
            func=rl_events.reset_stage2_mixed_snapshots,
            mode="reset",
            params={
                "handoff_snapshot_path": HANDOFF_SNAPSHOT_PATH,
                "expert_snapshot_path": EXPERT_SNAPSHOT_PATH,
                "handoff_prob": 0.5,
                "expert_prob": 0.35,
                # remaining 0.15 of envs keep the plain default reset (A pre-seated on B, C random on table)
            },
        )
