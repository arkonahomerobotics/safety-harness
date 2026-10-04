"""G1 two-block stacking, from-scratch E2E RL: no BC/imitation, no warm-start, milestone + real
potential-based-shaping sparse reward (see rl_rewards_g1_sparse.milestone_stack_reward_g1 for the
full design rationale -- ported from the Franka team's own validated recipe, KAN-48/PR#42).

KAN-49 (Kaoru's direct instruction, relayed 2026-10-03 night): move both G1 and Franka to pure
E2E RL. A minimal fraction of episodes (``EXPERT_START_PROB``, well under the historical 50-85%
reverse-curriculum mixes used elsewhere in this project) start from the existing scripted-expert
snapshot file -- a pre-recorded SCRIPTED expert's example positions, not a trained/warm-started
policy and not an imitation loss on the actor -- specifically to solve the cold-start discovery
problem (a narrow 7-DOF grasp target that pure random exploration may never stumble into), per
Kaoru's own explicit allowance for that one purpose, not as a general curriculum crutch.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.locomanip_pick_place.g1_block_stack_rl_env_cfg import (
    SNAPSHOT_PATH,
    G1BlockStackRLEnvCfg,
)
from isaaclab_tasks.contrib.locomanip_pick_place.mdp import rl_rewards_g1_sparse

EXPERT_START_PROB = 0.15


@configclass
class SparseRewardsCfg:
    """Milestone + potential-shaping reward -- see rl_rewards_g1_sparse's own docstring for why
    there are deliberately no smoothness/action-rate penalties either.

    BUG FOUND 2026-10-04 (after Run L/L2/L3/M all stalled at "over" for ~6000+ iterations):
    potential_scale defaults to 0.0 in milestone_stack_reward_g1, and this cfg never overrode it,
    so every one of those runs trained on milestones ALONE -- exactly the "milestone-only was
    undiscoverable" failure mode this whole design exists to avoid (see Franka's own KAN-48
    history). The shaping code was written and never actually turned on. potential_scale=10
    matches Franka's own validated MilestonePotential config."""

    stack = RewTerm(
        func=rl_rewards_g1_sparse.milestone_stack_reward_g1, weight=1.0, params={"potential_scale": 10.0}
    )


@configclass
class G1BlockStackRLSparseEnvCfg(G1BlockStackRLEnvCfg):
    rewards: SparseRewardsCfg = SparseRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        from isaaclab_tasks.contrib.locomanip_pick_place.mdp import rl_events_g1 as rl_events

        self.events.reset_from_snapshots = EventTerm(
            func=rl_events.reset_from_snapshots,
            mode="reset",
            params={"snapshot_path": SNAPSHOT_PATH, "prob": EXPERT_START_PROB, "cube_names": ("object", "block_b")},
        )
