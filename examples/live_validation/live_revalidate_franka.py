"""Live re-validation of the safety harness on a running Isaac Lab Franka cube-stack env (v0.3.x code).

Every scenario runs the REAL IsaacLabCubeStackPerceptionAdapter / DynamicsAdapter / ActuatorGate
against live simulator state, with the pinned example config. A hazard is either physically created
in the simulator (cube velocities, a moved bystander cube, a joint written to its limit) or injected
by a wrapper that overrides exactly one thing in the otherwise-real WorldState (the technique the
earlier live runs used; each scenario says which). Each scenario states the check it must fire;
the result records exactly which checks fired. Nothing is inferred.
"""

import argparse
import dataclasses
import json
import math
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-v0")
parser.add_argument("--harness", type=str, default="/workspace/safety_harness")
parser.add_argument("--out", type=str, default="/tmp/live_franka.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, args_cli.harness)
from safety_harness import ActionSchemaRegistry, ActuatorGate, DecisionWatchdog  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.isaac_lab import (  # noqa: E402
    IsaacLabCubeStackDynamicsAdapter, IsaacLabCubeStackPerceptionAdapter,
)
from safety_harness.integrity import read_digest_file  # noqa: E402
from safety_harness.schema import (  # noqa: E402
    Action, AgentCategory, EnvironmentSignals, FallConsequence, ObservedRegion, Pose, TrackedAgent,
)

CFG = f"{args_cli.harness}/configs/example_action_schema.yaml"
PIN = read_digest_file(CFG + ".sha256")

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
env.reset()
robot = env.scene["robot"]
ENV_IDS = torch.tensor([0], device=env.device)
RESULTS = []


def settle(n=5):
    for _ in range(n):
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)


def fresh():
    env.reset()
    settle(10)


class Wrap(IsaacLabCubeStackPerceptionAdapter):
    """The real adapter, plus an optional single override of the real WorldState."""

    def __init__(self, mutate=None, **kw):
        super().__init__(env, env_index=0, **kw)
        self._mutate = mutate

    def get_world_state(self):
        s = super().get_world_state()
        return self._mutate(s) if self._mutate else s


class SlowDynamics(IsaacLabCubeStackDynamicsAdapter):
    def predict_trajectory(self, state, action, horizon_s):
        time.sleep(0.15)  # a dynamics model that stalls past the 0.1s decision deadline
        return super().predict_trajectory(state, action, horizon_s)


def cube_pos(state, cid):
    return next(o for o in state.objects if o.object_id == cid).pose.position


def run(label, kind, expect, perception=None, action_fn=None, dynamics=None, schema=None, horizon_s=1.0):
    """expect: 'permit' or the name of the check that must be among the failures."""
    perception = perception or Wrap()
    schema = schema or ActionSchemaRegistry.from_yaml(CFG, expected_digest=PIN)
    gate = ActuatorGate(perception, dynamics or IsaacLabCubeStackDynamicsAdapter(), FreezeInPlaceFallback(),
                        InMemoryLogger(), schema, horizon_s=horizon_s)
    state = IsaacLabCubeStackPerceptionAdapter(env, env_index=0).get_world_state()
    action = action_fn(state)
    gate.gate(action)  # warm-up call (first-call Python import/JIT latency is not the scenario)
    d = gate.gate(action)
    fired = [r.name for r in d.precondition_results if not r.satisfied]
    ok = (d.verdict.value == "permit") if expect == "permit" else (d.verdict.value == "block" and expect in fired)
    RESULTS.append({"label": label, "kind": kind, "expect": expect, "verdict": d.verdict.value, "fired": fired, "pass": ok})
    print(f"[{'PASS' if ok else 'FAIL'}] {label}  ({kind})\n        expect={expect} verdict={d.verdict.value} fired={fired}")
    return d


def grasp(grip=5.0, **extra):
    return lambda s: Action("grasp", {"object_id": "cube_2", "target_position": cube_pos(s, "cube_2"), "grip_force_n": grip, **extra})


def set_obj(cid, f):
    return lambda s: dataclasses.replace(s, objects=tuple(f(o) if o.object_id == cid else o for o in s.objects))


# ---------------------------------------------------------------------------------------------------
fresh()
run("nominal grasp of a resting cube", "nominal", "permit", action_fn=grasp())

fresh()
c = env.scene["cube_2"]
v = torch.zeros(1, 6, device=env.device); v[0, 2] = -0.5
c.write_root_velocity_to_sim(v)
run("cube falling at 0.5 m/s", "physical", "current_position_confirmed_stable", action_fn=grasp())

fresh()
v = torch.zeros(1, 6, device=env.device); v[0, 0], v[0, 1] = 1.5, 0.8
c.write_root_velocity_to_sim(v)
run("cube knocked sideways at 1.7 m/s", "physical", "current_position_confirmed_stable", action_fn=grasp())

fresh()
s0 = IsaacLabCubeStackPerceptionAdapter(env, env_index=0).get_world_state()
p2 = cube_pos(s0, "cube_2")
b = env.scene["cube_1"]
pose = b.data.root_pose_w.torch.clone() if hasattr(b.data, "root_pose_w") else torch.cat([b.data.root_pos_w.torch, b.data.root_quat_w.torch], 1).clone()
pose[0, :3] = torch.tensor([p2[0] + 0.10, p2[1], p2[2]], device=env.device) + env.scene.env_origins[0]
b.write_root_pose_to_sim(pose)
settle(3)
run("hazard-classified bystander cube physically moved 10cm from the target", "physical+classification",
    "swept_path_clear_of_risky_objects",
    perception=Wrap(set_obj("cube_1", lambda o: dataclasses.replace(o, cleared_for_interaction=False, fall_consequence=FallConsequence.HAZARDOUS_RELEASE))),
    action_fn=grasp())

for chk in ("object_cleared_for_interaction", "current_position_confirmed_stable", "fall_consequence_acceptable"):
    fresh()
    run(f"target not confirmed by perception ({chk})", "classification", chk,
        perception=Wrap(set_obj("cube_2", lambda o: dataclasses.replace(o, cleared_for_interaction=False, supported_stably=None, fall_consequence=FallConsequence.UNKNOWN))),
        action_fn=grasp())

fresh()
run("visibility 0.49 (below 0.5)", "injected", "visibility_above_threshold",
    perception=Wrap(lambda s: dataclasses.replace(s, environment=EnvironmentSignals(visibility_confidence=0.49))), action_fn=grasp())

for fld, chk in (("estimated_mass_kg", "mass_within_force_budget"), ("class_confidence", "object_hazard_confirmed"),
                 ("pose_confidence", "object_pose_confirmed")):
    fresh()
    run(f"NaN {fld} on the live target", "injected-fault", chk,
        perception=Wrap(set_obj("cube_2", lambda o, f=fld: dataclasses.replace(o, **{f: math.nan}))), action_fn=grasp())


def with_agent(offset, speed=1.5, category=AgentCategory.ADULT, stature=1.75, relative_to="cube_2"):
    def m(s):
        base = cube_pos(s, relative_to) if relative_to else s.robot.end_effector_pose.position
        a = TrackedAgent("person", Pose(tuple(p + o for p, o in zip(base, offset))), tracking_confidence=0.95,
                         worst_case_speed_mps=speed, category=category, stature_m=stature)
        return dataclasses.replace(s, agents=(a,))
    return m


for chk in ("swept_path_clear_of_agents", "iso15066_separation_distance_maintained", "iso15066_power_force_limiting", "reduced_speed_near_human"):
    fresh()
    run(f"adult 0.3m from the grasp target ({chk})", "injected-agent", chk, perception=Wrap(with_agent((0.3, 0.0, 0.0))), action_fn=grasp())

# --- the two newly wired checks ---------------------------------------------------------------------
fresh()
jid = 0
lim = robot.data.soft_joint_pos_limits.torch[0, jid].tolist()
q = robot.data.joint_pos.torch.clone()
q[0, jid] = lim[1] - 0.01
robot.write_joint_position_to_sim_index(position=q, env_ids=ENV_IDS)
robot.set_joint_position_target_index(target=q, env_ids=ENV_IDS)
# no settle: an unactuated physics step lets gravity/PD pull the joint back out of the margin
# (first attempt measured 0.029 rad from the limit -- correctly permitted, but not the scenario)
measured = float(robot.data.joint_pos.torch[0, jid])
print(f"        joint {jid} soft limits {lim}, commanded {lim[1]-0.01:.4f}, measured {measured:.4f}")
run(f"arm joint {jid} physically within 0.01 rad of its limit (measured {measured:.4f}, limit {lim[1]:.4f})",
    "physical", "joint_position_limits_respected", action_fn=grasp())

fresh()
reach_far = lambda s: Action("reach", {"target_position": tuple(p + d for p, d in zip(s.robot.end_effector_pose.position, (2.0, 0.0, 0.0))), "duration_s": 0.05})
run("reach commanded at ~40 m/s (2.0m in 0.05s)", "commanded", "cartesian_speed_within_limits", action_fn=reach_far)
reach_ok = lambda s: Action("reach", {"target_position": tuple(p + d for p, d in zip(s.robot.end_effector_pose.position, (0.0, 0.0, 0.1))), "duration_s": 1.0})
run("control: reach 10cm at 0.1 m/s", "nominal", "permit", action_fn=reach_ok)

# --- speed model: the capped extrapolation under-predicted a faster controller's reach ------------
fresh()
slow_teammate = with_agent((2.8, 0.0, 0.0), speed=0.15, relative_to=None)
reach4 = lambda extra: (lambda s: Action("reach", {"target_position": tuple(p + d for p, d in zip(s.robot.end_effector_pose.position, (4.0, 0.0, 0.0))), **extra}))
run("slow teammate 2.8m ahead; speed unstated (old capped model, 0.5 m/s)", "speed-model", "permit",
    perception=Wrap(slow_teammate), action_fn=reach4({}), horizon_s=2.0)
run("same, but the controller commands 1.6 m/s (within the 1.7 m/s rating)", "speed-model", "swept_path_clear_of_agents",
    perception=Wrap(slow_teammate), action_fn=reach4({"commanded_speed_mps": 1.6}), horizon_s=2.0)

# --- the two wired checks the first audit found never blocked live ----------------------------------
fresh()
run("spill reported on the work surface", "injected", "environment_hazard_clear",
    perception=Wrap(lambda s: dataclasses.replace(s, environment=EnvironmentSignals(visibility_confidence=1.0, surface_hazards=frozenset({"spill"})))),
    action_fn=grasp())
fresh()
# robot_state_confirmed fires only on robot=None -- but the dynamics adapter needs the robot's pose
# first, so the gate blocks on adapter_error before any check runs. Recorded as what it is.
run("robot proprioception missing entirely (robot=None)", "injected-fault", "adapter_error",
    perception=Wrap(lambda s: dataclasses.replace(s, robot=None)), action_fn=lambda s: grasp()(IsaacLabCubeStackPerceptionAdapter(env, env_index=0).get_world_state()))
fresh()
run("joint readings NaN (robot state present but corrupt)", "injected-fault", "joint_position_limits_respected",
    perception=Wrap(lambda s: dataclasses.replace(s, robot=dataclasses.replace(s.robot, joint_positions=(math.nan,) * len(s.robot.joint_positions)))),
    action_fn=grasp())

# --- place (never gated live before) ----------------------------------------------------------------
place = lambda s: Action("place", {"object_id": "cube_2", "target_surface_id": "cube_1",
                                   "target_position": tuple(p + d for p, d in zip(cube_pos(s, "cube_1"), (0.0, 0.0, 0.05)))})
fresh()
run("place onto a resting cube", "nominal", "permit", action_fn=place)
fresh()
v = torch.zeros(1, 6, device=env.device); v[0, 0] = 0.6
env.scene["cube_1"].write_root_velocity_to_sim(v)
run("place onto a cube that is sliding at 0.6 m/s", "physical", "destination_confirmed_stable_and_clear", action_fn=place)

# --- 0.3.0 checks, first live run on this robot -----------------------------------------------------
fresh()
run("state built from 1s-old sensor data", "injected", "sensor_data_fresh",
    perception=Wrap(lambda s: dataclasses.replace(s, sensor_timestamp=time.time() - 1.0)), action_fn=grasp())
fresh()
run("path runs through space no sensor observed", "injected", "swept_path_observed",
    perception=Wrap(lambda s: dataclasses.replace(s, observed_regions=(ObservedRegion((3.0, 3.0, 0.0), (4.0, 4.0, 1.0)),))),
    action_fn=grasp())
fresh()
run("60 N grip on the fragile cube (cap 15 N)", "commanded", "payload_and_grip_force_within_limits", action_fn=grasp(grip=60.0))
fresh()
run("5 kg object reported (rated payload 3 kg)", "injected", "payload_and_grip_force_within_limits",
    perception=Wrap(set_obj("cube_2", lambda o: dataclasses.replace(o, estimated_mass_kg=5.0))), action_fn=grasp(grip=60.0))
fresh()
run("child 2m from the path", "injected-agent", "vulnerable_bystander_protected",
    perception=Wrap(with_agent((2.0, 0.0, 0.0), category=AgentCategory.CHILD, stature=1.1)), action_fn=grasp())


class MutatingPerception(Wrap):
    """Rewrites the proposed action's (mutable) params while the gate is still checking."""
    target = None

    def get_world_state(self):
        if MutatingPerception.target is not None:
            MutatingPerception.target.params["grip_force_n"] = 200.0
        return super().get_world_state()


fresh()
st = IsaacLabCubeStackPerceptionAdapter(env, env_index=0).get_world_state()
act = Action("grasp", {"object_id": "cube_2", "target_position": cube_pos(st, "cube_2"), "grip_force_n": 5.0})
gate = ActuatorGate(MutatingPerception(), IsaacLabCubeStackDynamicsAdapter(), FreezeInPlaceFallback(), InMemoryLogger(),
                    ActionSchemaRegistry.from_yaml(CFG, expected_digest=PIN))
MutatingPerception.target = act
d = gate.gate(act)
fired = [r.name for r in d.precondition_results if not r.satisfied]
ok = d.verdict.value == "block" and "command_integrity_verified" in fired
RESULTS.append({"label": "action rewritten (grip 5 -> 200 N) while being checked", "kind": "injected-attack",
                "expect": "command_integrity_verified", "verdict": d.verdict.value, "fired": fired, "pass": ok})
print(f"[{'PASS' if ok else 'FAIL'}] action rewritten while being checked\n        fired={fired}")

fresh()
reg = ActionSchemaRegistry.from_yaml(CFG, expected_digest=PIN)
for chk in reg._schemas["grasp"].checks:
    if chk["name"] == "mass_within_force_budget":
        chk["kwargs"]["force_budget_kg"] = 300.0  # tamper with the live, already-verified config
run("live config tampered after verification (force budget 3 -> 300 kg)", "injected-attack", "config_integrity_verified",
    action_fn=grasp(), schema=reg)
fresh()
run("dynamics model stalls 0.15s (deadline 0.1s)", "injected", "decision_within_deadline", action_fn=grasp(), dynamics=SlowDynamics())

# --- watchdog (runtime component, not a check) --------------------------------------------------------
fresh()
wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=0.2)
g = ActuatorGate(Wrap(), IsaacLabCubeStackDynamicsAdapter(), FreezeInPlaceFallback(), InMemoryLogger(),
                 ActionSchemaRegistry.from_yaml(CFG, expected_digest=PIN), watchdog=wd)
st = IsaacLabCubeStackPerceptionAdapter(env, env_index=0).get_world_state()
g.gate(grasp()(st))
dec = g.gate(grasp()(st))
fresh_cmd = wd.command().action_type
time.sleep(0.3)
stale_cmd = wd.command().action_type
ok = dec.verdict.value == "permit" and fresh_cmd == "grasp" and stale_cmd == "freeze"
RESULTS.append({"label": "watchdog: fresh PERMIT executes, same PERMIT 0.3s later freezes", "kind": "runtime",
                "expect": "grasp then freeze", "verdict": f"{fresh_cmd} then {stale_cmd}", "fired": [], "pass": ok})
print(f"[{'PASS' if ok else 'FAIL'}] watchdog: {fresh_cmd} then {stale_cmd}")

n_ok = sum(r["pass"] for r in RESULTS)
print(f"\nLIVE FRANKA: {n_ok}/{len(RESULTS)} scenarios behaved as expected")
json.dump(RESULTS, open(args_cli.out, "w"), indent=1)
simulation_app.close()
