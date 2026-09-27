"""Live re-validation of the safety harness on a running Isaac Lab ANYmal-C navigation env (v0.3.x
code): real base pose/velocity, real joint state, real foot geometry, through the real
IsaacLabAnymalNav adapters + ActuatorGate with the pinned example config. Hazards are physical where
the simulator allows (a leg joint written to its limit) and otherwise a single wrapper override of
the otherwise-real WorldState; each scenario says which and states the check it must fire."""

import argparse
import dataclasses
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Navigation-Flat-Anymal-C-v0")
parser.add_argument("--harness", type=str, default="/workspace/safety_harness")
parser.add_argument("--out", type=str, default="/tmp/live_anymal.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, args_cli.harness)
from safety_harness import ActionSchemaRegistry, ActuatorGate  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.isaac_lab_anymal import (  # noqa: E402
    IsaacLabAnymalNavDynamicsAdapter, IsaacLabAnymalNavPerceptionAdapter,
)
from safety_harness.integrity import read_digest_file  # noqa: E402
from safety_harness.schema import Action, AgentCategory, Pose, TrackedAgent  # noqa: E402

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


class Wrap(IsaacLabAnymalNavPerceptionAdapter):
    def __init__(self, mutate=None):
        super().__init__(env, env_index=0)
        self._mutate = mutate

    def get_world_state(self):
        s = super().get_world_state()
        return self._mutate(s) if self._mutate else s


def ahead(state, dx=3.0):
    b = state.robot.end_effector_pose.position
    return (b[0] + dx, b[1], b[2])


def run(label, kind, expect, perception=None, extra=None):
    perception = perception or Wrap()
    gate = ActuatorGate(perception, IsaacLabAnymalNavDynamicsAdapter(max_base_speed_mps=1.0, body_radius_m=0.35),
                        FreezeInPlaceFallback(), InMemoryLogger(), ActionSchemaRegistry.from_yaml(CFG, expected_digest=PIN))
    st = IsaacLabAnymalNavPerceptionAdapter(env, env_index=0).get_world_state()
    a = Action("navigate", {"target_position": ahead(st), **(extra or {})})
    gate.gate(a)  # warm-up
    d = gate.gate(a)
    fired = [r.name for r in d.precondition_results if not r.satisfied]
    ok = (d.verdict.value == "permit") if expect == "permit" else (d.verdict.value == "block" and expect in fired)
    RESULTS.append({"label": label, "kind": kind, "expect": expect, "verdict": d.verdict.value, "fired": fired, "pass": ok})
    print(f"[{'PASS' if ok else 'FAIL'}] {label}  ({kind})\n        expect={expect} verdict={d.verdict.value} fired={fired}")


def agent_at(offset, speed=1.5):
    def m(s):
        b = s.robot.end_effector_pose.position
        a = TrackedAgent("person", Pose(tuple(p + o for p, o in zip(b, offset))), tracking_confidence=0.95,
                         worst_case_speed_mps=speed, category=AgentCategory.ADULT, stature_m=1.75)
        return dataclasses.replace(s, agents=(a,))
    return m


fresh()
st = Wrap().get_world_state()
print(f"        real support polygon {st.robot.support_polygon}\n        CoM {st.robot.center_of_mass} CoM vel {st.robot.center_of_mass_velocity}")
run("nominal: clear path, real standing geometry", "nominal", "permit")
for chk in ("swept_path_clear_of_agents", "iso15066_separation_distance_maintained", "reduced_speed_near_human"):
    fresh()
    run(f"human 1.5m ahead on the path ({chk})", "injected-agent", chk, perception=Wrap(agent_at((1.5, 0.0, 0.0))))
fresh()
run("control: human 8m off to the side", "nominal", "permit", perception=Wrap(agent_at((0.5, 8.0, 0.0))))


def shifted_feet(s):
    bx, by = s.robot.center_of_mass[:2]
    poly = tuple((bx + 0.6 + dx, by + dy) for dx, dy in ((0.0, 0.2), (0.0, -0.2), (0.3, 0.2), (0.3, -0.2)))
    return dataclasses.replace(s, robot=dataclasses.replace(s.robot, support_polygon=poly))


def com_moving(v):
    return lambda s: dataclasses.replace(s, robot=dataclasses.replace(s.robot, center_of_mass_velocity=v))


fresh()
run("feet slipped clear of the CoM (static margin)", "injected", "stability_margin_maintained", perception=Wrap(shifted_feet))
fresh()
run("real feet, CoM moving 1.5 m/s toward the front edge (capture point)", "injected", "stability_margin_maintained",
    perception=Wrap(com_moving((1.5, 0.0, 0.0))))

fresh()
st = Wrap().get_world_state()
lims = st.robot.joint_position_limits
RESULTS.append({"label": "adapter reports ANYmal-C joint limits as unreported (asset defines none)", "kind": "adapter",
                "expect": "None", "verdict": repr(lims), "fired": [], "pass": lims is None})
print(f"[{'PASS' if lims is None else 'FAIL'}] ANYmal-C joint_position_limits reported as {lims!r}")

fresh()
run("navigate commanded at 3.0 m/s (rated 1.0)", "commanded", "cartesian_speed_within_limits", extra={"commanded_speed_mps": 3.0})
fresh()
run("control: navigate commanded at 0.8 m/s", "nominal", "permit", extra={"commanded_speed_mps": 0.8})

n_ok = sum(r["pass"] for r in RESULTS)
print(f"\nLIVE ANYMAL: {n_ok}/{len(RESULTS)} scenarios behaved as expected")
json.dump(RESULTS, open(args_cli.out, "w"), indent=1)
simulation_app.close()
