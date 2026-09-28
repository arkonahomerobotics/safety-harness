"""Runnable, self-checking demonstration: a real ActuatorGate, gating real proposed actions, fed by
a real ROS 2 graph -- two rclpy nodes (MockRobotNode, SafetyHarnessBridgeNode) actually publishing
and subscribing over real DDS, in this one process. No Isaac Lab, no GPU, no physical robot.

    python3 gate_demo.py            # prints each decision, exits 0 on the expected PASS, 1 otherwise
    ros2 topic echo /safety_harness/decisions      # from another shell, while this runs: watch it live

Expected result, and why (configs/ros2_action_schema.yaml, see mock_robot.py's own docstring):
- grasp block_a (5kg) -> BLOCK on mass_within_force_budget (over the 3kg budget) and
  payload_and_grip_force_within_limits (the fixed 14N grip force can't hold 5kg either).
- grasp block_b (0.4kg) -> PERMIT: within budget, 14N sits inside the friction-hold/fragile-cap
  window for 0.4kg (13.1N needed, 15N fragile cap), every other check reads real, fresh, in-range data.
"""

from __future__ import annotations

import os
import sys
import time

import rclpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from safety_harness import ActionSchemaRegistry, ActuatorGate, DecisionVerdict, DecisionWatchdog  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.ros2 import ROS2DynamicsAdapter, ROS2PerceptionAdapter  # noqa: E402
from safety_harness.integrity import read_digest_file  # noqa: E402
from safety_harness.schema import Action  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_node import SafetyHarnessBridgeNode  # noqa: E402
from mock_robot import MockRobotNode  # noqa: E402

HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = os.path.join(HARNESS_ROOT, "configs", "ros2_action_schema.yaml")
PIN = read_digest_file(CFG + ".sha256")
GRIP_FORCE_N = 14.0  # fixed, project-wide convention -- see other examples' GRIP_FORCE_N


def grasp(object_id: str, position) -> Action:
    return Action("grasp", {"object_id": object_id, "target_position": tuple(position),
                             "grip_force_n": GRIP_FORCE_N, "commanded_speed_mps": 0.2})


def main() -> int:
    rclpy.init()
    mock = MockRobotNode()
    bridge = SafetyHarnessBridgeNode(max_cartesian_speed_mps=1.5, rated_payload_kg=3.0,
                                      joint_limit_exempt=("gripper",),
                                      observed_regions=(((-2.0, -2.0, -1.0), (2.0, 2.0, 3.0)),))

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(mock)
    executor.add_node(bridge)

    deadline = time.monotonic() + 10.0
    while not bridge.ready() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
    if not bridge.ready():
        print("FAIL: bridge never received a full joint_state + robot_description over real ROS 2 topics")
        return 1

    schema = ActionSchemaRegistry.from_yaml(CFG, expected_digest=PIN)
    watchdog = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=0.5)
    gate = ActuatorGate(
        perception=ROS2PerceptionAdapter(bridge, joint_limit_exempt=("gripper",)),
        dynamics=ROS2DynamicsAdapter(max_ee_speed_mps=0.5),
        fallback=FreezeInPlaceFallback(),
        logger=InMemoryLogger(),
        action_schema=schema,
        watchdog=watchdog,
    )

    cases = [("block_a", (0.5, 0.0, 0.05), DecisionVerdict.BLOCK), ("block_b", (0.6, 0.1, 0.05), DecisionVerdict.PERMIT)]
    all_ok = True
    for object_id, position, expected in cases:
        executor.spin_once(timeout_sec=0.1)  # let a fresh joint_states/objects cycle land first
        action = grasp(object_id, position)
        decision = gate.gate(action)
        bridge.publish_decision(decision, action)
        executor.spin_once(timeout_sec=0.05)  # flush the publish
        failing = [r.name for r in decision.precondition_results if not r.satisfied]
        ok = decision.verdict == expected
        all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'}: grasp {object_id} -> {decision.verdict.value} "
              f"(expected {expected.value}){' failing=' + str(failing) if failing else ''}")

    bridge.destroy_node()
    mock.destroy_node()
    rclpy.shutdown()
    print("PASS: gate_demo" if all_ok else "FAIL: gate_demo")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
