"""Runnable, self-checking demonstration: a real ActuatorGate gating a real `navigate` action for a
real TurtleBot3, physically simulated in a real headless Gazebo (Classic) -- not a mock publisher
(examples/ros2_hooks), a real physics-simulated robot with real /odom.

    python3 gate_demo.py
    ros2 topic echo /safety_harness/decisions      # from another shell, while this runs

Scene: the robot's real pose comes from Gazebo's own diff-drive plugin (real physics, real /odom).
The hazard (a person near the robot) is injected the same way every other demo in this project
injects a hazard that isn't itself physically simulated -- published on this project's own
/safety_harness/tracked_agents bridge topic, not represented as a real Gazebo model. Two scenarios:
- agent far away (5m) -> PERMIT
- agent close to the robot's proposed path (0.5m) -> BLOCK on swept_path_clear_of_agents /
  iso15066_separation_distance_maintained
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import rclpy
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from safety_harness import ActionSchemaRegistry, ActuatorGate, DecisionVerdict, DecisionWatchdog  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.ros2 import ROS2DynamicsAdapter, ROS2PerceptionAdapter  # noqa: E402
from safety_harness.integrity import read_digest_file  # noqa: E402
from safety_harness.schema import Action  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_node import SafetyHarnessBridgeNode  # noqa: E402

HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = os.path.join(HARNESS_ROOT, "configs", "turtlebot3_action_schema.yaml")
PIN = read_digest_file(CFG + ".sha256")


def launch_gazebo_headless():
    """gzserver only -- no gzclient (no display needed). A real subprocess, not a mock: the exact
    same gazebo_ros_pkgs launch machinery `ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py`
    uses, minus the gzclient GUI action that launch file also starts."""
    world = subprocess.check_output(
        ["ros2", "pkg", "prefix", "turtlebot3_gazebo"], text=True
    ).strip() + "/share/turtlebot3_gazebo/worlds/turtlebot3_world.world"
    gzserver = subprocess.Popen(
        ["ros2", "launch", "gazebo_ros", "gzserver.launch.py", f"world:={world}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    spawn = None
    return gzserver, world


def wait_for_spawn_service(timeout_s: float = 90.0) -> bool:
    """gzserver's ROS factory plugin (which provides /spawn_entity) can take well over the
    spawn_entity.py script's own hardcoded 30s wait under emulation -- confirmed live: gzserver
    itself starts and consumes real CPU well before the factory service is actually ready. Poll for
    the real service instead of guessing a fixed sleep duration that may or may not be enough."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            out = subprocess.run(["ros2", "service", "list"], capture_output=True, text=True, timeout=10).stdout
            if "/spawn_entity" in out:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2)
    return False


def spawn_robot():
    return subprocess.Popen(
        ["ros2", "launch", "turtlebot3_gazebo", "spawn_turtlebot3.launch.py"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def robot_state_publisher():
    return subprocess.Popen(
        ["ros2", "launch", "turtlebot3_gazebo", "robot_state_publisher.launch.py"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def navigate(target_xy):
    return Action("navigate", {"target_position": (target_xy[0], target_xy[1], 0.0), "commanded_speed_mps": 0.15})


def publish_agent(node, pub, x, y, stamp_s):
    agents = [{"agent_id": "person_1", "position": [x, y, 0.0], "tracking_confidence": 0.95,
               "category": "adult", "time_since_confirmed_s": 0.0, "worst_case_speed_mps": 1.5, "stamp": stamp_s}]
    pub.publish(String(data=json.dumps(agents)))


def main() -> int:
    procs = []
    try:
        gzserver, world = launch_gazebo_headless()
        procs.append(gzserver)
        print(f"gzserver starting with world {world} ...")
        if not wait_for_spawn_service():
            print("FAIL: /spawn_entity service never became available -- gzserver's ROS factory plugin didn't finish initializing")
            return 1
        print("gzserver's /spawn_entity service is live")
        spawn = subprocess.run(
            ["ros2", "launch", "turtlebot3_gazebo", "spawn_turtlebot3.launch.py"],
            capture_output=True, text=True, timeout=40,
        )
        if "Successfully spawned" not in spawn.stdout + spawn.stderr:
            print(f"FAIL: spawn_turtlebot3 did not report success:\n{spawn.stdout}\n{spawn.stderr}")
            return 1
        print("Robot spawned")
        procs.append(robot_state_publisher())
        time.sleep(3)  # real controller/odometry-plugin bring-up after spawn

        rclpy.init()
        bridge = SafetyHarnessBridgeNode(observed_regions=(((-5.0, -5.0, -1.0), (5.0, 5.0, 3.0)),))
        agents_pub = bridge.create_publisher(String, "/safety_harness/tracked_agents", 10)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(bridge)

        deadline = time.monotonic() + 30.0
        while not bridge.ready() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.2)
        if not bridge.ready():
            print("FAIL: bridge never received real /odom from the running Gazebo simulation")
            return 1
        print(f"Bridge is live: real pose = {bridge.ee_pose.pose.position}")

        schema = ActionSchemaRegistry.from_yaml(CFG, expected_digest=PIN)
        watchdog = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=0.5)
        gate = ActuatorGate(
            perception=ROS2PerceptionAdapter(bridge),
            dynamics=ROS2DynamicsAdapter(max_ee_speed_mps=0.22),
            fallback=FreezeInPlaceFallback(),
            logger=InMemoryLogger(),
            action_schema=schema,
            watchdog=watchdog,
        )

        p0 = bridge.ee_pose.pose.position
        target = (p0.x + 1.0, p0.y)
        cases = [("far agent (5m away)", p0.x + 5.0, p0.y + 5.0, DecisionVerdict.PERMIT),
                 ("close agent (0.5m from the path)", p0.x + 0.5, p0.y, DecisionVerdict.BLOCK)]

        all_ok = True
        for label, ax, ay, expected in cases:
            publish_agent(bridge, agents_pub, ax, ay, time.time())
            for _ in range(5):
                executor.spin_once(timeout_sec=0.1)
            action = navigate(target)
            decision = gate.gate(action)
            bridge.publish_decision(decision, action)
            executor.spin_once(timeout_sec=0.1)
            failing = [r.name for r in decision.precondition_results if not r.satisfied]
            ok = decision.verdict == expected
            all_ok &= ok
            print(f"{'PASS' if ok else 'FAIL'}: navigate with {label} -> {decision.verdict.value} "
                  f"(expected {expected.value}){' failing=' + str(failing) if failing else ''}")

        bridge.destroy_node()
        rclpy.shutdown()
        print("PASS: gate_demo" if all_ok else "FAIL: gate_demo")
        return 0 if all_ok else 1
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    sys.exit(main())
