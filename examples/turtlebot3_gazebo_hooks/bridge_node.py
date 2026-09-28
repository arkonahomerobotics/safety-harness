"""SafetyHarnessBridgeNode: turns a real, running TurtleBot3 Gazebo simulation into the ``bridge``
object safety_harness/adapters/ros2.py's ROS2PerceptionAdapter reads -- see that module's docstring
for the exact duck-typed contract.

Different from examples/ros2_hooks/bridge_node.py in exactly one way: a differential-drive mobile
base has no single "end effector" and no gripper-joint URDF to parse for limits, so its pose comes
from the real /odom topic (nav_msgs/Odometry, published by the Gazebo diff-drive plugin) instead of
a manipulator's /ee_pose + /robot_description, following the same "base pose reported through
end_effector_pose, since no less manipulator-specific field exists" convention already used for the
ANYmal-C quadruped (safety_harness/adapters/isaac_lab_anymal.py). Reuses the exact same JSON bridge
topics as ros2_hooks for this project's own tracked-object/agent fields.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

MAX_LINEAR_SPEED_MPS = 0.22  # TurtleBot3 Burger datasheet max linear speed


class SafetyHarnessBridgeNode(Node):
    def __init__(self, observed_regions=None):
        super().__init__("safety_harness_bridge")
        self.joint_state = None  # this bridge reports none -- see the module docstring
        self.ee_pose = None  # populated from /odom, not a manipulator pose topic -- see _on_odom
        self.joint_position_limits = None
        self.joint_velocity_limits = None
        self.joint_effort_limits = None
        self.max_cartesian_speed_mps = MAX_LINEAR_SPEED_MPS
        self.rated_payload_kg = None
        self.tracked_objects: list = []
        self.tracked_agents: list = []
        self.observed_regions = observed_regions
        self.visibility_confidence = 1.0
        self._last_odom_stamp = None

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(String, "/safety_harness/tracked_objects", self._on_tracked_objects, 10)
        self.create_subscription(String, "/safety_harness/tracked_agents", self._on_tracked_agents, 10)
        self.decision_pub = self.create_publisher(String, "/safety_harness/decisions", 10)

    def _on_odom(self, msg: Odometry):
        # A wheeled base has no joints in the manipulator sense -- an empty tuple is the honest
        # report ("no joints of interest"), not a fabricated one.
        #
        # Stamp with THIS node's own wall-clock receipt time (self.get_clock().now()), not
        # msg.header.stamp -- found live, not assumed: TurtleBot3's Gazebo odometry plugin stamps
        # with SIMULATION time (starts near zero when gzserver launches), not wall-clock time, even
        # though this node itself runs on the system clock (use_sim_time was never set on it). Using
        # the sim-time stamp directly made sensor_data_fresh compare "12 seconds after the Unix
        # epoch" against a real 2026 timestamp -- always look catastrophically stale. Receipt time
        # is an honest proxy for capture time here: real ROS 2 delivery over loopback DDS is on the
        # order of milliseconds, not a meaningful source of staleness for a 0.5s freshness budget.
        stamp = self.get_clock().now().to_msg()
        header = SimpleNamespace(stamp=stamp, frame_id=msg.header.frame_id)
        self.joint_state = SimpleNamespace(name=[], position=[], velocity=[], header=header)
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        self.ee_pose = SimpleNamespace(pose=SimpleNamespace(position=p, orientation=o), header=header)

    def _on_tracked_objects(self, msg):
        self.tracked_objects = [SimpleNamespace(**o) for o in json.loads(msg.data)]

    def _on_tracked_agents(self, msg):
        self.tracked_agents = [SimpleNamespace(**a) for a in json.loads(msg.data)]

    def publish_decision(self, decision, action):
        payload = {
            "action_type": action.action_type,
            "verdict": decision.verdict.value,
            "failing_checks": [r.name for r in decision.precondition_results if not r.satisfied],
        }
        self.decision_pub.publish(String(data=json.dumps(payload)))

    def ready(self) -> bool:
        return self.joint_state is not None
