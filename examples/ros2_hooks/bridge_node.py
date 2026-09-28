"""SafetyHarnessBridgeNode: a real rclpy node that turns a real ROS 2 graph into the ``bridge``
object safety_harness/adapters/ros2.py's ROS2PerceptionAdapter reads -- see that module's docstring
for the exact duck-typed contract this node fills in.

This is the only place in this example that reads real message wire formats: sensor_msgs/JointState,
geometry_msgs/PoseStamped, std_msgs/String (JSON) for this project's own tracked-object/agent fields,
and /robot_description (std_msgs/String, the URDF XML) for joint position/velocity/effort limits --
parsed with the standard library's xml.etree, no extra URDF-parsing dependency.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


def _parse_urdf_limits(urdf_xml: str, joint_order: list):
    """Returns (position_limits, velocity_limits, effort_limits), each a tuple aligned to
    joint_order, or (None, None, None) if the URDF has no joints matching that order at all. A
    joint present in the URDF but with no <limit> tag (e.g. a continuous joint) gets None entries in
    every list -- default-deny for that joint, not a fabricated infinite limit."""
    root = ET.fromstring(urdf_xml)
    limits_by_name = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if limit is None:
            limits_by_name[name] = (None, None, None)
            continue
        lo = limit.get("lower")
        hi = limit.get("upper")
        vel = limit.get("velocity")
        eff = limit.get("effort")
        pos = (float(lo), float(hi)) if lo is not None and hi is not None else None
        limits_by_name[name] = (pos, float(vel) if vel is not None else None, float(eff) if eff is not None else None)
    if not any(n in limits_by_name for n in joint_order):
        return None, None, None
    pos_limits = tuple(limits_by_name.get(n, (None, None, None))[0] for n in joint_order)
    vel_limits = tuple(limits_by_name.get(n, (None, None, None))[1] for n in joint_order)
    eff_limits = tuple(limits_by_name.get(n, (None, None, None))[2] for n in joint_order)
    return pos_limits, vel_limits, eff_limits


class SafetyHarnessBridgeNode(Node):
    """Subscribes to a real robot's standard topics plus this project's own minimal JSON bridge
    topics, and exposes exactly the attributes ROS2PerceptionAdapter's docstring documents. Nothing
    here is safety_harness-import-time-required -- see that module's own docstring on why it never
    imports rclpy itself; this node is the glue that makes the duck-typed contract real."""

    def __init__(self, max_cartesian_speed_mps=None, rated_payload_kg=None, joint_limit_exempt=(), observed_regions=None):
        super().__init__("safety_harness_bridge")
        self.joint_state = None
        self.ee_pose = None
        self.joint_position_limits = None
        self.joint_velocity_limits = None
        self.joint_effort_limits = None
        self.tracked_objects: list = []
        self.tracked_agents: list = []
        # Static per-deployment config, same as max_cartesian_speed_mps below -- not a topic, because
        # real occlusion-aware coverage tracking is its own perception subsystem, out of scope for
        # this minimal bridge. A generous fixed box here is the same honest simplification the Isaac
        # Lab adapters make for their own PRIVILEGED_STATE_OBSERVED_REGION -- see that module.
        self.observed_regions = observed_regions
        self.visibility_confidence = 1.0
        self.max_cartesian_speed_mps = max_cartesian_speed_mps
        self.rated_payload_kg = rated_payload_kg
        self._joint_limit_exempt = joint_limit_exempt
        self._urdf_parsed = False

        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(PoseStamped, "/ee_pose", self._on_ee_pose, 10)
        self.create_subscription(String, "/robot_description", self._on_robot_description, 10)
        self.create_subscription(String, "/safety_harness/tracked_objects", self._on_tracked_objects, 10)
        self.create_subscription(String, "/safety_harness/tracked_agents", self._on_tracked_agents, 10)

        self.decision_pub = self.create_publisher(String, "/safety_harness/decisions", 10)

    def _on_joint_state(self, msg):
        self.joint_state = msg

    def _on_ee_pose(self, msg):
        self.ee_pose = msg

    def _on_robot_description(self, msg):
        if self._urdf_parsed or self.joint_state is None:
            return
        pos, vel, eff = _parse_urdf_limits(msg.data, list(self.joint_state.name))
        if pos is None:
            return
        if self._joint_limit_exempt:
            pos = tuple(None if any(tag in name for tag in self._joint_limit_exempt) else lim
                        for name, lim in zip(self.joint_state.name, pos))
        self.joint_position_limits, self.joint_velocity_limits, self.joint_effort_limits = pos, vel, eff
        self._urdf_parsed = True

    def _on_tracked_objects(self, msg):
        self.tracked_objects = [SimpleNamespace(**o) for o in json.loads(msg.data)]

    def _on_tracked_agents(self, msg):
        self.tracked_agents = [SimpleNamespace(**a) for a in json.loads(msg.data)]

    def publish_decision(self, decision, action):
        """A real ROS 2 publish, not just an in-process log -- `ros2 topic echo
        /safety_harness/decisions` shows this live from outside the process."""
        payload = {
            "action_type": action.action_type,
            "verdict": decision.verdict.value,
            "failing_checks": [r.name for r in decision.precondition_results if not r.satisfied],
        }
        self.decision_pub.publish(String(data=json.dumps(payload)))

    def ready(self) -> bool:
        return self.joint_state is not None and self._urdf_parsed
