"""A real rclpy node standing in for a robot's own driver stack: publishes real sensor_msgs/
JointState, geometry_msgs/PoseStamped, and /robot_description on a timer, plus two tracked objects
(one over the 3kg force budget, one not) and one far-away tracked agent on this project's own
minimal JSON bridge topics. Nothing here is safety_harness-specific except the JSON shape of the
last two -- everything else is exactly how a real robot's driver publishes this data.

Scene, deliberately mirroring the Franka closed-loop demo's own heavy-block scenario:
- `block_a`: 5kg -- over the 3kg force budget in configs/ros2_action_schema.yaml. Expected: BLOCK.
- `block_b`: 0.4kg -- ordinary, liftable. Expected: PERMIT.
"""

from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

MOCK_URDF = """<?xml version="1.0"?>
<robot name="mock_arm">
  <link name="base_link"/>
  <link name="ee_link"/>
  <joint name="shoulder_pan" type="revolute">
    <parent link="base_link"/><child link="ee_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9" upper="2.9" velocity="2.0" effort="50.0"/>
  </joint>
  <link name="gripper_link"/>
  <joint name="gripper" type="prismatic">
    <parent link="ee_link"/><child link="gripper_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="0.04" velocity="1.0" effort="20.0"/>
  </joint>
</robot>"""

JOINT_NAMES = ["shoulder_pan", "gripper"]


class MockRobotNode(Node):
    def __init__(self):
        super().__init__("mock_robot")
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.ee_pub = self.create_publisher(PoseStamped, "/ee_pose", 10)
        self.urdf_pub = self.create_publisher(String, "/robot_description", 10)
        self.objects_pub = self.create_publisher(String, "/safety_harness/tracked_objects", 10)
        self.agents_pub = self.create_publisher(String, "/safety_harness/tracked_agents", 10)
        self.create_timer(0.05, self._tick)  # 20 Hz, matching this project's other demo scripts

    def _tick(self):
        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.name = list(JOINT_NAMES)
        js.position = [0.0, 0.02]
        js.velocity = [0.0, 0.0]
        js.effort = [0.5, 0.1]
        self.joint_pub.publish(js)

        ee = PoseStamped()
        ee.header.stamp = now
        ee.pose.position.x, ee.pose.position.y, ee.pose.position.z = 0.5, 0.0, 0.3
        ee.pose.orientation.w = 1.0
        self.ee_pub.publish(ee)

        self.urdf_pub.publish(String(data=MOCK_URDF))

        stamp_s = now.sec + now.nanosec * 1e-9
        objects = [
            {"object_id": "block_a", "object_class": "cube", "position": [0.5, 0.0, 0.05], "mass_kg": 5.0,
             "hazard_tags": ["fragile"], "pose_confidence": 1.0, "class_confidence": 1.0,
             "cleared_for_interaction": True, "supported_stably": True, "fall_consequence": "none", "stamp": stamp_s},
            {"object_id": "block_b", "object_class": "cube", "position": [0.6, 0.1, 0.05], "mass_kg": 0.4,
             "hazard_tags": ["fragile"], "pose_confidence": 1.0, "class_confidence": 1.0,
             "cleared_for_interaction": True, "supported_stably": True, "fall_consequence": "none", "stamp": stamp_s},
        ]
        self.objects_pub.publish(String(data=json.dumps(objects)))

        agents = [
            {"agent_id": "person_1", "position": [5.0, 5.0, 0.0], "tracking_confidence": 0.95,
             "category": "adult", "time_since_confirmed_s": 0.0, "worst_case_speed_mps": 1.5, "stamp": stamp_s},
        ]
        self.agents_pub.publish(String(data=json.dumps(agents)))


def main():
    rclpy.init()
    node = MockRobotNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
