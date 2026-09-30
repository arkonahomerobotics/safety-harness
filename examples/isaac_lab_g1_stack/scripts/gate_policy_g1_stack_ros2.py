"""Same trained G1 block-stacking policy as gate_policy_g1_stack.py, gated the same way -- but every
robot/object reading the harness sees comes off real ROS 2 wire messages (sensor_msgs/JointState,
geometry_msgs/PoseStamped, std_msgs/String JSON for tracked objects), published and consumed via
real rclpy pub/sub inside the same process, instead of the direct in-process PerceptionAdapter
gate_policy_g1_stack.py uses.

This is the second half of the "does isaacsim.ros2.bridge work against a real Isaac Lab robot env"
question: sub-part 1 (does the extension load at all on this Isaac Lab 3.0.0 build) was answered by
a standalone smoke test (see the isaac-ros2-bridge-status memory) -- it does, once ROS_DISTRO /
RMW_IMPLEMENTATION / LD_LIBRARY_PATH are set to point at the bundled internal libs (a real, otherwise
undocumented env-var bug: the extension auto-detects "jazzy" from this container's Ubuntu 24.04, and
that bundled distro's RMW lib fails to dlopen one of its own transitive deps unless the lib dir is on
LD_LIBRARY_PATH). This script is sub-part 2: proving ROS2PerceptionAdapter/ROS2DynamicsAdapter
(safety_harness/adapters/ros2.py) can be driven by a real GPU-simulated humanoid, not just the
hand-written mock publisher in examples/ros2_hooks/ or a non-Isaac real sim (TurtleBot3+Gazebo).

Reuses SafetyHarnessBridgeNode from examples/ros2_hooks/ as-is (added to sys.path below) rather than
reimplementing the same subscriber glue -- it already implements exactly the duck-typed contract
ROS2PerceptionAdapter's docstring documents, and reusing it means this test genuinely exercises the
same bridge code the TurtleBot3/ros2_hooks examples already verified against a different robot.

Scope decision, made explicitly rather than silently: joint position/velocity/effort limits are set
directly from the same real per-joint sim data gate_policy_g1_stack.py uses
(_real_joint_position_limits), NOT round-tripped through SafetyHarnessBridgeNode's /robot_description
URDF-parsing path. That path is real code and already exercised by examples/ros2_hooks/ and
examples/turtlebot3_gazebo_hooks/ against their own robots' real URDFs; re-proving URDF parsing here
would add wall-clock GPU cost without testing anything new. What IS new here, and what this script
actually verifies: a real Isaac Lab humanoid's joint state and end-effector pose, serialized through
real sensor_msgs/geometry_msgs messages and deserialized back through a real rclpy subscription, then
gated -- not privileged in-process Python objects.
"""

import argparse
import json
import math
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--harness", type=str, default="/workspace/safety_harness")
parser.add_argument("--out", type=str, default="")
parser.add_argument("--max_steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

sys.path.insert(0, args_cli.harness)
sys.path.insert(0, f"{args_cli.harness}/examples/ros2_hooks")
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from safety_harness.action_schema import ActionSchemaRegistry  # noqa: E402
from safety_harness.adapters._isaac_lab_common import joint_position_limits as _real_joint_position_limits  # noqa: E402
from safety_harness.adapters.ros2 import ROS2DynamicsAdapter, ROS2PerceptionAdapter  # noqa: E402
from safety_harness.adapters.simple import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.engine import ActuatorGate, PerceptionFailure  # noqa: E402
from safety_harness.integrity import read_digest_file, verify_decision_action  # noqa: E402
from safety_harness.schema import Action, DecisionVerdict  # noqa: E402

BLOCK = 0.045
POS_SCALE = 0.02
HORIZON_STEPS = 25
GRIP_FORCE_N = 10.0
RATED_PAYLOAD_KG = 2.0
G1_JOINT_LIMIT_EXEMPT = ("_hand_",)
MAX_CARTESIAN_SPEED_MPS = 1.5
WORKSPACE = ((-1.0, -0.2, 0.0), (0.8, 1.2, 1.6))


def xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


cfg_path = f"{args_cli.harness}/configs/g1_action_schema.yaml"
registry = ActionSchemaRegistry.from_yaml(cfg_path, expected_digest=read_digest_file(cfg_path + ".sha256"))


def load_actor(path, device):
    import torch.nn as nn

    sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    ks = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    layers = []
    for j, k in enumerate(ks):
        w, b = sd[f"mlp.{k}.weight"], sd[f"mlp.{k}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0]).to(device)
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if j < len(ks) - 1:
            layers.append(nn.ELU())
    mlp = nn.Sequential(*layers).eval()
    mean, std = sd["obs_normalizer._mean"].to(device), sd["obs_normalizer._std"].to(device)
    return lambda o: mlp((o - mean) / (std + 1e-2))


env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env_cfg.seed = args_cli.seed
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

# The ROS 2 bridge extension is enabled here, AFTER env creation, not before -- a real bug found
# running this script: enabling it earlier (between AppLauncher and gym.make(), following
# NVIDIA's own standalone_examples/api/isaacsim.ros2.bridge/clock.py ordering) makes
# ManagerBasedEnv's own seeding step (env_cfg.seed -> isaaclab's env.seed() ->
# omni.replicator.core's rep.set_global_seed()) fail with
# "OmniGraphError: Failed to wrap graph in node given {'graph_path': '/Replicator/SDGPipeline', ...}"
# -- enabling isaacsim.ros2.bridge pulls in enough extra OmniGraph/extension state that Replicator's
# own SDGPipeline graph creation breaks. gate_policy_g1_stack.py never hits this because it never
# enables the ROS 2 bridge at all. Deferring bridge setup until after the env (and its one-time
# seeding) already exists avoids the conflict entirely, and nothing about this test needs the bridge
# active any earlier -- publishing/subscribing only happens inside the step loop below.
app_utils.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
import rclpy  # noqa: E402
from bridge_node import SafetyHarnessBridgeNode  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402

rclpy.init()
publisher_node = rclpy.create_node("g1_state_publisher")
js_pub = publisher_node.create_publisher(JointState, "/joint_states", 10)
pose_pub = publisher_node.create_publisher(PoseStamped, "/ee_pose", 10)
objects_pub = publisher_node.create_publisher(String, "/safety_harness/tracked_objects", 10)

bridge = SafetyHarnessBridgeNode(
    max_cartesian_speed_mps=MAX_CARTESIAN_SPEED_MPS,
    rated_payload_kg=RATED_PAYLOAD_KG,
    joint_limit_exempt=G1_JOINT_LIMIT_EXEMPT,
    observed_regions=(WORKSPACE,),
)

perception = ROS2PerceptionAdapter(bridge, joint_limit_exempt=G1_JOINT_LIMIT_EXEMPT)
dynamics = ROS2DynamicsAdapter(max_ee_speed_mps=MAX_CARTESIAN_SPEED_MPS)
logger = InMemoryLogger()
gate = ActuatorGate(perception, dynamics, FreezeInPlaceFallback(), logger, registry, horizon_s=0.5)

stats = {"mode": "ros2_bridge", "checkpoint": args_cli.checkpoint, "decisions": {}, "blocked_by": {}}

with torch.inference_mode():
    obs, _ = env.reset()
    env.episode_length_buf[:] = 0
    dev = env.device
    policy = load_actor(args_cli.checkpoint, dev)
    robot, ba, bb = env.scene["robot"], env.scene["object"], env.scene["block_b"]
    o = env.scene.env_origins
    widx = robot.data.body_names.index("left_wrist_yaw_link")
    hand_ids, _ = robot.find_joints(["left_hand_index_0_joint"])
    joint_names = list(robot.data.joint_names)
    pos_limits = _real_joint_position_limits(robot, 0, G1_JOINT_LIMIT_EXEMPT)
    # Scope decision documented in the module docstring: set directly from real sim data rather
    # than round-tripped through the bridge's /robot_description URDF-parsing path.
    bridge.joint_position_limits = pos_limits
    mass_a = float(ba.root_physx_view.get_masses().numpy().reshape(1, -1)[0, 0])
    mass_b = float(bb.root_physx_view.get_masses().numpy().reshape(1, -1)[0, 0])
    lim = robot.data.soft_joint_pos_limits.torch[0, hand_ids[0]].tolist()
    rest_z = float((ba.data.root_pos_w.torch - o)[0, 2])

    held_grip = -1.0
    prev_cmd = -1.0
    step_ts = {"waited_for_messages": 0}

    for t in range(args_cli.max_steps):
        act = policy(obs["policy"]).clamp(-1, 1)
        w = robot.data.body_pos_w.torch[:, widx] - o
        pa = ba.data.root_pos_w.torch - o
        pb = bb.data.root_pos_w.torch - o
        rq = robot.data.root_quat_w.torch
        reach_tgt = (w + quat_apply(rq, act[:, :3] * POS_SCALE) * HORIZON_STEPS)[0].tolist()

        # --- Publish real robot/object state as real ROS 2 messages ---
        now = publisher_node.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = now
        js.name = joint_names
        js.position = robot.data.joint_pos.torch[0].tolist()
        js.velocity = robot.data.joint_vel.torch[0].tolist()
        js_pub.publish(js)

        ee_pose = PoseStamped()
        ee_pose.header.stamp = now
        wq = xyzw_to_wxyz(robot.data.body_quat_w.torch[:, widx][0].tolist())
        ee_pose.pose.position.x, ee_pose.pose.position.y, ee_pose.pose.position.z = w[0].tolist()
        ee_pose.pose.orientation.w, ee_pose.pose.orientation.x, ee_pose.pose.orientation.y, ee_pose.pose.orientation.z = wq
        pose_pub.publish(ee_pose)

        grip_open = 1.0 - max(0.0, min(1.0, (robot.data.joint_pos.torch[0, hand_ids[0]].item() - lim[0]) / (lim[1] - lim[0])))
        stamp_s = now.sec + now.nanosec * 1e-9
        qa = ba.data.root_quat_w.torch[0].tolist()
        qb = bb.data.root_quat_w.torch[0].tolist()
        va = ba.data.root_lin_vel_w.torch[0].tolist()
        vb = bb.data.root_lin_vel_w.torch[0].tolist()
        objects_pub.publish(String(data=json.dumps([
            {
                "object_id": "block_a", "object_class": "block", "stamp": stamp_s,
                "position": pa[0].tolist(), "velocity": va, "mass_kg": mass_a,
                "hazard_tags": [], "pose_confidence": 1.0, "class_confidence": 1.0,
                "cleared_for_interaction": True,
                "supported_stably": math.sqrt(sum(v * v for v in va)) < 0.05,
                "fall_consequence": "none",
            },
            {
                "object_id": "block_b", "object_class": "block", "stamp": stamp_s,
                "position": pb[0].tolist(), "velocity": vb, "mass_kg": mass_b,
                "hazard_tags": [], "pose_confidence": 1.0, "class_confidence": 1.0,
                "cleared_for_interaction": True,
                "supported_stably": math.sqrt(sum(v * v for v in vb)) < 0.05,
                "fall_consequence": "none",
            },
        ])))

        # --- Real rclpy round trip: spin both nodes until the bridge has this step's messages ---
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            rclpy.spin_once(publisher_node, timeout_sec=0.0)
            rclpy.spin_once(bridge, timeout_sec=0.0)
            if bridge.joint_state is not None and bridge.joint_state.header.stamp.sec == now.sec \
                    and bridge.joint_state.header.stamp.nanosec == now.nanosec:
                break
            time.sleep(0.005)
        else:
            step_ts["waited_for_messages"] += 1

        # --- Same grasp/place/reach action selection as gate_policy_g1_stack.py ---
        g = float(act[0, 6])
        d_wa = math.dist(pa[0].tolist(), w[0].tolist())
        holding = d_wa < 0.16 and pa[0, 2].item() > rest_z + 0.02
        if not holding and g > 0 >= prev_cmd and d_wa < 0.16:
            a = Action("grasp", {"object_id": "block_a", "target_position": tuple(pa[0].tolist()), "grip_force_n": GRIP_FORCE_N})
        elif holding and g < 0 <= prev_cmd:
            seat = (pb[0, 0].item(), pb[0, 1].item(), pb[0, 2].item() + BLOCK)
            a = Action("place", {"object_id": "block_a", "target_surface_id": "block_b", "target_position": seat})
        else:
            a = Action("reach", {"object_id": "block_a", "target_position": tuple(reach_tgt)})

        exe = act.clone()
        try:
            dec = gate.gate(a)
        except PerceptionFailure:
            dec = None
        ok = dec is not None and dec.verdict == DecisionVerdict.PERMIT and verify_decision_action(dec)
        verdict = "permit" if ok else "block"
        if not ok:
            exe[0, :6] = 0.0
            exe[0, 6] = held_grip
            if dec is not None:
                for r in dec.precondition_results:
                    if not r.satisfied:
                        stats["blocked_by"][r.name] = stats["blocked_by"].get(r.name, 0) + 1
            else:
                stats["blocked_by"]["perception_failure"] = stats["blocked_by"].get("perception_failure", 0) + 1
        else:
            bridge.publish_decision(dec, a)
        key = f"{a.action_type}:{verdict}"
        stats["decisions"][key] = stats["decisions"].get(key, 0) + 1

        held_grip = float(exe[0, 6])
        prev_cmd = held_grip
        obs, *_ = env.step(exe)

    pa = ba.data.root_pos_w.torch - o
    pb = bb.data.root_pos_w.torch - o
    w = robot.data.body_pos_w.torch[:, widx] - o
    stacked = (torch.linalg.norm((pa - pb)[0, :2]) < 0.015) and (abs(float(pa[0, 2] - pb[0, 2] - BLOCK)) < 0.012) \
        and (torch.linalg.norm(ba.data.root_vel_w.torch[0, :3]) < 0.03) and (torch.linalg.norm(pa[0] - w[0]) > 0.11)
    stats["stacked"] = bool(stacked)
    stats["steps_without_fresh_message"] = step_ts["waited_for_messages"]
    print("HARNESS_RESULT " + json.dumps(stats))
    if args_cli.out:
        json.dump(stats, open(args_cli.out, "w"), indent=1)

rclpy.shutdown()
simulation_app.close()
