# ROS 2 hooks: gating real actions on a real ROS 2 graph

`safety_harness/adapters/ros2.py` plus this example: a second reference integration pattern after
Isaac Lab, this time against a real `rclpy` node graph instead of a simulator -- real topics, real
message types, real DDS message passing, real timing. No Isaac Lab, no GPU, no physical robot.

## Why this exists

Feedback from a ROS Discourse moderator (a real OSRA TGC member) on an earlier, since-removed
`safety-harness` post: she's seen several similar "LLM-driven precondition-gating" submissions
recently, mostly vibe-coded and unvalidated, and suggested building ROS 2 hooks and testing against
a real ROS 2 system as a concrete first step before asking the community to evaluate the concept.
This is that step. It does not require real hardware to be meaningful -- ROS 2 itself is software; a
real `rclpy` node graph is fully testable against a simulated/mocked robot, which is exactly what
`mock_robot.py` is.

**What this proves:** the harness genuinely integrates into a real ROS 2 control loop -- real
`sensor_msgs/JointState`, real `geometry_msgs/PoseStamped`, real URDF-sourced joint limits, real
message timestamps flowing into `sensor_data_fresh`, gated through the actual `ActuatorGate` and
`DecisionWatchdog` this project uses everywhere else, with the resulting decision published back
onto a real topic (`ros2 topic echo /safety_harness/decisions` shows it live from outside the
process).

**What this does NOT prove:** this is not a substitute for real-hardware validation. ISO 13849-2
requires actual fault-condition testing on physical hardware for a certifiable Performance Level
claim -- see the design doc's [Development Roadmap](../../docs/design.md#development-roadmap)
(stages 6-7: real perception, then real hardware, both still ahead) and
[Certification packaging](../../docs/design.md#commercialization).
This proves the *software* correctly plugs into a real ROS 2 system; it says nothing about actuator
faults, sensor faults, or anything downstream of the topics it reads. Not wasted work either, though:
[The Construct's UR3e remote lab](https://www.theconstruct.ai/warehouse-robot-lab/) (found while
researching real-hardware access options) is explicitly ROS 2-native -- this adapter is directly
reusable the moment real-hardware access happens, not a one-off built just to answer a moderator.

## What's here

- **`safety_harness/adapters/ros2.py`** (in the core package): `ROS2PerceptionAdapter` and
  `ROS2DynamicsAdapter`. Like every other adapter in this project, it imports nothing
  ROS-2-specific -- no `rclpy` here, same as the Isaac Lab adapters import nothing from `isaaclab`.
  See its own docstring for the exact duck-typed `bridge` contract, spelled out field by field
  against the real ROS 2 message shapes it mirrors.
- **`bridge_node.py`**: the real `rclpy.node.Node` that subscribes to `/joint_states`, `/ee_pose`,
  `/robot_description` (parses the URDF's `<limit>` tags for position/velocity/effort limits with
  the standard library's `xml.etree`, no extra URDF-parsing dependency), and this project's own
  minimal JSON bridge topics (`/safety_harness/tracked_objects`, `/safety_harness/tracked_agents` --
  deliberately not a ROS standard message; see the adapter's own docstring for why no existing
  message type could carry these fields anyway). Publishes each decision to
  `/safety_harness/decisions`.
- **`mock_robot.py`**: a real `rclpy.node.Node` standing in for a robot's driver stack, publishing
  synthetic but realistic data at 20 Hz -- two tracked objects (`block_a`, 5 kg, over the 3 kg force
  budget; `block_b`, 0.4 kg, ordinary) and one tracked agent, far away.
- **`gate_demo.py`**: the actual runnable, self-checking demonstration. Spins both nodes in one
  process (still real message passing over real DDS -- multi-process isn't what makes ROS 2 real),
  waits for the bridge to receive a full `joint_state` + parsed URDF, then gates two `grasp`
  proposals and checks the verdict against what's expected. Exits 0 on pass, 1 on fail.
- **`configs/ros2_action_schema.yaml`** (repo root, alongside the other action schemas): mirrors
  `configs/example_action_schema.yaml`'s Franka `grasp`/`place`/`reach` one-for-one -- the ROS2
  adapter reports the same fields, so the same checks are all meaningful here.
- **`Dockerfile`**: `ros:humble-ros-base` + this package installed. This is exactly what was used to
  verify the demo actually passes (see Results below) -- reproducible, not "trust me it worked once."

## Run it

```bash
docker build -f examples/ros2_hooks/Dockerfile -t safety-harness-ros2-demo .
docker run --rm safety-harness-ros2-demo
```

## Results (2026-09-28, `ros:humble-ros-base`, verified in Docker on this machine)

```
PASS: grasp block_a -> block (expected block) failing=['mass_within_force_budget', 'payload_and_grip_force_within_limits']
PASS: grasp block_b -> permit (expected permit)
PASS: gate_demo
```

`block_a` (5 kg) correctly blocks on both the flat force budget and the fixed 14 N grip force being
nowhere near enough to hold it -- the same two-check pattern the Franka and G1 heavy-block demos show
elsewhere in this project. `block_b` (0.4 kg) correctly permits: within budget, and 14 N sits inside
the real window for it (13.1 N needed to hold it by friction, 15 N fragile-object cap) -- every other
check reads real, fresh, in-range data off the real topics.

## Known limitations, stated plainly

- **One demonstrated scenario, not a campaign.** This is the "prove it's real, not vibe-coded" step,
  not the hazard-campaign depth the Franka/ANYmal-C/G1 examples have. A next step: add the same
  injected-hazard scenarios (stale sensor, low visibility, human proximity) this project already has
  a pattern for, replayed over real ROS 2 topics instead of Isaac Lab sim state.
- **`gripper_state` is unreported** (hardcoded 0.5) -- this minimal bridge doesn't have a gripper
  topic to read; a real integration adds one for its specific gripper hardware/driver.
- **No `tf2` yet.** `ee_pose` is read directly off a `PoseStamped` topic for simplicity; a more
  complete integration would look it up via `tf2_ros` (base_link -> end-effector), the more standard
  ROS 2 pattern for robot pose, and was deliberately deferred to keep this first step's scope honest
  and checkable rather than sprawling.
- **`sensor_timestamp` assumes wall-clock time** (`use_sim_time` false) -- see
  `_stamp_to_epoch_s`'s docstring in `ros2.py`.
- **This has not been run against a real robot's actual driver stack**, only against `mock_robot.py`.
  That's exactly the gap real-hardware access (see the roadmap doc) would close next.
