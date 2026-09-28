# TurtleBot3 + Gazebo + Nav2: gating a real physics-simulated robot over real ROS 2

A fuller "real ROS 2 system" validation than [`examples/ros2_hooks`](../ros2_hooks/)'s mock
publisher: a real TurtleBot3, physically simulated in a real headless Gazebo (Classic), spawned via
the real `gazebo_ros` factory service, with the gate reading its actual live `/odom` -- not a
hand-written stand-in.

## Why this exists

Direct follow-up to the same ROS Discourse moderator feedback that prompted `ros2_hooks` (a real
OSRA TGC member, on a removed stale post): "test this out on a real robot, running on a real ROS 2
system." `ros2_hooks` proved the software plugs into real ROS 2 message passing; this proves it
against an actually-running physics simulation instead of a synthetic mock.

## What's here

- **`bridge_node.py`**: a real `rclpy.node.Node`. Different from `ros2_hooks/bridge_node.py` in one
  deliberate way -- a differential-drive base has no manipulator joints and no single end effector,
  so its pose comes from the real `/odom` topic (`nav_msgs/Odometry`, published by Gazebo's
  diff-drive plugin) instead of a `/ee_pose` + URDF pair, following the same "base pose reported
  through `end_effector_pose`" convention the ANYmal-C quadruped adapter already uses.
- **`gate_demo.py`**: launches a real headless `gzserver` (no `gzclient` GUI, no display needed),
  polls for the real `/spawn_entity` service instead of guessing a fixed startup delay, spawns the
  real robot, brings up `robot_state_publisher`, then gates two real `navigate` proposals through
  `ActuatorGate` using `configs/turtlebot3_action_schema.yaml`.
- **`configs/turtlebot3_action_schema.yaml`** (repo root): mirrors the Isaac Lab ANYmal-C `navigate`
  schema, with checks omitted where there's no real data behind them for this robot -- see that
  file's own header comment for exactly which three and why.
- **`Dockerfile`**: `--platform linux/amd64` **on purpose**, even on Apple Silicon -- see "A real
  finding" below.

## Run it

```bash
docker build --platform linux/amd64 -f examples/turtlebot3_gazebo_hooks/Dockerfile -t tb3-gazebo-demo .
docker run --platform linux/amd64 --rm tb3-gazebo-demo
```

## Results (2026-09-28, real run, output pasted verbatim)

```
gzserver starting with world /opt/ros/humble/share/turtlebot3_gazebo/worlds/turtlebot3_world.world ...
gzserver's /spawn_entity service is live
Robot spawned
Bridge is live: real pose = geometry_msgs.msg.Point(x=0.4982334126699315, y=1.934298255038599, z=0.029587692847770714)
PASS: navigate with far agent (5m away) -> permit (expected permit)
PASS: navigate with close agent (0.5m from the path) -> block (expected block) failing=['swept_path_clear_of_agents', 'iso15066_separation_distance_maintained', 'vulnerable_bystander_protected']
PASS: gate_demo
```

The robot's pose is real physics -- not injected. The hazard (a person near the path) is injected
the same way every other demo in this project injects a hazard that isn't itself a simulated
physical body: published on this project's own `/safety_harness/tracked_agents` bridge topic.

## Two real bugs found and fixed while building this (not written correctly on the first try)

1. **`gzserver`'s ROS factory plugin took longer to initialize than `spawn_entity.py`'s own
   hardcoded 30s wait**, confirmed by manually inspecting the running container: `gzserver` starts
   and consumes real CPU well before `/spawn_entity` actually becomes available under emulation
   (see below). The first version of `gate_demo.py` used fixed `time.sleep()` calls and failed
   outright. Fixed by polling for the real service instead of guessing a duration.
2. **`sensor_data_fresh` always saw the bridge's data as catastrophically stale**, even seconds after
   a fresh reading. Root cause, found by echoing the real `/odom` topic directly: TurtleBot3's Gazebo
   odometry plugin stamps messages with **simulation time** (starts near zero when `gzserver`
   launches), not wall-clock time -- even though nothing about this bridge node itself requested sim
   time. `msg.header.stamp` of "12 seconds since the Unix epoch" compared against a real 2026
   timestamp is always stale. Fixed by stamping with the bridge node's own wall-clock receipt time
   (`self.get_clock().now()`) instead of passing the message's own sim-time stamp through -- an
   honest proxy, since real ROS 2 delivery over loopback DDS is milliseconds, not a meaningful
   staleness source against a 0.5s budget.

## A real finding, not a guess: `--platform linux/amd64` is deliberate

Checked directly against the `packages.ros.org` apt index before building anything:
`ros-humble-gazebo-ros-pkgs` and `ros-humble-turtlebot3-gazebo` ship as prebuilt binaries for amd64,
**not** for arm64. Building Gazebo Classic + `gazebo_ros_pkgs` from source for arm64 would have been
a much larger, more fragile undertaking than running the prebuilt amd64 binaries under Docker's
x86_64 emulation (Rosetta on Apple Silicon). Real cost of that choice, observed directly: the image
build (mostly `apt-get install`, pulling a large transitive dependency tree -- GDAL, GIS libraries,
image codecs) took on the order of 15-20 minutes under emulation, and `gzserver`'s own startup was
slow enough to need the real polling fix above rather than a short fixed sleep.

## Honest limitations, stated plainly

- **One scenario, not a hazard campaign.** Two `navigate` proposals (far agent / close agent), not
  the multi-scenario depth the Franka/ANYmal-C/G1 examples have.
- **`robot_state_confirmed` is not wired** for this robot -- explained in the action schema's own
  comment, not silently dropped.
- **No LIDAR-derived `swept_path_observed` coverage.** `observed_regions` is a generous static box
  (same simplification the Isaac Lab adapters make for their own privileged-state coverage), not
  derived from the real `/scan` topic this simulation does publish -- a real next step, not attempted
  here.
- **Not reviewed by a functional-safety assessor.** Engineering proof-of-concept evidence.
