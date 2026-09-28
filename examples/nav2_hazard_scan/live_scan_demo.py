"""Runnable, self-checking demonstration: the SAME hazard rules as scan_demo.py (corridor width,
blind corners), run against a real LIVE Nav2 costmap instead of a downloaded map file -- a real
TurtleBot3, physically simulated in a real headless Gazebo, with real `nav2_bringup` producing a
real `/global_costmap/costmap`, read ONCE.

DESIGN BOUNDARY (see the top-level README and hazard_rules.py's own docstring): this launches Nav2,
waits for ONE real costmap message, converts it, runs the rules, and exits -- it does not subscribe
continuously, does not loop, and never feeds a finding back into navigation. That's deliberate: a
one-time read that produces a report for a human is what keeps this tool outside "safety component"
classification. Extending this into a continuous re-evaluation loop would be a real design change
needing that classification question revisited first, not a casual next step.

    python3 live_scan_demo.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node

SPAWN_X, SPAWN_Y = -2.0, -0.5  # turtlebot3_world.launch.py's own default x_pose/y_pose

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import clearance_transform  # noqa: E402
from hazard_rules import blind_corner_absent, corridor_width_sufficient  # noqa: E402
from live_costmap import occupancy_grid_from_msg  # noqa: E402
from map_io import FREE  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "turtlebot3_gazebo_hooks"))
from gate_demo import launch_gazebo_headless, robot_state_publisher, wait_for_spawn_service  # noqa: E402


class _OneShotCostmapReader(Node):
    def __init__(self):
        super().__init__("hazard_scan_costmap_reader")
        self.msg = None
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap", self._on_costmap, 10)

    def _on_costmap(self, msg):
        if self.msg is None:  # ONE message only -- see the module docstring
            self.msg = msg


def main() -> int:
    procs = []
    try:
        gzserver, world = launch_gazebo_headless()
        procs.append(gzserver)
        print(f"gzserver starting with world {world} ...")
        if not wait_for_spawn_service():
            print("FAIL: /spawn_entity service never became available")
            return 1
        spawn = subprocess.run(
            ["ros2", "launch", "turtlebot3_gazebo", "spawn_turtlebot3.launch.py"],
            capture_output=True, text=True, timeout=40,
        )
        if "Successfully spawned" not in spawn.stdout + spawn.stderr:
            print(f"FAIL: spawn_turtlebot3 did not report success:\n{spawn.stdout}\n{spawn.stderr}")
            return 1
        print("Robot spawned")
        procs.append(robot_state_publisher())
        time.sleep(3)

        # Deliberately our own checked-in map, not a map of the actual turtlebot3_world Gazebo
        # world the robot is really standing in -- AMCL only needs A valid map to produce a
        # map->odom transform (which is all global_costmap needs to start publishing at all), and
        # using our own map keeps this consistent with scan_demo.py's static-file run. This means
        # AMCL's own localization won't be accurate against the real environment -- irrelevant here,
        # since nothing downstream of the costmap depends on true localization, only on getting a
        # real, structurally valid OccupancyGrid message to run the same hazard rules against.
        nav2_map = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "warehouse.yaml")
        print(f"Launching nav2_bringup against {nav2_map} for a live costmap ...")
        nav2 = subprocess.Popen(
            ["ros2", "launch", "nav2_bringup", "bringup_launch.py",
             f"map:={nav2_map}", "use_sim_time:=true", "autostart:=true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(nav2)

        rclpy.init()
        reader = _OneShotCostmapReader()
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(reader)

        # AMCL never publishes a map->odom transform until it has an initial pose -- without one,
        # global_costmap can wait forever for a "map" frame that will never appear, independent of
        # which map file was given. Found live: exec'd into a debug container, watched
        # global_costmap log "Invalid frame ID 'map' ... frame does not exist" indefinitely, then
        # confirmed publishing this once unblocks it immediately. The real spawn pose is known
        # (turtlebot3_world.launch.py's own x_pose/y_pose default), so this isn't a guess.
        initial_pose_pub = reader.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        pose_deadline = time.monotonic() + 5.0
        while time.monotonic() < pose_deadline:
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = "map"
            msg.header.stamp = reader.get_clock().now().to_msg()
            msg.pose.pose.position.x = SPAWN_X
            msg.pose.pose.position.y = SPAWN_Y
            msg.pose.pose.orientation.w = 1.0
            msg.pose.covariance[0] = 0.25
            msg.pose.covariance[7] = 0.25
            msg.pose.covariance[35] = 0.06
            initial_pose_pub.publish(msg)
            executor.spin_once(timeout_sec=0.5)
        print(f"Published initial pose ({SPAWN_X}, {SPAWN_Y}) to bootstrap AMCL")

        deadline = time.monotonic() + 90.0
        while reader.msg is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.5)
        if reader.msg is None:
            print("FAIL: never received a real /global_costmap/costmap message from nav2_bringup")
            return 1

        grid = occupancy_grid_from_msg(reader.msg)
        n_free = int((grid.cells == FREE).sum())
        print(f"Live costmap received: {grid.cells.shape[1]}x{grid.cells.shape[0]} cells @ "
              f"{grid.resolution_m}m/cell, {n_free} free cells")
        reader.destroy_node()
        rclpy.shutdown()

        if n_free < 4:
            print("FAIL: live costmap reports almost no free space -- can't scan a meaningful route")
            return 1

        # Same rules, same code path as scan_demo.py's static-file run -- just fed live data.
        import numpy as np
        free_idx = np.argwhere(grid.cells == FREE)
        route = [tuple(int(v) for v in free_idx[i]) for i in range(0, len(free_idx), max(1, len(free_idx) // 20))]
        clearance = clearance_transform(grid.cells)
        findings = corridor_width_sufficient(grid, route, min_clearance_m=0.3, clearance_cells=clearance) + \
            blind_corner_absent(grid, route, required_sightline_m=1.5, sample_every=1)
        for f in findings:
            print(f"{'OK   ' if f.satisfied else 'FLAG '}{f.name:28s} {f.reason}")
        print(f"\nPASS: {len(findings)} findings against a REAL live costmap, same rules as the static-map demo")
        return 0
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
