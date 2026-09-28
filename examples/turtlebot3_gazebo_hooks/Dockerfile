# A real TurtleBot3 in a real headless Gazebo (Classic) simulation, Nav2 bringup, on ROS 2 Humble --
# a fuller "real ROS 2 system" test than examples/ros2_hooks' synthetic mock publisher: a genuine
# physics-simulated robot, real gazebo_ros_pkgs bridging, real sensor_msgs/JointState + nav_msgs/
# Odometry + sensor_msgs/LaserScan.
#
ARG TARGETPLATFORM=linux/amd64
# --platform linux/amd64, passed at build time (docker build --platform linux/amd64 ...): deliberate,
# not the default. ros-humble-turtlebot3-gazebo and ros-humble-gazebo-ros-pkgs ship as prebuilt
# binaries for amd64 on packages.ros.org but NOT for arm64 (checked directly against the apt index,
# not assumed) -- building Gazebo Classic + gazebo_ros_pkgs from source for arm64 is a much larger,
# more fragile undertaking than running under Docker's amd64 emulation (Rosetta on Apple Silicon).
# Slower per-command, but far more reliable than a from-source build.
FROM --platform=$TARGETPLATFORM ros:humble-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ros-humble-turtlebot3-gazebo \
    ros-humble-turtlebot3-navigation2 \
    ros-humble-turtlebot3-bringup \
    ros-humble-nav2-bringup \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace
RUN pip3 install --no-cache-dir .

ENV TURTLEBOT3_MODEL=burger
ENV GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models

SHELL ["/bin/bash", "-c"]
# No X server needed: gzserver only (see gate_demo.py), no gzclient GUI. -u: unbuffered stdout, so
# real progress is visible in `docker logs` as it happens, not only when the process exits.
CMD source /opt/ros/humble/setup.bash && \
    python3 -u examples/turtlebot3_gazebo_hooks/gate_demo.py
