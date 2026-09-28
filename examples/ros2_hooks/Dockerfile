# Reproduces exactly the environment this example was actually verified against: real rclpy, real
# DDS message passing, no Isaac Lab, no GPU. Build and run from the repo root:
#
#   docker build -f examples/ros2_hooks/Dockerfile -t safety-harness-ros2-demo .
#   docker run --rm safety-harness-ros2-demo
#
FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace
RUN pip3 install --no-cache-dir .

SHELL ["/bin/bash", "-c"]
CMD source /opt/ros/humble/setup.bash && python3 examples/ros2_hooks/gate_demo.py
