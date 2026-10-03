"""Print the prim hierarchy (with physics schemas) of the spawned cubes and the Franka hand/finger links.

Used to verify the contact-sensor prim-path regexes against the current asset layout.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="IsaacContrib-Stack-Cube-Franka-IK-Rel")
parser.add_argument("--filter", type=str, default="panda_hand,panda_leftfinger,panda_rightfinger,Cube_1,Cube_2,Cube_3")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

env = gym.make(args_cli.task, cfg=parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)).unwrapped
stage = sim_utils.get_current_stage()
keys = args_cli.filter.split(",")
for prim in stage.Traverse():
    path = prim.GetPath().pathString
    if not path.startswith("/World/envs/env_0/"):
        continue
    if not any(k in path for k in keys):
        continue
    schemas = [s for s in prim.GetAppliedSchemas() if "Physics" in s or "Physx" in s or "Contact" in s or "Rigid" in s]
    print(f"PRIM {path}  [{prim.GetTypeName()}]  {schemas}")
robot = env.scene["robot"]
print("ROBOT body_names:", robot.data.body_names)
print("ROBOT joint_names:", robot.data.joint_names)
for name in ("cube_1", "cube_2", "cube_3"):
    print(name, "root_physx_view prim paths:", getattr(env.scene[name], "root_view", None))
env.close()
simulation_app.close()
