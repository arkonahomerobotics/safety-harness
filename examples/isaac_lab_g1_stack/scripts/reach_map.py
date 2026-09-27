"""Map where the G1 left wrist can actually reach (Pink IK, fixed base) at block-grasp heights and
orientations, plus the packing table's real bounding box -- to put the blocks where the hand works."""

import argparse
import itertools

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab.utils.math import quat_error_magnitude, quat_from_angle_axis, quat_mul  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

XS = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05]
YS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
ZS = [0.73, 0.78]
ROLLS = [0.0, 30.0, 60.0]
GRID = list(itertools.product(XS, YS, ZS, ROLLS))
N = len(GRID)

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

stage = omni.usd.get_context().get_stage()
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
r = bbox.ComputeWorldBound(stage.GetPrimAtPath("/World/envs/env_0/PackingTable")).ComputeAlignedRange()
o0 = env.scene.env_origins[0].tolist()
print("TABLE bbox env-rel min", [round(a - b, 3) for a, b in zip(r.GetMin(), o0)], "max", [round(a - b, 3) for a, b in zip(r.GetMax(), o0)])

with torch.inference_mode():
    obs, _ = env.reset()
    p = obs["policy"]
    dev = env.device
    t = torch.tensor([g[:3] for g in GRID], device=dev)
    roll = torch.deg2rad(torch.tensor([g[3] for g in GRID], device=dev))
    q_rot = quat_from_angle_axis(roll, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(N, 3))
    q = quat_mul(q_rot, p["left_eef_quat"])
    a = torch.zeros(N, 28, device=dev)
    a[:, 3:7] = q
    a[:, 7:10] = p["right_eef_pos"]
    a[:, 10:14] = p["right_eef_quat"]
    start = p["left_eef_pos"].clone()
    for k in range(150):
        f = min(1.0, (k + 1) / 80)
        a[:, 0:3] = start + f * (t - start)
        obs, *_ = env.step(a)
    e = torch.linalg.norm(obs["policy"]["left_eef_pos"] - t, dim=1)
    qe = quat_error_magnitude(obs["policy"]["left_eef_quat"], q)
    for z in ZS:
        for rl in ROLLS:
            print(f"--- z={z} roll={rl}: pos err cm (rows y, cols x={XS}) / rot err deg")
            for y in YS:
                cells = []
                for x in XS:
                    i = GRID.index((x, y, z, rl))
                    cells.append(f"{e[i]*100:4.1f}/{qe[i]*57.3:3.0f}")
                print(f"  y={y:.2f} " + " ".join(cells))

simulation_app.close()
