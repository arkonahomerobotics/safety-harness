"""Real-physics G1 grasp grid parameterized in the WRIST frame, from hand_fk.py's measurements.

Measured left Dex3 geometry (wrist frame, cm): fingers extend along +x (index_1/middle_1 origins at
x=16.5, index at z=+2.6, middle at z=-3.1); fingers curl toward -y; the thumb sticks out along -y
from x~6.5 and swings toward +x on closing. So the palm faces wrist -y and the grasp pocket is the
region x in [7, 15], y < 0 -- the block is squeezed between the thumb (near side) and the curled
fingers (far side) along wrist x. Earlier grids approached along the finger axis, which drags the
sideways-sticking thumb straight through the block (514/540 knocked it before closing).

Each env: pick a roll of the hand about the finger axis (world Y; 0 = palm faces world +X, 90 = palm
faces down), place the block at wrist-frame offset p = (px, -py, pz), approach along the palm
normal from 10cm back, close, lift, hold. The IK lags its target by 1-3cm, so the commanded wrist
position is corrected in closed loop from the observed eef position.
"""

import argparse
import itertools
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0")
parser.add_argument("--grid", type=str, required=True)
parser.add_argument("--chunk", type=str, default="0/1")
parser.add_argument("--lift", type=float, default=0.12)
parser.add_argument("--dump", type=str, default="")
parser.add_argument("--block_pos", type=str, default="[-0.26,0.36,0.72]", help="env-relative spawn; default is inside the left wrist's reachable band (reach_map.py)")
parser.add_argument("--block_size", type=float, default=0.045)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul  # noqa: E402  (quats are xyzw in this Isaac Lab)

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

g = json.loads(args_cli.grid)
KEYS = ["roll", "px", "py", "pz", "thumb", "close"]
FULL = list(itertools.product(*(g[k] for k in KEYS)))
ci, cn = (int(v) for v in args_cli.chunk.split("/"))
GRID = FULL[ci::cn]
N = len(GRID)

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
original = env_cfg.scene.object
env_cfg.scene.object = RigidObjectCfg(
    prim_path=original.prim_path,
    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(json.loads(args_cli.block_pos)), rot=(0.0, 0.0, 0.0, 1.0)),
    spawn=sim_utils.CuboidCfg(
        size=(args_cli.block_size,) * 3,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.8)),
    ),
)
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

HAND_JOINT_NAMES = [
    "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
    "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "right_hand_thumb_2_joint",
]
LEFT_THUMB_YAW_SLOT = 2
LEFT_CURL_SLOTS = [0, 1, 6, 7, 8, 12]

with torch.inference_mode():
    obs, _ = env.reset()
    p = obs["policy"]
    dev = env.device
    robot = env.scene["robot"]
    obj = env.scene["object"]
    origins = env.scene.env_origins

    q0 = p["left_eef_quat"].clone()
    right_pos0 = p["right_eef_pos"].clone()
    right_quat0 = p["right_eef_quat"].clone()
    obj0 = obj.data.root_pos_w.torch - origins

    col = {k: torch.tensor([c[i] for c in GRID], device=dev, dtype=torch.float32) for i, k in enumerate(KEYS)}
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(N, 3)
    q_rot = quat_from_angle_axis(torch.deg2rad(col["roll"]), y_axis)  # about world Y (finger axis); +roll tips the palm down
    q_grasp = quat_mul(q_rot, q0)

    # the eef frame's orientation in world; hand_fk measured link offsets in the wrist_yaw_link frame,
    # which the eef quat tracks (obs quat [0,0,.707,.707] vs wrist body quat [-.01,.02,.71,.70]).
    p_local = torch.stack([col["px"], -col["py"], col["pz"]], dim=-1)
    palm_n = quat_apply(q_grasp, torch.tensor([0.0, -1.0, 0.0], device=dev).expand(N, 3))
    block_target = obj0.clone()
    grasp_pos = block_target - quat_apply(q_grasp, p_local)
    pre_pos = grasp_pos - 0.10 * palm_n  # back off against the palm normal, then move along it
    pre_pos[:, 2] = torch.maximum(pre_pos[:, 2], grasp_pos[:, 2])  # never approach from under the table
    lift_pos = grasp_pos.clone()
    lift_pos[:, 2] += args_cli.lift

    hand_ids, _ = robot.find_joints(HAND_JOINT_NAMES)
    lim = robot.data.soft_joint_pos_limits.torch[0, hand_ids]
    lo, hi = lim[:, 0], lim[:, 1]
    open_vals = torch.clamp(torch.zeros_like(lo), lo, hi)
    closed_vals = torch.where(lo.abs() > hi.abs(), lo, hi)
    open_hand = open_vals.unsqueeze(0).repeat(N, 1)
    open_hand[:, LEFT_THUMB_YAW_SLOT] = col["thumb"]
    closed_hand = open_hand.clone()
    frac = col["close"].unsqueeze(1)
    closed_hand[:, LEFT_CURL_SLOTS] = open_vals[LEFT_CURL_SLOTS] + frac * (closed_vals[LEFT_CURL_SLOTS] - open_vals[LEFT_CURL_SLOTS])

    state = {"obs": p, "corr": torch.zeros(N, 3, device=dev)}

    def act(target, hand, correct=True):
        if correct:  # integral correction of the IK's steady-state lag, clamped
            err = target - state["obs"]["left_eef_pos"]
            state["corr"] = (state["corr"] + 0.15 * err).clamp(-0.05, 0.05)
        a = torch.zeros(N, 28, device=dev)
        a[:, 0:3] = target + state["corr"]
        a[:, 3:7] = q_grasp
        a[:, 7:10] = right_pos0
        a[:, 10:14] = right_quat0
        a[:, 14:28] = hand
        o, *_ = env.step(a)
        state["obs"] = o["policy"]

    rel = lambda: obj.data.root_pos_w.torch - origins  # noqa: E731

    for k in range(60):  # go to pre-grasp (lift first so nothing sweeps the table)
        mid = pre_pos.clone()
        mid[:, 2] += 0.08 * max(0.0, 1 - k / 40)
        act(mid, open_hand)
    for k in range(60):
        f = min(1.0, (k + 1) / 40)
        act(pre_pos + f * (grasp_pos - pre_pos), open_hand)
    after_approach = rel().clone()
    eef_err = torch.linalg.norm(state["obs"]["left_eef_pos"] - grasp_pos, dim=1)
    for k in range(30):
        f = (k + 1) / 30
        act(grasp_pos, open_hand + f * (closed_hand - open_hand))
    for _ in range(20):
        act(grasp_pos, closed_hand)
    after_close = rel().clone()
    for k in range(80):
        f = min(1.0, (k + 1) / 50)
        act(grasp_pos + f * (lift_pos - grasp_pos), closed_hand)
    after_lift = rel().clone()
    for _ in range(40):
        act(lift_pos, closed_hand)
    after_hold = rel().clone()

    settle = after_approach[:, 2]
    rise_hold = after_hold[:, 2] - settle
    knocked = torch.linalg.norm((after_approach - obj0)[:, :2], dim=1)
    touched = torch.linalg.norm(after_close - after_approach, dim=1)
    carried = torch.linalg.norm(after_hold - state["obs"]["left_eef_pos"], dim=1)

    order = torch.argsort(rise_hold, descending=True)
    tag = args_cli.chunk
    print(f"RESULTS[{tag}] N={N} grip(>6cm)={int((rise_hold > 0.06).sum())} partial(>2cm)={int((rise_hold > 0.02).sum())} "
          f"touched={int((touched > 0.005).sum())} knocked={int((knocked > 0.02).sum())} "
          f"median_eef_err={eef_err.median():.3f}")
    for i in order[:15].tolist():
        c = dict(zip(KEYS, GRID[i]))
        print(f"ROW[{tag}] " + " ".join(f"{k}={v}" for k, v in c.items())
              + f" | eef_err={eef_err[i]:.3f} knock={knocked[i]:.3f} touch={touched[i]:.3f} hold={rise_hold[i]:.3f} obj-eef={carried[i]:.3f}")
    if args_cli.dump:
        rows = [dict(zip(KEYS, GRID[i]), hold=float(rise_hold[i]), knock=float(knocked[i]), touch=float(touched[i]),
                     eef_err=float(eef_err[i])) for i in range(N)]
        json.dump(rows, open(args_cli.dump, "w"))

simulation_app.close()
