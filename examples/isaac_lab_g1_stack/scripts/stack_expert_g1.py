"""Scripted G1 expert: grasp the blue block with real contact physics and stack it on the red block.

No kinematic attach anywhere -- the blue block only moves if the Dex3 fingers carry it. Runs every
env in lockstep with its own phase machine, with per-env randomized block positions, so one
process measures the expert's success rate over many trials at once.

Grasp: palm-facing pinch from hand_fk.py's measured Dex3 geometry -- the hand rolled about its finger
axis (world Y), the block placed at a wrist-frame pocket offset (px along the fingers, py off the
palm, pz across the finger spread), approached along the palm normal so the sideways thumb never
sweeps through it, parameters from grid_pocket.py. Blocks sit in the left wrist's reachable band
(reach_map.py) -- the task's stock spawn at (-0.35, 0.45) is outside it. The place step closes the loop on the
blue block's OBSERVED position (not the wrist), so whatever offset the block ended up at inside the
hand is compensated before release.
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--grasp", type=str, default='{"roll":45,"px":0.105,"py":0.04,"pz":0.0,"thumb":0.0,"close":1.0}')
parser.add_argument("--a_pos", type=str, default="[-0.28,0.36,0.72]")
parser.add_argument("--b_pos", type=str, default="[-0.17,0.36,0.72]")
parser.add_argument("--block_size", type=float, default=0.045)
parser.add_argument("--rand", type=float, default=0.02, help="uniform +/- xy noise on both block positions")
parser.add_argument("--max_steps", type=int, default=700)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", type=str, default="", help="mp4 path: record env 0 from a fixed camera")
parser.add_argument("--dump", type=str, default="")
parser.add_argument("--release_steps", type=int, default=20)
parser.add_argument("--release_order", choices=["all", "thumb_first", "fingers_first", "fingers_only"], default="all")
parser.add_argument("--retreat", choices=["normal", "up", "back"], default="normal")
parser.add_argument("--open_frac", type=float, default=1.0, help="how far toward fully open the fingers go on release")
parser.add_argument("--press", type=float, default=0.0, help="push the wrist down this much (m) while opening")
parser.add_argument("--corr_gain", type=float, default=0.15, help="IK-lag integral correction gain (0 = raw IK, as in the grasp grids)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul  # noqa: E402  (quats are xyzw in this Isaac Lab)

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

GRASP = json.loads(args_cli.grasp)
A_POS = tuple(json.loads(args_cli.a_pos))
B_POS = tuple(json.loads(args_cli.b_pos))
BLOCK = args_cli.block_size

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed
env_cfg.episode_length_s = 1e4  # the expert decides when it is done
if getattr(env_cfg, "terminations", None) is not None:
    for name in list(vars(env_cfg.terminations)):
        if name != "time_out":
            setattr(env_cfg.terminations, name, None)


def block_cfg(prim_path, pos, color):
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(BLOCK, BLOCK, BLOCK),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


a_pos = A_POS
env_cfg.scene.object = block_cfg(env_cfg.scene.object.prim_path, A_POS, (0.15, 0.4, 0.85))
env_cfg.scene.block_b = block_cfg("{ENV_REGEX_NS}/BlockB", B_POS, (0.85, 0.2, 0.15))

if args_cli.video:
    from isaaclab.sensors import CameraCfg

    env_cfg.scene.demo_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCam",
        update_period=0.0,
        height=540,
        width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
    )

env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

HAND_JOINT_NAMES = [
    "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
    "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "right_hand_thumb_2_joint",
]
LEFT_CURL_SLOTS = [0, 1, 6, 7, 8, 12]
FINGER_SLOTS = [0, 1, 6, 7]  # left index/middle proximal + distal
THUMB_SLOTS = [8, 12]  # left thumb_1, thumb_2

# phases
ABOVE, DESCEND, CLOSE, LIFT, CARRY, LOWER, RELEASE, RETREAT, DONE = range(9)
PHASE_NAMES = ["above", "descend", "close", "lift", "carry", "lower", "release", "retreat", "done"]
TIMEOUT = torch.tensor([90, 70, 45, 70, 140, 110, 10**9, 40, 10**9])  # RELEASE length set from --release_steps below

with torch.inference_mode():
    obs, _ = env.reset()
    dev = env.device
    N = env.num_envs
    robot = env.scene["robot"]
    blk_a = env.scene["object"]
    blk_b = env.scene["block_b"]
    origins = env.scene.env_origins
    gen = torch.Generator(device=dev).manual_seed(args_cli.seed)

    # randomize both blocks in xy and settle
    def place(blk, base):
        pos = torch.tensor(base, device=dev).repeat(N, 1)
        pos[:, :2] += (torch.rand(N, 2, device=dev, generator=gen) * 2 - 1) * args_cli.rand
        pose = torch.cat([pos + origins, torch.tensor([0.0, 0, 0, 1.0], device=dev).repeat(N, 1)], dim=1)
        blk.write_root_pose_to_sim(pose)
        blk.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))

    if args_cli.video:  # front-left view of the hand and both blocks
        mid = torch.tensor([(A_POS[0] + B_POS[0]) / 2, (A_POS[1] + B_POS[1]) / 2, A_POS[2] + 0.03], device=dev)
        eye = mid + torch.tensor([0.30, 0.50, 0.32], device=dev)
        env.scene["demo_cam"].set_world_poses_from_view(eye.repeat(N, 1) + origins, mid.repeat(N, 1) + origins)

    place(blk_a, a_pos)
    place(blk_b, B_POS)

    p = obs["policy"]
    q0 = p["left_eef_quat"].clone()
    rest_pos = p["left_eef_pos"].clone()
    right_pos0 = p["right_eef_pos"].clone()
    right_quat0 = p["right_eef_quat"].clone()

    roll = torch.deg2rad(torch.full((N,), float(GRASP["roll"]), device=dev))
    q_rot = quat_from_angle_axis(roll, torch.tensor([0.0, 1.0, 0.0], device=dev).repeat(N, 1))
    q_grasp = quat_mul(q_rot, q0)  # roll about world Y = the finger axis
    p_local = torch.tensor([GRASP["px"], -GRASP["py"], GRASP["pz"]], device=dev).repeat(N, 1)
    grasp_off = -quat_apply(q_grasp, p_local)  # wrist - block at grasp
    palm_n = quat_apply(q_grasp, torch.tensor([0.0, -1.0, 0.0], device=dev).repeat(N, 1))
    pre_off = grasp_off - 0.10 * palm_n
    pre_off[:, 2] = torch.maximum(pre_off[:, 2], grasp_off[:, 2])

    hand_ids, _ = robot.find_joints(HAND_JOINT_NAMES)
    lim = robot.data.soft_joint_pos_limits.torch[0, hand_ids]
    lo, hi = lim[:, 0], lim[:, 1]
    open_vals = torch.clamp(torch.zeros_like(lo), lo, hi)
    closed_vals = torch.where(lo.abs() > hi.abs(), lo, hi)
    open_hand = open_vals.clone()
    open_hand[2] = GRASP["thumb"]
    closed_hand = open_hand.clone()
    closed_hand[LEFT_CURL_SLOTS] = open_vals[LEFT_CURL_SLOTS] + GRASP["close"] * (closed_vals[LEFT_CURL_SLOTS] - open_vals[LEFT_CURL_SLOTS])

    phase = torch.full((N,), ABOVE, dtype=torch.long, device=dev)
    t_in = torch.zeros(N, dtype=torch.long, device=dev)
    corr = torch.zeros(N, 3, device=dev)
    carry_goal = torch.zeros(N, 3, device=dev)
    held_z0 = torch.zeros(N, device=dev)
    grasp_hold = torch.zeros(N, 3, device=dev)
    close_pos = torch.zeros(N, 3, device=dev)
    release_pos = torch.zeros(N, 3, device=dev)
    rel_info = torch.zeros(N, 5, device=dev)  # at release: xy err, height above seat, steps spent lowering, dx, dy
    after_rel = torch.zeros(N, 3, device=dev)  # blue-red offset at the end of RELEASE (before retreat)
    ever_lifted = torch.zeros(N, dtype=torch.bool, device=dev)
    fail_reason = ["" for _ in range(N)]
    timeout_t = TIMEOUT.to(dev)
    timeout_t[RELEASE] = args_cli.release_steps + 15
    frames = []
    hand_cmd = open_hand.repeat(N, 1)

    for step in range(args_cli.max_steps):
        pa = blk_a.data.root_pos_w.torch - origins
        pb = blk_b.data.root_pos_w.torch - origins
        eef = obs["policy"]["left_eef_pos"]

        target = eef.clone()
        grasp_pos = pa + grasp_off
        stack_z = pb[:, 2] + BLOCK
        carry_h = 0.10

        m = phase == ABOVE  # pre-grasp: backed off along the palm normal (raised first so nothing sweeps)
        target[m] = (pa + pre_off)[m]
        target[m, 2] += 0.06 * (1 - t_in[m].float() / 40).clamp(min=0.0)
        hand_cmd[m] = open_hand
        m = phase == DESCEND  # approach along the palm normal, ramped
        f = (t_in.float() / 35).clamp(max=1.0).unsqueeze(1)
        target[m] = (pa + pre_off + f * (grasp_off - pre_off))[m]
        m = phase == CLOSE
        target[m] = close_pos[m]
        f = (t_in.float() / 30).clamp(max=1.0).unsqueeze(1)
        hand_cmd[m] = (open_hand + f * (closed_hand - open_hand))[m]
        m = phase == LIFT
        f = (t_in.float() / 40).clamp(max=1.0)
        target[m] = grasp_hold[m]
        target[m, 2] = (grasp_hold[:, 2] + f * carry_h)[m]
        hand_cmd[m] = closed_hand
        # CARRY / LOWER: move a rate-limited goal for the OBSERVED blue block toward above / onto the
        # red block, and command the wrist at goal + the block's measured in-hand offset -- so any
        # offset the grasp left the block at is compensated before release.
        goal_blk = torch.cat([pb[:, :2], (stack_z + carry_h).unsqueeze(1)], dim=1)
        m_lower = phase == LOWER
        goal_blk[m_lower, 2] = stack_z[m_lower] - 0.002  # aim slightly into the seat; release triggers at +6mm
        m_cl = (phase == CARRY) | m_lower
        step_vec = goal_blk - carry_goal
        step_len = torch.linalg.norm(step_vec, dim=1, keepdim=True).clamp(min=1e-9)
        carry_goal[m_cl] = (carry_goal + step_vec * (0.003 / step_len).clamp(max=1.0))[m_cl]
        in_hand = eef - pa
        target[m_cl] = (carry_goal + in_hand)[m_cl]
        hand_cmd[m_cl] = closed_hand
        m = phase == RELEASE  # open gradually while holding the release pose
        f = (t_in.float() / args_cli.release_steps).clamp(max=1.0).unsqueeze(1)
        if args_cli.release_order == "all":
            fw = f.expand(N, 14)
        else:
            f1 = (2 * f).clamp(max=1.0).expand(N, 14).clone()
            f2 = (2 * f - 1).clamp(min=0.0).expand(N, 14).clone()
            first, second = (THUMB_SLOTS, FINGER_SLOTS) if args_cli.release_order == "thumb_first" else (FINGER_SLOTS, THUMB_SLOTS)
            fw = torch.zeros(N, 14, device=dev)
            fw[:, first] = f1[:, first]
            if args_cli.release_order != "fingers_only":
                fw[:, second] = f2[:, second]
        hand_cmd[m] = (closed_hand + args_cli.open_frac * fw * (open_hand - closed_hand))[m]
        target[m] = release_pos[m]
        target[m, 2] -= args_cli.press * f[m, 0]
        m = phase == RETREAT  # back away along the palm normal, then up -- straight up drags the block
        f = (t_in.float() / 25).clamp(max=1.0).unsqueeze(1)
        if args_cli.retreat == "normal":
            away = -0.06 * palm_n + torch.tensor([0, 0, 0.04], device=dev)
        elif args_cli.retreat == "back":  # slide out along -Y, i.e. back along the finger axis
            away = torch.tensor([0.0, -0.08, 0.02], device=dev).repeat(N, 1)
        else:
            away = torch.tensor([0.0, 0.0, 0.07], device=dev).repeat(N, 1)
        target[m] = (release_pos + f * away)[m]
        hand_cmd[m] = closed_hand + args_cli.open_frac * (open_hand - closed_hand)  # stay partly open until clear
        m = phase == DONE
        target[m] = eef[m]

        # closed-loop IK-lag correction while tracking; FROZEN (not zeroed) during release/retreat --
        # zeroing it made the wrist lurch by up to 5cm the moment the fingers opened, shoving the
        # just-placed block off the red one.
        m_c = phase <= LOWER
        corr[m_c] = (corr[m_c] + args_cli.corr_gain * (target - eef)[m_c]).clamp(-0.05, 0.05)
        cmd = target + corr

        act = torch.zeros(N, 28, device=dev)
        act[:, 0:3] = cmd
        act[:, 3:7] = q_grasp
        act[:, 7:10] = right_pos0
        act[:, 10:14] = right_quat0
        act[:, 14:28] = hand_cmd
        obs, *_ = env.step(act)
        t_in += 1

        if args_cli.video:
            frames.append(env.scene["demo_cam"].data.output["rgb"][0, ..., :3].clone().cpu())

        # transitions
        eef = obs["policy"]["left_eef_pos"]
        pa = blk_a.data.root_pos_w.torch - origins
        pb = blk_b.data.root_pos_w.torch - origins
        d = torch.linalg.norm(eef - target, dim=1)
        nxt = phase.clone()
        nxt[(phase == ABOVE) & (((d < 0.02) & (t_in > 45)) | (t_in >= timeout_t[ABOVE]))] = DESCEND
        to_close = (phase == DESCEND) & (((d < 0.012) & (t_in > 40)) | (t_in >= timeout_t[DESCEND]))
        close_pos[to_close] = target[to_close]
        nxt[to_close] = CLOSE
        closed_done = (phase == CLOSE) & (t_in >= timeout_t[CLOSE])
        held_z0[closed_done] = pa[closed_done, 2]
        grasp_hold[closed_done] = close_pos[closed_done]
        nxt[closed_done] = LIFT
        lifted = pa[:, 2] > held_z0 + 0.05
        ever_lifted |= lifted & (phase >= LIFT)
        to_carry = (phase == LIFT) & lifted & (t_in > 25)
        carry_goal[to_carry] = pa[to_carry]
        nxt[to_carry] = CARRY
        blk_xy = torch.linalg.norm((pa - pb)[:, :2], dim=1)
        nxt[(phase == CARRY) & (blk_xy < 0.01) & ((pa[:, 2] - pb[:, 2]) > BLOCK + 0.03)] = LOWER
        to_rel = (phase == LOWER) & (((pa[:, 2] - pb[:, 2]) < BLOCK + 0.006) & (blk_xy < 0.008) | (t_in >= timeout_t[LOWER]))
        release_pos[to_rel] = eef[to_rel]
        rel_info[to_rel] = torch.stack([blk_xy, pa[:, 2] - pb[:, 2] - BLOCK, t_in.float(), (pa - pb)[:, 0], (pa - pb)[:, 1]], dim=1)[to_rel]
        nxt[to_rel] = RELEASE
        to_ret = (phase == RELEASE) & (t_in >= timeout_t[RELEASE])
        after_rel[to_ret] = (pa - pb)[to_ret]
        nxt[to_ret] = RETREAT
        nxt[(phase == RETREAT) & (t_in >= timeout_t[RETREAT])] = DONE
        # failures: dropped during carry, never lifted, carry timeout
        dropped = (phase >= LIFT) & (phase <= LOWER) & (pa[:, 2] < held_z0 + 0.01) & (t_in > 40) & (phase != LIFT)
        lift_fail = (phase == LIFT) & (t_in >= timeout_t[LIFT])
        carry_to = (phase == CARRY) & (t_in >= timeout_t[CARRY])
        for i in torch.nonzero(dropped | lift_fail | carry_to).flatten().tolist():
            fail_reason[i] = "dropped" if dropped[i] else ("no_lift" if lift_fail[i] else "carry_timeout")
        nxt[dropped | lift_fail | carry_to] = DONE
        t_in[nxt != phase] = 0
        phase = nxt
        if bool((phase == DONE).all()):
            break

    for _ in range(30):  # let everything settle, then judge
        act[:, 0:3] = obs["policy"]["left_eef_pos"]
        act[:, 14:28] = open_hand
        obs, *_ = env.step(act)
        if args_cli.video:
            frames.append(env.scene["demo_cam"].data.output["rgb"][0, ..., :3].clone().cpu())
    pa = blk_a.data.root_pos_w.torch - origins
    pb = blk_b.data.root_pos_w.torch - origins
    dxy = torch.linalg.norm((pa - pb)[:, :2], dim=1)
    dz = pa[:, 2] - pb[:, 2]
    vel = torch.linalg.norm(blk_a.data.root_vel_w.torch[:, :3], dim=1)
    success = (dxy < 0.02) & ((dz - BLOCK).abs() < 0.01) & (vel < 0.05) & ever_lifted
    print(f"STACK RESULT release={args_cli.release_order}/{args_cli.release_steps} open={args_cli.open_frac} retreat={args_cli.retreat} press={args_cli.press} seed={args_cli.seed} a={A_POS} b={B_POS} grasp={GRASP}: success {int(success.sum())}/{N}  "
          f"lifted {int(ever_lifted.sum())}/{N}  steps={step+1}")
    reasons = {}
    for i in range(N):
        r = "success" if success[i] else (fail_reason[i] or f"misplaced(dxy={dxy[i]:.3f},dz={dz[i]-BLOCK:+.3f})")
        key = r.split("(")[0]
        reasons[key] = reasons.get(key, 0) + 1
    print("OUTCOMES:", reasons)
    fails = [i for i in range(N) if not success[i]]
    print("FAILED envs:", len(fails))
    for i in fails[:12]:
        ri, ar = rel_info[i].tolist(), after_rel[i].tolist()
        print(f"  FAIL env{i}: final dxy={dxy[i]:.3f} dz={dz[i]-BLOCK:+.3f} | at release xy={ri[0]:.3f} (dx {ri[3]:+.3f} dy {ri[4]:+.3f}) "
              f"h={ri[1]:+.3f} lower_steps={ri[2]:.0f} | end-of-release off=({ar[0]:+.3f},{ar[1]:+.3f},{ar[2]-BLOCK:+.3f}) fail={fail_reason[i]!r}")
    okm = success
    if okm.any():
        print(f"  successes at release: mean xy={rel_info[okm, 0].mean():.4f} h={rel_info[okm, 1].mean():+.4f} lower_steps={rel_info[okm, 2].mean():.0f}")
    for i in range(min(N, 3)):
        print(f"  env{i}: dxy={dxy[i]:.3f} dz-block={dz[i]-BLOCK:+.3f} vel={vel[i]:.3f} lifted={bool(ever_lifted[i])} "
              f"fail={fail_reason[i]!r}")
    if args_cli.dump:
        json.dump({"grasp": GRASP, "success": success.float().mean().item(), "lifted": ever_lifted.float().mean().item(),
                   "reasons": reasons}, open(args_cli.dump, "w"))
    if args_cli.video and frames:
        import imageio

        with imageio.get_writer(args_cli.video, fps=30, codec="libx264", quality=8) as w:
            for fr in frames:
                w.append_data(fr.numpy().astype("uint8"))
        print(f"VIDEO {args_cli.video} frames={len(frames)}")

simulation_app.close()
