"""The two-block stacking expert, driven through the RL task's own action space.

Same phase machine and grasp as stack_expert_g1.py (palm-pocket pinch, real contact physics, no
attach), but commanding Isaac-Stack-Blocks-G1-RL-v0's 7-dim action (relative-pose differential IK +
scalar grip) instead of Pink IK. Three uses: (1) proves the GPU action space can do the task at all,
(2) measures env-steps/s for sizing PPO, (3) saves reverse-curriculum snapshots -- robot joint state
and both block poses at the moment the blue block is held above the red one -- for the reset event.
"""

import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--grasp", type=str, default='{"roll":20,"px":0.11,"py":0.05,"pz":-0.02}')
parser.add_argument("--max_steps", type=int, default=480)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--corr_gain", type=float, default=0.1)
parser.add_argument("--snapshots", type=str, default="")
parser.add_argument("--video", type=str, default="")
parser.add_argument("--record", type=str, default="", help="save (policy obs, clean expert action) pairs from successful episodes")
parser.add_argument("--act_noise", type=float, default=0.0, help="DART: max per-env gaussian noise on EXECUTED arm actions (labels stay clean)")
parser.add_argument("--quiet", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_apply_inverse, quat_from_angle_axis, quat_mul  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

GRASP = json.loads(args_cli.grasp)
BLOCK = 0.045
POS_SCALE, ROT_SCALE = 0.02, 0.05

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed
env_cfg.episode_length_s = 1e4
env_cfg.terminations.success = None  # the expert judges its own outcome at the end
if args_cli.video:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    env_cfg.scene.demo_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCam", update_period=0.0, height=540, width=960, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
    )
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

ABOVE, DESCEND, CLOSE, LIFT, CARRY, LOWER, RELEASE, RETREAT, DONE = range(9)
TIMEOUT = [130, 70, 45, 70, 140, 110, 35, 40, 10**9]

with torch.inference_mode():
    obs, _ = env.reset()
    dev, N = env.device, env.num_envs
    robot, blk_a, blk_b = env.scene["robot"], env.scene["object"], env.scene["block_b"]
    origins = env.scene.env_origins
    widx = robot.data.body_names.index("left_wrist_yaw_link")

    if args_cli.video:
        mid = torch.tensor([-0.20, 0.36, 0.75], device=dev)
        env.scene["demo_cam"].set_world_poses_from_view((mid + torch.tensor([0.30, 0.50, 0.32], device=dev)).repeat(N, 1) + origins, mid.repeat(N, 1) + origins)

    q0 = torch.tensor([0.0, 0.0, 0.70710678, 0.70710678], device=dev).repeat(N, 1)  # xyzw rest wrist orientation
    roll = torch.deg2rad(torch.full((N,), float(GRASP["roll"]), device=dev))
    q_grasp = quat_mul(quat_from_angle_axis(roll, torch.tensor([0.0, 1.0, 0.0], device=dev).repeat(N, 1)), q0)
    p_local = torch.tensor([GRASP["px"], -GRASP["py"], GRASP["pz"]], device=dev).repeat(N, 1)
    grasp_off = -quat_apply(q_grasp, p_local)
    palm_n = quat_apply(q_grasp, torch.tensor([0.0, -1.0, 0.0], device=dev).repeat(N, 1))
    pre_off = grasp_off - 0.10 * palm_n
    pre_off[:, 2] = torch.maximum(pre_off[:, 2], grasp_off[:, 2])

    phase = torch.full((N,), ABOVE, dtype=torch.long, device=dev)
    t_in = torch.zeros(N, dtype=torch.long, device=dev)
    corr = torch.zeros(N, 3, device=dev)
    carry_goal = torch.zeros(N, 3, device=dev)
    close_pos = torch.zeros(N, 3, device=dev)
    release_pos = torch.zeros(N, 3, device=dev)
    held_z0 = torch.zeros(N, device=dev)
    ever_lifted = torch.zeros(N, dtype=torch.bool, device=dev)
    snap_taken = torch.zeros(N, dtype=torch.bool, device=dev)
    snap_joint = torch.zeros(N, robot.data.joint_pos.torch.shape[1], device=dev)
    snap_blocks = torch.zeros(N, 2, 7, device=dev)
    grip = -torch.ones(N, device=dev)
    frames = []
    tmo = torch.tensor(TIMEOUT, device=dev)
    t0 = time.time()
    rec_obs, rec_act, rec_live = [], [], []
    sigma = torch.rand(N, 1, device=dev) * args_cli.act_noise
    a0 = (blk_a.data.root_pos_w.torch - origins).clone()
    print("robot root quat (xyzw):", [round(v, 3) for v in robot.data.root_quat_w.torch[0].tolist()])

    for step in range(args_cli.max_steps):
        eef = robot.data.body_pos_w.torch[:, widx] - origins
        eq = robot.data.body_quat_w.torch[:, widx]
        pa = blk_a.data.root_pos_w.torch - origins
        pb = blk_b.data.root_pos_w.torch - origins
        stack_z = pb[:, 2] + BLOCK
        target = eef.clone()

        m = phase == ABOVE
        target[m] = (a0 + pre_off)[m]  # aim at the block's spawn pose, so a nudge can't drag the approach
        target[m, 2] += (0.12 * (1 - (t_in.float() - 40) / 40).clamp(0.0, 1.0))[m]  # high transit, then drop
        grip[m] = -1
        m = phase == DESCEND
        f = (t_in.float() / 35).clamp(max=1.0).unsqueeze(1)
        target[m] = (pa + pre_off + f * (grasp_off - pre_off))[m]
        m = phase == CLOSE
        target[m] = close_pos[m]
        grip[m] = (-1 + 2 * (t_in.float() / 30).clamp(max=1.0))[m]
        m = phase == LIFT
        target[m] = close_pos[m]
        target[m, 2] += (0.10 * (t_in.float() / 40).clamp(max=1.0))[m]
        grip[m] = 1
        goal_blk = torch.cat([pb[:, :2], (stack_z + 0.10).unsqueeze(1)], dim=1)
        ml = phase == LOWER
        goal_blk[ml, 2] = stack_z[ml] - 0.002
        mcl = (phase == CARRY) | ml
        sv = goal_blk - carry_goal
        carry_goal[mcl] = (carry_goal + sv * (0.003 / torch.linalg.norm(sv, dim=1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0))[mcl]
        target[mcl] = (carry_goal + (eef - pa))[mcl]
        grip[mcl] = 1
        m = phase == RELEASE
        target[m] = release_pos[m]
        grip[m] = (1 - 2 * (t_in.float() / 20).clamp(max=1.0))[m]
        m = phase == RETREAT
        f = (t_in.float() / 25).clamp(max=1.0).unsqueeze(1)
        target[m] = (release_pos + f * (-0.06 * palm_n + torch.tensor([0, 0, 0.04], device=dev)))[m]
        grip[m] = -1
        m = phase == DONE
        target[m] = eef[m]

        mc = phase <= LOWER
        corr[mc] = (corr[mc] + args_cli.corr_gain * (target - eef)[mc]).clamp(-0.05, 0.05)
        pos_err, rot_err = compute_pose_error(eef, eq, target + corr, q_grasp, rot_error_type="axis_angle")  # (source, target)
        act = torch.zeros(N, 7, device=dev)
        # the diff-IK action takes deltas in the robot ROOT frame (the G1 pelvis is yawed in the world)
        rq = robot.data.root_quat_w.torch
        act[:, 0:3] = (quat_apply_inverse(rq, pos_err) / POS_SCALE).clamp(-1, 1)
        act[:, 3:6] = (quat_apply_inverse(rq, rot_err) / ROT_SCALE).clamp(-1, 1)
        act[:, 6] = grip
        exe = act.clone()
        if args_cli.act_noise > 0:
            exe[:, :6] = (exe[:, :6] + sigma * torch.randn(N, 6, device=dev)).clamp(-1, 1)
        if args_cli.record:
            rec_obs.append(obs["policy"].clone())
            rec_act.append(act.clone())
            rec_live.append(phase != DONE)
        obs, *_ = env.step(exe)
        t_in += 1
        if step % 20 == 0 and not args_cli.quiet:
            e_now = robot.data.body_pos_w.torch[:, widx] - origins
            pe, re = compute_pose_error(e_now, robot.data.body_quat_w.torch[:, widx], target, q_grasp, rot_error_type="axis_angle")
            moved = torch.linalg.norm((blk_a.data.root_pos_w.torch - origins - a0)[:, :2], dim=1)
            print(f"DBG step {step:3d} phase={torch.bincount(phase, minlength=9).tolist()} pos_err med={pe.norm(dim=1).median():.3f} max={pe.norm(dim=1).max():.3f} "
                  f"rot_err med={re.norm(dim=1).median():.3f} blockA_moved>2cm={int((moved > 0.02).sum())} act_abs={act[:, :6].abs().mean():.2f} eef0={[round(v, 3) for v in e_now[0].tolist()]} "
                  f"tgt0={[round(v, 3) for v in target[0].tolist()]} corr0={[round(v, 3) for v in corr[0].tolist()]}")
        if args_cli.video:
            frames.append(env.scene["demo_cam"].data.output["rgb"][0, ..., :3].clone().cpu())

        eef = robot.data.body_pos_w.torch[:, widx] - origins
        pa = blk_a.data.root_pos_w.torch - origins
        pb = blk_b.data.root_pos_w.torch - origins
        d = torch.linalg.norm(eef - target, dim=1)
        nxt = phase.clone()
        nxt[(phase == ABOVE) & (((d < 0.02) & (t_in > 85)) | (t_in >= tmo[ABOVE]))] = DESCEND
        to_close = (phase == DESCEND) & (((d < 0.012) & (t_in > 40)) | (t_in >= tmo[DESCEND]))
        close_pos[to_close] = target[to_close]
        nxt[to_close] = CLOSE
        cd = (phase == CLOSE) & (t_in >= tmo[CLOSE])
        held_z0[cd] = pa[cd, 2]
        nxt[cd] = LIFT
        lifted = pa[:, 2] > held_z0 + 0.05
        ever_lifted |= lifted & (phase >= LIFT)
        tc = (phase == LIFT) & lifted & (t_in > 25)
        carry_goal[tc] = pa[tc]
        nxt[tc] = CARRY
        bxy = torch.linalg.norm((pa - pb)[:, :2], dim=1)
        tl = (phase == CARRY) & (bxy < 0.01) & ((pa[:, 2] - pb[:, 2]) > BLOCK + 0.03)
        # reverse-curriculum snapshot: block held and aligned above the red block
        tk = tl & ~snap_taken
        snap_joint[tk] = robot.data.joint_pos.torch[tk]
        snap_blocks[tk, 0] = torch.cat([pa, blk_a.data.root_quat_w.torch], dim=1)[tk]
        snap_blocks[tk, 1] = torch.cat([pb, blk_b.data.root_quat_w.torch], dim=1)[tk]
        snap_taken |= tk
        nxt[tl] = LOWER
        tr = (phase == LOWER) & ((((pa[:, 2] - pb[:, 2]) < BLOCK + 0.006) & (bxy < 0.008)) | (t_in >= tmo[LOWER]))
        release_pos[tr] = eef[tr]
        nxt[tr] = RELEASE
        nxt[(phase == RELEASE) & (t_in >= tmo[RELEASE])] = RETREAT
        nxt[(phase == RETREAT) & (t_in >= tmo[RETREAT])] = DONE
        fail = ((phase == LIFT) & (t_in >= tmo[LIFT])) | ((phase == CARRY) & (t_in >= tmo[CARRY]))
        fail |= (phase >= CARRY) & (phase <= LOWER) & (pa[:, 2] < held_z0 + 0.01)
        nxt[fail] = DONE
        t_in[nxt != phase] = 0
        phase = nxt
        if bool((phase == DONE).all()):
            break
    steps = step + 1
    dt = time.time() - t0

    for _ in range(30):
        act[:, :6] = 0
        act[:, 6] = -1
        env.step(act)
    pa = blk_a.data.root_pos_w.torch - origins
    pb = blk_b.data.root_pos_w.torch - origins
    dxy = torch.linalg.norm((pa - pb)[:, :2], dim=1)
    dz = pa[:, 2] - pb[:, 2] - BLOCK
    vel = torch.linalg.norm(blk_a.data.root_vel_w.torch[:, :3], dim=1)
    success = (dxy < 0.02) & (dz.abs() < 0.01) & (vel < 0.05) & ever_lifted
    print(f"RL-SPACE EXPERT: success {int(success.sum())}/{N}  lifted {int(ever_lifted.sum())}/{N}  "
          f"snapshots {int((snap_taken & success).sum())}  throughput {N * steps / dt:.0f} env-steps/s ({steps} steps, {dt:.1f}s)")
    print("phase histogram at end:", torch.bincount(phase, minlength=9).tolist())
    if args_cli.record:
        O = torch.stack(rec_obs, 1)[success]  # (S, T, obs)
        A = torch.stack(rec_act, 1)[success]
        L = torch.stack(rec_live, 1)[success]
        torch.save({"obs": O[L].cpu(), "act": A[L].cpu(), "n_episodes": int(success.sum())}, args_cli.record)
        print(f"recorded {int(L.sum())} (obs, action) pairs from {int(success.sum())} successful episodes -> {args_cli.record}")
    if args_cli.snapshots:
        keep = snap_taken & success
        torch.save({"joint_pos": snap_joint[keep].cpu(), "cube_pose": snap_blocks[keep].cpu()}, args_cli.snapshots)
        print(f"saved {int(keep.sum())} snapshots -> {args_cli.snapshots}")
    if args_cli.video and frames:
        import imageio

        with imageio.get_writer(args_cli.video, fps=30, codec="libx264", quality=8) as w:
            for fr in frames:
                w.append_data(fr.numpy().astype("uint8"))
        print(f"VIDEO {args_cli.video} frames={len(frames)}")

simulation_app.close()
