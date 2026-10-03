"""Stage-2 scripted expert: picks up block C and places it on top of block A (which the Stage-2
env cfg's own reset events already seat on block B -- see g1_block_stack_3stage2_rl_env_cfg.py's
docstring).

First version reused Stage 1's phase machine almost verbatim (fly a wide pre-grasp hover, descend
in a blended line) with C_POS=(0.0, 0.35) only 14cm from B_POS -- the hand's approach swept close
enough to the standing A-on-B tower to knock it over before ever reaching C (confirmed on video:
tower standing at step 100-120, visibly toppling by step 140-160). Two independent fixes, per
diagnosis: (1) C_POS moved to (0.05, 0.25), 22cm from B_POS, reach-checked separately
(reach_check_cpos.py: 0.76-0.86cm pos error at roll 0-20deg); (2) the motion itself now never moves
laterally at a height anywhere near the tower, regardless of how far away C is placed -- every pick
and every place is RISE (vertical only, to a safe height above the tower top) -> TRANSIT (lateral
only, frozen at the safe height) -> DESCEND (vertical only, onto the target). Same fix pattern the
Franka expert on this box already uses for exactly this failure mode.

Same grasp model as Stage 1 and stack_expert_rl.py -- palm-pocket pinch, real contact physics, no
kinematic attach. Reuses the Stage 1 GRASP defaults (C and A are both standard 4.5cm blocks; only
table XY differs, which the IK handles regardless of absolute position). Also tracks whether A
moved >1cm off its seated pose at any point before Stage 2's own placement interaction with it
("tower disturbed") -- the quantitative version of the exact failure the frame evidence showed, so
a future collision shows up in the success-rate numbers even without a human watching the video.

Three uses, same as Stage 1's expert: (1) proves the task is solvable at all, (2) validates the
tolerance/timing constants transfer from Stage 1 without retuning, (3) saves snapshots (robot
joints + block_c pose only -- block_a/block_b are restored by the env's own normal reset events,
not by the snapshot) for the eventual reverse-curriculum / handoff-state training mix.
"""

import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-Stage2-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--grasp", type=str, default='{"roll":20,"px":0.11,"py":0.05,"pz":-0.02}')
parser.add_argument("--max_steps", type=int, default=480)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--corr_gain", type=float, default=0.1)
parser.add_argument("--safe_z", type=float, default=0.90, help="rise/transit height, world z -- above the tower top (0.81) with hand clearance")
parser.add_argument("--snapshots", type=str, default="")
parser.add_argument("--record", type=str, default="", help="save (policy obs, clean expert action) pairs from successful episodes, for bc_train.py")
parser.add_argument("--video", type=str, default="")
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
if args_cli.video:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    env_cfg.scene.demo_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCam", update_period=0.0, height=540, width=960, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
    )
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

# Every pick and every place is a strict RISE (vertical only) -> TRANSIT (lateral only, frozen at
# safe_z) -> DESCEND/LOWER (vertical-ish, blending the final grasp/place offset in) sequence -- no
# phase ever moves laterally at a height near the tower. See module docstring for why.
RISE_PICK, TRANSIT_PICK, DESCEND, CLOSE, LIFT, RISE_CARRY, TRANSIT_CARRY, LOWER, RELEASE, RETREAT, DONE = range(11)
TIMEOUT = [40, 70, 70, 45, 70, 40, 70, 110, 35, 40, 10**9]

with torch.inference_mode():
    obs, _ = env.reset()
    dev, N = env.device, env.num_envs
    robot = env.scene["robot"]
    blk_pick, blk_base = env.scene["block_c"], env.scene["object"]  # pick C, place on A
    origins = env.scene.env_origins
    widx = robot.data.body_names.index("left_wrist_yaw_link")

    if args_cli.video:
        mid = torch.tensor([-0.10, 0.36, 0.78], device=dev)
        env.scene["demo_cam"].set_world_poses_from_view((mid + torch.tensor([0.30, 0.50, 0.32], device=dev)).repeat(N, 1) + origins, mid.repeat(N, 1) + origins)

    q0 = torch.tensor([0.0, 0.0, 0.70710678, 0.70710678], device=dev).repeat(N, 1)
    roll = torch.deg2rad(torch.full((N,), float(GRASP["roll"]), device=dev))
    q_grasp = quat_mul(quat_from_angle_axis(roll, torch.tensor([0.0, 1.0, 0.0], device=dev).repeat(N, 1)), q0)
    p_local = torch.tensor([GRASP["px"], -GRASP["py"], GRASP["pz"]], device=dev).repeat(N, 1)
    grasp_off = -quat_apply(q_grasp, p_local)
    palm_n = quat_apply(q_grasp, torch.tensor([0.0, -1.0, 0.0], device=dev).repeat(N, 1))
    pre_off = grasp_off - 0.10 * palm_n
    pre_off[:, 2] = torch.maximum(pre_off[:, 2], grasp_off[:, 2])

    phase = torch.full((N,), RISE_PICK, dtype=torch.long, device=dev)
    t_in = torch.zeros(N, dtype=torch.long, device=dev)
    corr = torch.zeros(N, 3, device=dev)
    close_pos = torch.zeros(N, 3, device=dev)
    release_pos = torch.zeros(N, 3, device=dev)
    phase_start = torch.zeros(N, 3, device=dev)  # eef pos captured at the moment each phase begins
    held_z0 = torch.zeros(N, device=dev)
    ever_lifted = torch.zeros(N, dtype=torch.bool, device=dev)
    snap_taken = torch.zeros(N, dtype=torch.bool, device=dev)
    snap_joint = torch.zeros(N, robot.data.joint_pos.torch.shape[1], device=dev)
    snap_block_c = torch.zeros(N, 7, device=dev)
    grip = -torch.ones(N, device=dev)
    frames = []
    rec_obs, rec_act, rec_live = [], [], []
    tmo = torch.tensor(TIMEOUT, device=dev)
    t0 = time.time()
    c0 = (blk_pick.data.root_pos_w.torch - origins).clone()
    a0 = (blk_base.data.root_pos_w.torch - origins).clone()
    phase_start[:] = robot.data.body_pos_w.torch[:, widx] - origins
    # "tower disturbed": A moved >1cm off its seated pose at any point before Stage 2's own
    # intentional placement interaction with it (phase >= LOWER) -- the quantitative trace of the
    # exact failure the first C_POS/motion design produced (see module docstring).
    tower_disturbed = torch.zeros(N, dtype=torch.bool, device=dev)
    print("robot root quat (xyzw):", [round(v, 3) for v in robot.data.root_quat_w.torch[0].tolist()])

    for step in range(args_cli.max_steps):
        eef = robot.data.body_pos_w.torch[:, widx] - origins
        eq = robot.data.body_quat_w.torch[:, widx]
        pc = blk_pick.data.root_pos_w.torch - origins
        pa = blk_base.data.root_pos_w.torch - origins
        stack_z = pa[:, 2] + BLOCK
        target = eef.clone()
        safe_z = args_cli.safe_z

        # -- pick: rise (vertical) -> transit (lateral, at safe_z) -> descend onto C --
        m = phase == RISE_PICK
        target[m] = phase_start[m]
        target[m, 2] = safe_z
        grip[m] = -1
        m = phase == TRANSIT_PICK
        target[m, 0] = (c0 + pre_off)[m, 0]
        target[m, 1] = (c0 + pre_off)[m, 1]
        target[m, 2] = safe_z
        grip[m] = -1
        m = phase == DESCEND
        f = (t_in.float() / tmo[DESCEND].float()).clamp(max=1.0).unsqueeze(1)
        target[m] = (phase_start + f * ((pc + grasp_off) - phase_start))[m]
        m = phase == CLOSE
        target[m] = close_pos[m]
        grip[m] = (-1 + 2 * (t_in.float() / 30).clamp(max=1.0))[m]
        m = phase == LIFT
        target[m] = close_pos[m]
        target[m, 2] += (0.10 * (t_in.float() / 40).clamp(max=1.0))[m]
        grip[m] = 1

        # -- carry: rise (vertical, still gripping) -> transit (lateral, at safe_z) -> lower onto A --
        # Real contact physics, no kinematic attach: we command the WRIST, the held block follows.
        # (eef - pc) is the current wrist-minus-block offset established by the grip -- used to
        # convert "put the block above A" into the equivalent wrist target.
        m = phase == RISE_CARRY
        target[m] = phase_start[m]  # wrist frozen at its own xy from phase entry, rising only
        target[m, 2] = safe_z
        grip[m] = 1
        m = phase == TRANSIT_CARRY
        target[m, 0] = pa[m, 0] + (eef - pc)[m, 0]
        target[m, 1] = pa[m, 1] + (eef - pc)[m, 1]
        target[m, 2] = safe_z
        grip[m] = 1
        m = phase == LOWER
        goal = torch.cat([pa[:, :2], (stack_z - 0.002).unsqueeze(1)], dim=1) + (eef - pc)
        f = (t_in.float() / tmo[LOWER].float()).clamp(max=1.0).unsqueeze(1)
        target[m] = (phase_start + f * (goal - phase_start))[m]
        grip[m] = 1
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
        pos_err, rot_err = compute_pose_error(eef, eq, target + corr, q_grasp, rot_error_type="axis_angle")
        act = torch.zeros(N, 7, device=dev)
        rq = robot.data.root_quat_w.torch
        act[:, 0:3] = (quat_apply_inverse(rq, pos_err) / POS_SCALE).clamp(-1, 1)
        act[:, 3:6] = (quat_apply_inverse(rq, rot_err) / ROT_SCALE).clamp(-1, 1)
        act[:, 6] = grip
        if args_cli.record:
            rec_obs.append(obs["policy"].clone())
            rec_act.append(act.clone())
            rec_live.append(phase != DONE)
        obs, *_ = env.step(act)
        t_in += 1
        if step % 20 == 0 and not args_cli.quiet:
            e_now = robot.data.body_pos_w.torch[:, widx] - origins
            pe, re = compute_pose_error(e_now, robot.data.body_quat_w.torch[:, widx], target, q_grasp, rot_error_type="axis_angle")
            moved = torch.linalg.norm((blk_pick.data.root_pos_w.torch - origins - c0)[:, :2], dim=1)
            print(f"DBG step {step:3d} phase={torch.bincount(phase, minlength=11).tolist()} pos_err med={pe.norm(dim=1).median():.3f} max={pe.norm(dim=1).max():.3f} "
                  f"rot_err med={re.norm(dim=1).median():.3f} blockC_moved>2cm={int((moved > 0.02).sum())} act_abs={act[:, :6].abs().mean():.2f} eef0={[round(v, 3) for v in e_now[0].tolist()]} "
                  f"tgt0={[round(v, 3) for v in target[0].tolist()]} corr0={[round(v, 3) for v in corr[0].tolist()]}")
        if args_cli.video:
            frames.append(env.scene["demo_cam"].data.output["rgb"][0, ..., :3].clone().cpu())

        eef = robot.data.body_pos_w.torch[:, widx] - origins
        pc = blk_pick.data.root_pos_w.torch - origins
        pa = blk_base.data.root_pos_w.torch - origins
        d = torch.linalg.norm(eef - target, dim=1)

        # A moved off its seated pose before Stage 2 intentionally interacts with it (phase < LOWER)
        a_off = torch.linalg.norm(pa - a0, dim=1)
        tower_disturbed |= (phase < LOWER) & (a_off > 0.01)

        nxt = phase.clone()
        nxt[(phase == RISE_PICK) & (((d < 0.02) & (t_in > 15)) | (t_in >= tmo[RISE_PICK]))] = TRANSIT_PICK
        nxt[(phase == TRANSIT_PICK) & (((d < 0.02) & (t_in > 40)) | (t_in >= tmo[TRANSIT_PICK]))] = DESCEND
        # DESCEND is a timed ramp (phase_start -> pc+grasp_off over tmo[DESCEND] steps), not a
        # point target -- the controller tracks each step's moving waypoint closely well before the
        # ramp itself is done, so an early-exit on "d small" (copied from Stage 1's much shorter,
        # fixed-35-step blend) would fire as soon as tracking catches up, not when the hand has
        # actually finished descending. Exactly this bug sent the hand to CLOSE ~10cm above the
        # block (it closed on air, 0/16 lifted) before being caught here. Only exit on full timeout.
        to_close = (phase == DESCEND) & (t_in >= tmo[DESCEND])
        close_pos[to_close] = target[to_close]
        nxt[to_close] = CLOSE
        cd = (phase == CLOSE) & (t_in >= tmo[CLOSE])
        held_z0[cd] = pc[cd, 2]
        nxt[cd] = LIFT
        lifted = pc[:, 2] > held_z0 + 0.05
        ever_lifted |= lifted & (phase >= LIFT)
        tc = (phase == LIFT) & lifted & (t_in > 25)
        nxt[tc] = RISE_CARRY
        nxt[(phase == RISE_CARRY) & (((d < 0.02) & (t_in > 15)) | (t_in >= tmo[RISE_CARRY]))] = TRANSIT_CARRY
        bxy = torch.linalg.norm((pc - pa)[:, :2], dim=1)
        tl = (phase == TRANSIT_CARRY) & (((d < 0.02) & (t_in > 40)) | (t_in >= tmo[TRANSIT_CARRY]))
        tk = tl & ~snap_taken
        snap_joint[tk] = robot.data.joint_pos.torch[tk]
        snap_block_c[tk] = torch.cat([pc, blk_pick.data.root_quat_w.torch], dim=1)[tk]
        snap_taken |= tk
        nxt[tl] = LOWER
        tr = (phase == LOWER) & ((((pc[:, 2] - pa[:, 2]) < BLOCK + 0.006) & (bxy < 0.008)) | (t_in >= tmo[LOWER]))
        release_pos[tr] = eef[tr]
        nxt[tr] = RELEASE
        nxt[(phase == RELEASE) & (t_in >= tmo[RELEASE])] = RETREAT
        nxt[(phase == RETREAT) & (t_in >= tmo[RETREAT])] = DONE
        fail = ((phase == LIFT) & (t_in >= tmo[LIFT])) | ((phase == TRANSIT_CARRY) & (t_in >= tmo[TRANSIT_CARRY]))
        fail |= (phase >= RISE_CARRY) & (phase <= LOWER) & (pc[:, 2] < held_z0 + 0.01)
        nxt[fail] = DONE
        changed = nxt != phase
        t_in[changed] = 0
        phase_start[changed] = eef[changed]
        phase = nxt
        if bool((phase == DONE).all()):
            break
    steps = step + 1
    dt = time.time() - t0

    for _ in range(30):
        act[:, :6] = 0
        act[:, 6] = -1
        env.step(act)
    pc = blk_pick.data.root_pos_w.torch - origins
    pa = blk_base.data.root_pos_w.torch - origins
    dxy = torch.linalg.norm((pc - pa)[:, :2], dim=1)
    dz = pc[:, 2] - pa[:, 2] - BLOCK
    vel = torch.linalg.norm(blk_pick.data.root_vel_w.torch[:, :3], dim=1)
    success = (dxy < 0.025) & (dz.abs() < 0.01) & (vel < 0.05) & ever_lifted & ~tower_disturbed
    print(f"STAGE-2 RL-SPACE EXPERT: success {int(success.sum())}/{N}  lifted {int(ever_lifted.sum())}/{N}  "
          f"tower_disturbed {int(tower_disturbed.sum())}/{N}  "
          f"snapshots {int((snap_taken & success).sum())}  throughput {N * steps / dt:.0f} env-steps/s ({steps} steps, {dt:.1f}s)")
    print("phase histogram at end:", torch.bincount(phase, minlength=11).tolist())
    if args_cli.snapshots:
        keep = snap_taken & success
        torch.save({"joint_pos": snap_joint[keep].cpu(), "cube_pose": snap_block_c[keep].cpu().unsqueeze(1)}, args_cli.snapshots)
        print(f"saved {int(keep.sum())} snapshots (robot joints + block_c pose only; A/B restored by the "
              f"env's own reset events) -> {args_cli.snapshots}")
    if args_cli.record:
        O = torch.stack(rec_obs, 1)[success]  # (S, T, obs)
        A = torch.stack(rec_act, 1)[success]
        L = torch.stack(rec_live, 1)[success]
        torch.save({"obs": O[L].cpu(), "act": A[L].cpu(), "n_episodes": int(success.sum())}, args_cli.record)
        print(f"recorded {int(L.sum())} (obs, action) pairs from {int(success.sum())} successful episodes -> {args_cli.record}")
    if args_cli.video and frames:
        import imageio

        with imageio.get_writer(args_cli.video, fps=30, codec="libx264", quality=8) as w:
            for fr in frames:
                w.append_data(fr.numpy().astype("uint8"))
        print(f"VIDEO {args_cli.video} frames={len(frames)}")

simulation_app.close()
