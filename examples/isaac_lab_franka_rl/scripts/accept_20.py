"""Franka 3-cube chained-RL acceptance harness: N sequential, single-env, deterministic episodes,
one different seed each, judged against the env's own chained-success definition on the last
pre-auto-reset state. Exists so the moment a candidate stage-1/stage-2 checkpoint pair "looks good"
on the aggregate eval_chain.py/record_chain.py metrics, there is one command that produces the real
20/20-style proof run (and, with --video, one caption-burned mp4 per episode) instead of re-deriving
the judging logic under time pressure.

Adapts eval_chain.py (the chained stage-1 -> stage-2 handoff, the `states()` success/failure
bookkeeping, and the failure-attribution buckets) and record_chain.py (the 1280x720 demo camera at
the Brev viewer pose, and mp4 writing) -- see ACCEPTANCE.md for the protocol this implements and why
each piece of it is there.

**The one correctness requirement this whole file exists to get right:** Isaac Lab auto-resets an
env *inside* the very `step()` call whose `dones` it returns True on -- so reading simulation state
(`u.scene[...]`) on any later iteration already sees the *next* episode, not the one that just ended.
(This produced a real bogus 0/N result on the G1 example before it was caught.) The fix, copied
exactly from eval_chain.py: compute `states()` at the *top* of each iteration, before that
iteration's `env.step()` runs, and latch the "_last" / failure-attribution values from that reading.
Then, specific to this file's one-episode-at-a-time structure (eval_chain.py runs many envs in
parallel and must keep masking out ones that already finished): the moment `dones` comes back True,
stop the step loop immediately with `break`, before ever reading `states()` again. The latched
values already hold exactly the last state the episode was confirmed still live in -- one physics
step before the terminating action's effect was committed, never the reset scene.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-RL-FullStack-Snap-Sparse-v0")
parser.add_argument("--stage1", type=str, required=True)
parser.add_argument("--stage2", type=str, required=True)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed_base", type=int, default=100, help="episode i uses seed seed_base + i")
parser.add_argument("--handoff_steps", type=int, default=10)
parser.add_argument("--start_episode", type=int, default=0, help="resume at 0-based episode index (seeds, numbering and video names follow the global index); the final ACCEPT denominator then counts only episodes run here")
parser.add_argument("--video", action="store_true", help="write one captioned mp4 per episode")
parser.add_argument("--video_dir", type=str, default="accept_20_videos")
parser.add_argument("--label", type=str, default="Franka 3-cube policy", help="caption text prefix burned into the video (say what the policy is, e.g. BC vs RL)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = args_cli.video
simulation_app = AppLauncher(args_cli).app

import copy
import os

import torch
from rsl_rl.runners import OnPolicyRunner

import gymnasium as gym

from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.stack.mdp.robosuite_rewards import _in_contact
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env_cfg.events.reset_from_snapshot = None  # natural, randomized resets only -- no expert start states
s2 = copy.deepcopy(env_cfg.observations.policy)
roles = {"cube_1_cfg": SceneEntityCfg("cube_2"), "cube_2_cfg": SceneEntityCfg("cube_3"), "cube_3_cfg": SceneEntityCfg("cube_1")}
for term in ("object", "cube_positions", "cube_orientations"):
    getattr(s2, term).params.update(roles)
env_cfg.observations.policy_s2 = s2

# Extra contact sensors for failure attribution -- identical set to eval_chain.py's.
from isaaclab.sensors import ContactSensorCfg  # noqa: E402

env_cfg.scene.cube_1.spawn = env_cfg.scene.cube_1.spawn.replace(activate_contact_sensors=True)
from isaaclab_tasks.contrib.stack.config.franka.stack_ik_rel_rl_env_cfg import _BLUE, _GREEN, _RED, _ROBOT_LINK  # noqa: E402

_R = _ROBOT_LINK
_C = {1: _BLUE, 2: _RED, 3: _GREEN}
env_cfg.scene.x_green_blue = ContactSensorCfg(prim_path=_C[3], filter_prim_paths_expr=[_C[1]])
env_cfg.scene.x_hand_red = ContactSensorCfg(prim_path=_R + "panda_hand", filter_prim_paths_expr=[_C[2]])
env_cfg.scene.x_hand_blue = ContactSensorCfg(prim_path=_R + "panda_hand", filter_prim_paths_expr=[_C[1]])
env_cfg.scene.x_lf_blue = ContactSensorCfg(prim_path=_R + "panda_leftfinger", filter_prim_paths_expr=[_C[1]])
env_cfg.scene.x_rf_blue = ContactSensorCfg(prim_path=_R + "panda_rightfinger", filter_prim_paths_expr=[_C[1]])

if args_cli.video:
    # Same 1280x720 demo camera, at the same Brev viewer pose, as record_chain.py.
    import isaaclab.sim as sim_utils  # noqa: E402
    from isaaclab.sensors import CameraCfg  # noqa: E402

    EYE, LOOKAT = (1.0, 0.45, 0.42), (0.5, 0.0, 0.06)
    env_cfg.scene.demo_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/demo_cam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)),
    )

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
dev = u.device
T = int(u.max_episode_length)

if args_cli.video:
    os.makedirs(args_cli.video_dir, exist_ok=True)
    u.scene["demo_cam"].set_world_poses_from_view(
        torch.tensor([EYE], device=dev) + u.scene.env_origins, torch.tensor([LOOKAT], device=dev) + u.scene.env_origins,
    )

pols = []
for ck in (args_cli.stage1, args_cli.stage2):
    r = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    r.load(ck, map_location=agent_cfg.device)
    pols.append(r.get_inference_policy(device=dev))

LIFT_Z, THR = 0.0403, 0.01


def states():
    """Verbatim from eval_chain.py (n=1 here, but the code is already fully vectorized, so nothing
    about it needed to change) -- reads the CURRENT live scene, which is only ever safe to call at
    the top of an iteration, before that iteration's env.step()."""
    oz = u.scene.env_origins[:, 2]
    red_z = u.scene["cube_2"].data.root_pos_w.torch[:, 2] - oz
    green_z = u.scene["cube_3"].data.root_pos_w.torch[:, 2] - oz
    g_red = _in_contact(u, "left_finger_red_contact", THR) & _in_contact(u, "right_finger_red_contact", THR)
    g_green = _in_contact(u, "left_finger_green_contact", THR) & _in_contact(u, "right_finger_green_contact", THR)
    s1 = (red_z > LIFT_Z) & _in_contact(u, "red_blue_contact", THR) & ~g_red
    tower = s1 & (green_z > LIFT_Z) & _in_contact(u, "green_red_contact", THR) & ~g_green
    aux = dict(
        g_green=g_green, g_red=g_red, green_z=green_z, red_z=red_z,
        g_red_any=_in_contact(u, "left_finger_red_contact", THR) | _in_contact(u, "right_finger_red_contact", THR),
        green_on_red=(green_z > LIFT_Z) & _in_contact(u, "green_red_contact", THR),
        x_green_red=_in_contact(u, "green_red_contact", THR), x_green_blue=_in_contact(u, "x_green_blue", THR),
        x_hand_red=_in_contact(u, "x_hand_red", THR), x_hand_blue=_in_contact(u, "x_hand_blue", THR),
        x_finger_blue=_in_contact(u, "x_lf_blue", THR) | _in_contact(u, "x_rf_blue", THR),
    )
    return s1, tower, aux


def classify(handed, lost_t, lost_green_held, s1_last, tower_last, end_green_held, end_green_on_red, end_red_z) -> str:
    """One human-readable failure (or success) reason per episode -- the per-episode half of
    eval_chain.py's aggregate FAILURES breakdown."""
    if tower_last:
        return "tower standing (success)"
    if not handed:
        return "never handed off to stage 2"
    if lost_t >= 0:
        held = "green was already held" if lost_green_held else "green not yet picked up"
        return f"red knocked off blue {lost_t} step(s) after handoff ({held})"
    if not s1_last:
        return "red not on blue at end (on table)" if end_red_z < LIFT_Z else "red not on blue at end (green stacked on it instead)"
    if end_green_on_red and end_green_held:
        return "green on red but still held at end"
    if not end_green_on_red and end_green_held:
        return "green held elsewhere at end, never placed"
    return "green loose, not on red, at end"


results = []
for i in range(args_cli.start_episode, args_cli.episodes):
    seed = args_cli.seed_base + i
    torch.manual_seed(seed)
    # Explicit, seeded reset for THIS episode -- deliberately not relying on the auto-reset the
    # previous episode's final env.step() already performed, since that reset's randomness was
    # drawn before this seed was set. env.reset()'s return value isn't used (its shape isn't the
    # dict every other script here reads); get_observations() right after is the same pattern
    # eval_chain.py/record_chain.py use to obtain the dict-shaped {"policy", "policy_s2"} observation.
    env.reset()
    obs = env.get_observations()

    handed = torch.zeros(1, dtype=torch.bool, device=dev)
    s1_run = torch.zeros(1, dtype=torch.long, device=dev)
    hand_t = -1
    s1_ever = torch.zeros(1, dtype=torch.bool, device=dev)
    tower_ever = torch.zeros(1, dtype=torch.bool, device=dev)
    first_tower = -1
    s1_last = torch.zeros(1, dtype=torch.bool, device=dev)
    tower_last = torch.zeros(1, dtype=torch.bool, device=dev)
    lost_t = -1
    lost_green_held = False
    end_green_held = torch.zeros(1, dtype=torch.bool, device=dev)
    end_green_on_red = torch.zeros(1, dtype=torch.bool, device=dev)
    end_red_z = torch.zeros(1, device=dev)
    prev_s1 = torch.zeros(1, dtype=torch.bool, device=dev)
    prev_aux = None
    early = False
    frames = []

    with torch.no_grad():
        for t in range(T):
            s1, tower, aux = states()

            s1_run = torch.where(s1, s1_run + 1, torch.zeros_like(s1_run))
            new = ~handed & (s1_run >= args_cli.handoff_steps)
            if bool(new.item()) and hand_t < 0:
                hand_t = t
            handed |= new
            if bool(tower.item()) and first_tower < 0:
                first_tower = t
            s1_ever |= s1
            tower_ever |= tower

            lost = handed & prev_s1 & ~s1 & ~aux["g_red"] & (lost_t < 0) & (hand_t >= 0) & (hand_t < t)
            if bool(lost.item()) and prev_aux is not None:
                lost_green_held = bool(prev_aux["g_green"].item())
                lost_t = t - hand_t
            prev_s1, prev_aux = s1, aux

            # Always update -- this iteration hasn't stepped (and so can't have auto-reset) yet.
            end_green_held, end_green_on_red, end_red_z = aux["g_green"], aux["green_on_red"], aux["red_z"]
            s1_last, tower_last = s1, tower

            if args_cli.video:
                frames.append(u.scene["demo_cam"].data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy())

            obs2 = obs.clone()
            obs2["policy"] = obs["policy_s2"]
            action = pols[1](obs2) if bool(handed.item()) else pols[0](obs)
            obs, _, dones, extras = env.step(action)
            for p in pols:
                p.reset(dones)

            if bool(dones.item()):
                early = not bool(extras.get("time_outs", torch.zeros_like(dones))[0].item())
                break  # the state latched above is the last one this episode was confirmed live in

    success = bool(tower_last.item())
    reason = classify(
        bool(handed.item()), lost_t, lost_green_held, bool(s1_last.item()), success,
        bool(end_green_held.item()), bool(end_green_on_red.item()), float(end_red_z.item()),
    )
    results.append({
        "episode": i + 1, "seed": seed, "success": success, "reason": reason,
        "handed_off": bool(handed.item()), "hand_t": hand_t, "early_termination": early,
        # "Achieved at any point" -- distinct from the "_last"/"success" judgment above: a tower that
        # stood briefly and then fell still counts here, the way eval_chain.py's "ever" columns do.
        "tower_ever": bool(tower_ever.item()), "s1_ever": bool(s1_ever.item()), "first_tower_step": first_tower,
    })
    print(
        f"EPISODE {i + 1}/{args_cli.episodes} seed={seed} -> {'PASS' if success else 'FAIL'}: {reason}"
        + (f"  [tower formed at some point, step {first_tower}, but not at episode end]"
           if bool(tower_ever.item()) and not success else "")
    )

    if args_cli.video:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import imageio.v2 as imageio

        caption = f"{args_cli.label}, episode {i + 1}/{args_cli.episodes}, {'SUCCESS' if success else 'FAIL'}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        bar_rgb = (46, 160, 67) if success else (218, 54, 51)
        captioned = []
        for frame in frames:
            h, w = frame.shape[0], frame.shape[1]
            canvas = Image.new("RGB", (w, h + 40), bar_rgb)
            canvas.paste(Image.fromarray(frame, mode="RGB"), (0, 40))
            ImageDraw.Draw(canvas).text((8, 10), caption, fill=(255, 255, 255), font=font)
            captioned.append(np.array(canvas))
        video_path = os.path.join(args_cli.video_dir, f"episode_{i + 1:02d}.mp4")
        imageio.mimwrite(video_path, captioned, fps=int(round(1.0 / u.step_dt)))
        print(f"  video -> {video_path}")

k = sum(r["success"] for r in results)
k_ever = sum(r["tower_ever"] for r in results)
print("=" * 90)
print(f"{'EP':>3} {'SEED':>6} {'RESULT':>7} {'EVER':>5} {'HANDED':>7} {'HAND_T':>7}  REASON")
for r in results:
    print(f"{r['episode']:>3} {r['seed']:>6} {'PASS' if r['success'] else 'FAIL':>7} "
          f"{'yes' if r['tower_ever'] else 'no':>5} {'yes' if r['handed_off'] else 'no':>7} "
          f"{r['hand_t']:>7}  {r['reason']}")
print("=" * 90)
n_run = args_cli.episodes - args_cli.start_episode
print(f"Tower standing at episode end: {k}/{n_run}   Tower achieved at any point: {k_ever}/{n_run}")
print(f"ACCEPT {k}/{n_run}")
print("=" * 90)

env.close()
simulation_app.close()
