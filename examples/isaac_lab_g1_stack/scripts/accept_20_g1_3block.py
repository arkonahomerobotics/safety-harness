"""G1 3-block chained-RL acceptance harness: 20 sequential, single-env, deterministic episodes,
Stage-1 policy handing off to Stage-2 the moment its own 2-block success holds for
``--handoff_steps`` consecutive steps, judged on the env's own full 3-block success definition at
the last pre-reset state. The G1 counterpart of ``examples/isaac_lab_franka_rl/scripts/accept_20.py``
-- same acceptance-gate purpose (run it the moment a candidate checkpoint pair looks good, get a
real, individually-reproducible proof run instead of re-deriving the judging logic under time
pressure), adapted to this task's actual architecture.

**Why this isn't a single running env with a swapped observation group, unlike the Franka
version.** Stage 1 (``Isaac-Stack-Blocks-G1-RL-v0``) and Stage 2 (``Isaac-Stack-Blocks-G1-RL-
Stage2-v0``) are different scenes -- block C doesn't exist in Stage 1's at all -- so there is no
single env whose obs group can just be swapped. Instead this runs ONE Stage-2 env instance for the
whole episode (so block C and its reset are always native and real), but overrides block A's reset
to Stage 1's own real starting distribution (``A_POS``, Stage 1's ``pose_range``) instead of Stage
2's own "pre-seeded on B" convenience default for standalone Stage-2 training -- see
``g1_block_stack_3stage2_rl_env_cfg.py``'s own docstring for why that default exists and isn't the
real chained starting condition. During the Stage-1 phase, Stage 1's own ``stack_state`` observation
is computed by hand against this same env's scene (same entity names: ``object``/``block_b``/
``robot``); once handed off, Stage 2's own ``stack_state_stage2`` is already what ``env.step()``
returns in ``obs["policy"]``, since the env's own registered observation function computes it every
step regardless of which policy is "logically" in control. Confirmed against the original Stage-2
design intent and reviewed before writing any of this -- see the PR discussion.

**The one correctness requirement this whole file exists to get right**, same as the Franka
version and ``eval_policy.py``/``eval_sequential_single_env.py`` before it: Isaac Lab auto-resets an
env inside the very ``step()`` call whose ``term``/``trunc`` comes back true, so reading scene state
on any later iteration already sees the next episode. Every per-step state used for judging
(``stage1_stacked_full`` et al.) is read at the *top* of each iteration, before that iteration's own
``env.step()``; the moment ``term``/``trunc`` fires, the loop ``break``s immediately, before ever
reading scene state again, so whatever was latched that same iteration -- one control step before
the terminating transition was committed -- is the judged final state.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-Stage2-v0")
parser.add_argument("--stage1", type=str, required=True, help="Stage-1 (2-block) checkpoint")
parser.add_argument("--stage2", type=str, required=True, help="Stage-2 (3-block) checkpoint")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--handoff_steps", type=int, default=15, help="consecutive Stage-1-success steps before handoff")
parser.add_argument("--video", action="store_true", help="write one captioned mp4 per episode")
parser.add_argument("--video_dir", type=str, default="accept_20_g1_3block_videos")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.contrib.locomanip_pick_place.g1_block_stack_rl_env_cfg import A_POS, BLOCK  # noqa: E402
from isaaclab_tasks.contrib.locomanip_pick_place.mdp.rl_obs_g1 import stack_state  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def load_actor(path, device):
    """Verbatim from eval_policy.py/eval_sequential_single_env.py -- rebuilds the bare MLP actor
    (with its observation normalizer) straight from an rsl_rl checkpoint's actor_state_dict, no
    OnPolicyRunner/RslRlVecEnvWrapper needed since this repo's G1 scripts drive gym.make(...)
    .unwrapped directly."""
    sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    ks = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    layers = []
    for j, k in enumerate(ks):
        w, b = sd[f"mlp.{k}.weight"], sd[f"mlp.{k}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0]).to(device)
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if j < len(ks) - 1:
            layers.append(nn.ELU())
    mlp = nn.Sequential(*layers).eval()
    mean, std = sd["obs_normalizer._mean"].to(device), sd["obs_normalizer._std"].to(device)
    return lambda o: mlp((o - mean) / (std + 1e-2))


env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env_cfg.seed = args_cli.seed
env_cfg.scene.robot_pov_cam = None  # XR-teleop leftover; only renders correctly under --video and this script never needs it

# Give Stage 1 a REAL 2-block task to earn the handoff from, instead of Stage 2's own "pre-seeded
# on B" standalone-training default -- see this file's module docstring and the PR discussion with
# Engineering Development #2 for why this is the correct mechanism, not a snapshot-based transfer.
env_cfg.scene.object.init_state.pos = A_POS
env_cfg.events.reset_block_a.params["pose_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02)}

# ~40 degrees from horizontal, same horizontal direction as eval_policy.py's own working demo_cam
# (opposite the left arm, which is all this task ever moves) -- just pitched steeper, per spec.
# UNVERIFIED without a real render: confirm the arm never crosses the sightline across a FULL clip,
# not just the opening frame -- see this example's own README for the camera-framing bug this exact
# mistake caused before (frame 0 looked fine; the arm's reach swept across the shot for most of it).
EYE_OFFSET = (0.30, 0.50, 0.49)
LOOKAT_MID = (-0.20, 0.36, 0.75)

if args_cli.video:
    import isaaclab.sim as sim_utils  # noqa: E402
    from isaaclab.sensors import CameraCfg  # noqa: E402

    env_cfg.scene.demo_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
    )

env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
dev = env.device

policy1 = load_actor(args_cli.stage1, dev)
policy2 = load_actor(args_cli.stage2, dev)

object_ = env.scene["object"]
block_b = env.scene["block_b"]
block_c = env.scene["block_c"]
robot = env.scene["robot"]
widx = robot.data.body_names.index("left_wrist_yaw_link")
o_ = env.scene.env_origins

if args_cli.video:
    os.makedirs(args_cli.video_dir, exist_ok=True)
    env.scene["demo_cam"].set_world_poses_from_view(
        (torch.tensor(LOOKAT_MID, device=dev) + torch.tensor(EYE_OFFSET, device=dev)).unsqueeze(0) + o_,
        torch.tensor(LOOKAT_MID, device=dev).unsqueeze(0) + o_,
    )


def _pos(entity):
    return entity.data.root_pos_w.torch[0] - o_[0]


def _speed(entity):
    return float(torch.linalg.norm(entity.data.root_vel_w.torch[0, :3]))


def _wrist_pos():
    return robot.data.body_pos_w.torch[0, widx] - o_[0]


def a_on_b_positions_ok() -> bool:
    """Positional-only: is A currently seated on B, xy/z tolerance only -- no still/hand-clear
    qualifiers. Used post-handoff to detect the tower being knocked over mid-manipulation; the hand
    legitimately spends time right next to the tower while placing C, so requiring it "clear" here
    would misclassify normal placement as a disturbance. Reserved for final-state judging instead
    (see full_tower_ok)."""
    pa, pb = _pos(object_), _pos(block_b)
    dxy = float(torch.linalg.norm((pa - pb)[:2]))
    dz = float(pa[2] - pb[2] - BLOCK)
    b_on_table = abs(float(pb[2]) - env.cfg.scene.block_b.init_state.pos[2]) < 0.01
    return (dxy < 0.025) and (abs(dz) < 0.012) and b_on_table


def c_on_a_positions_ok() -> bool:
    """Positional-only: is C currently seated on A. Same reasoning as a_on_b_positions_ok -- used
    for "ever placed" bookkeeping, not final judging."""
    pa, pc = _pos(object_), _pos(block_c)
    dxy = float(torch.linalg.norm((pc - pa)[:2]))
    dz = float(pc[2] - pa[2] - BLOCK)
    return (dxy < 0.025) and (abs(dz) < 0.012)


def stage1_stacked_full() -> bool:
    """A seated on B, settled, hand clear of A -- Stage 1's OWN success definition, unchanged from
    eval_policy.py's/eval_sequential_single_env.py's stacked_now(). Used only to decide when the
    real 2-block task has actually been earned (the handoff streak), not for final judging."""
    if not a_on_b_positions_ok():
        return False
    return (_speed(object_) < 0.03) and (float(torch.linalg.norm(_pos(object_) - _wrist_pos())) > 0.11)


def full_tower_ok() -> bool:
    """The env's own full 3-block success definition (Kaoru-approved thresholds, matching
    _STACK3's place_height/success_max_speed/hand_clear_dist params in
    g1_block_stack_3stage2_rl_env_cfg.py): A on B and C on A, both xy within 2.5cm and z within
    1.2cm, B still on the table within 1cm of its reset height, every block at rest (<0.03 m/s),
    and the hand clear of C (>0.11m) -- judged on the last pre-reset state, per this file's
    docstring."""
    if not (a_on_b_positions_ok() and c_on_a_positions_ok()):
        return False
    still = _speed(object_) < 0.03 and _speed(block_b) < 0.03 and _speed(block_c) < 0.03
    clear = float(torch.linalg.norm(_pos(block_c) - _wrist_pos())) > 0.11
    return still and clear


def classify(handed: bool, success: bool, tower_disturbed_before_c: bool, c_ever_placed: bool) -> str:
    """Per-episode failure (or success) reason, in the order asked for: never handed off / tower
    disturbed before C / C not placed / C placed but fell."""
    if success:
        return "tower standing (success)"
    if not handed:
        return "never handed off to stage 2"
    if tower_disturbed_before_c:
        return "tower disturbed before C was ever placed"
    if not c_ever_placed:
        return "C never placed on the tower"
    return "C placed on the tower but fell before episode end"


VIDEO_FPS = 30  # matches eval_policy.py's own demo-clip convention, not a measured real-time rate
HOLD_FINAL_FRAME_S = 2.0

results = []
for ep in range(args_cli.episodes):
    obs, _ = env.reset()
    env.episode_length_buf[:] = 0

    handed = False
    s1_run = 0
    hand_t = -1
    tower_ever = False
    first_tower = -1
    c_ever_placed = False
    tower_disturbed_before_c = False
    final_success = False
    frames = []

    for step in range(env.max_episode_length + 5):
        a_ok = a_on_b_positions_ok()
        c_ok = c_on_a_positions_ok()
        tower_now = full_tower_ok()

        if not handed:
            s1_now = stage1_stacked_full()
            s1_run = s1_run + 1 if s1_now else 0
            if s1_run >= args_cli.handoff_steps:
                handed, hand_t = True, step
        else:
            # Independent checks, not elif: "disturbed" and "placed" are different claims about the
            # same step, and a c_ok-but-also-a_ok-false reading (contrived, but possible) should
            # still register as placed without silently skipping the disturbed check for every
            # other step. c_ever_placed uses its value as of BEFORE this step's update, so a
            # same-step "just placed" reading takes priority over flagging that step as disturbed.
            if not a_ok and not c_ever_placed:
                tower_disturbed_before_c = True
            if c_ok:
                c_ever_placed = True
        tower_ever = tower_ever or tower_now
        if tower_now and first_tower < 0:
            first_tower = step
        final_success = tower_now  # always update; this iteration hasn't stepped (so can't have auto-reset) yet

        if args_cli.video:
            frames.append(env.scene["demo_cam"].data.output["rgb"][0, ..., :3].clone().cpu())

        if handed:
            act = policy2(obs["policy"]).clamp(-1, 1)
        else:
            act = policy1(stack_state(env)).clamp(-1, 1)
        obs, rew, term, trunc, extras = env.step(act)

        if bool(term[0]) or bool(trunc[0]):
            break  # the state latched above is the last one this episode was confirmed live in

    reason = classify(handed, final_success, tower_disturbed_before_c, c_ever_placed)
    results.append({
        "episode": ep + 1, "success": final_success, "reason": reason, "handed_off": handed,
        "hand_t": hand_t, "tower_ever": tower_ever, "first_tower_step": first_tower,
    })
    print(
        f"EPISODE {ep + 1}/{args_cli.episodes} -> {'PASS' if final_success else 'FAIL'}: {reason}"
        + (f"  [tower formed at some point, step {first_tower}, but not at episode end]"
           if tower_ever and not final_success else "")
    )

    if args_cli.video:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        from overlay import write_mp4

        caption = f"G1 3-block, episode {ep + 1}/{args_cli.episodes}, {'SUCCESS' if final_success else 'FAIL'}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
        bar_rgb = (46, 160, 67) if final_success else (218, 54, 51)
        captioned = []
        for fr in frames:
            arr = fr.numpy().astype("uint8")
            h, w = arr.shape[0], arr.shape[1]
            canvas = Image.new("RGB", (w, h + 44), bar_rgb)
            canvas.paste(Image.fromarray(arr, mode="RGB"), (0, 44))
            ImageDraw.Draw(canvas).text((8, 10), caption, fill=(255, 255, 255), font=font)
            captioned.append(np.array(canvas))
        if captioned:
            captioned.extend([captioned[-1]] * int(round(HOLD_FINAL_FRAME_S * VIDEO_FPS)))
        video_path = os.path.join(args_cli.video_dir, f"episode_{ep + 1:02d}.mp4")
        write_mp4(captioned, video_path, fps=VIDEO_FPS)
        print(f"  video -> {video_path}")

k = sum(r["success"] for r in results)
k_ever = sum(r["tower_ever"] for r in results)
print("=" * 90)
print(f"{'EP':>3} {'RESULT':>7} {'EVER':>5} {'HANDED':>7} {'HAND_T':>7}  REASON")
for r in results:
    print(f"{r['episode']:>3} {'PASS' if r['success'] else 'FAIL':>7} "
          f"{'yes' if r['tower_ever'] else 'no':>5} {'yes' if r['handed_off'] else 'no':>7} "
          f"{r['hand_t']:>7}  {r['reason']}")
print("=" * 90)
print(f"Tower standing at episode end: {k}/{args_cli.episodes}   Tower achieved at any point: {k_ever}/{args_cli.episodes}")
print(f"ACCEPT {k}/{args_cli.episodes}")
print("=" * 90)

simulation_app.close()
