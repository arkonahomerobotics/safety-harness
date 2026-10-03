"""Reconstructed closed-loop GR00T eval for the Franka visuomotor cube-stack task.

The original script that connected this Isaac Lab env to the GR00T policy server
only ever existed on the now-gone Brev box's disk and was never committed anywhere.
This is a from-scratch reconstruction built from: the checkpoint's own self-describing
training config (experiment_cfg/conf.yaml: video.table_cam/wrist_cam, state/action
single_arm(6)+gripper(1)), the env's real action-space split (arm_action=6,
gripper_action=1, confirmed live), and NVIDIA's own published reference client
(Isaac-GR00T examples/rebot-arm-dm/eval_rebot_arm_dm.py) for the server protocol shape.

Observation encoding, each piece verified against the checkpoint's statistics.json or the
original run's saved policy-camera video:
- state.single_arm (9D) = eef_pos(3) + rot6d(6), the first two columns of the eef rotation
  matrix (GR00T's standard rotation_6d state format). Isaac Lab 3.x quats are xyzw.
- state.gripper (2D) = gripper_pos as the env reports it, (finger1, -finger2).
- video = raw 200x200 table_cam / wrist_cam from the base visuomotor env (DLAA), as in the original run.
- action.single_arm (6D) = IK-Rel eef delta, passed straight to arm_action; action.gripper (1D)
  passed to the sign-thresholded BinaryJointPositionAction.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors import CameraCfg
from isaaclab.utils import validate
from isaaclab.utils.math import matrix_from_quat

import isaaclab_mimic.envs  # noqa: F401  (registers the *-Mimic-v0 tasks)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings

from gr00t.policy.server_client import PolicyClient

# The original 10/10 run (S3 gr00t_output/closed_loop_v4/rollout_v4_final.log) parsed
# isaaclab_mimic.envs.franka_stack_ik_rel_visuomotor_mimic_env_cfg:FrankaCubeStackIKRelVisuomotorMimicEnvCfg
# -- the base visuomotor env (DLAA, default dome light), not the Cosmos variant.
TASK = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0"
COSMOS_TASK = "IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos"
LANG_INSTRUCTION = "stack the cubes"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Franka + GR00T closed-loop eval")
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--policy_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=5556)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--out", type=str, default="/workspace/isaaclab/franka_gr00t_rollout.mp4")
    parser.add_argument("--render_sweep", action="store_true", help="measure camera noise under RTX settings and exit")
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli


def build_env_cfg(task: str, num_envs: int):
    env_cfg, _ = resolve_task_config(task, "")
    env_cfg.scene.num_envs = num_envs

    # Cameras stay at the task's native 200x200: the original run's saved policy-camera video
    # (closed_loop_v4/final/*_policy_cameras_table_wrist.mp4) is 400x200 = two 200x200 frames, and
    # the GR00T processor applies its own crop/resize server-side.

    # Version-skew workaround: this contrib task's ObsTerm params predate a stricter
    # CameraImageBase.__init__ that now requires data_type/frame_stack/channel_first
    # explicitly in params.
    for cam_name in ("table_cam", "wrist_cam"):
        term = getattr(env_cfg.observations.policy, cam_name)
        term.params.setdefault("data_type", "rgb")
        term.params.setdefault("frame_stack", 1)
        term.params.setdefault("channel_first", False)

    # Not needed for GR00T inference -- disable to dodge the same version-skew
    # issue in terms we don't use.
    for unused in ("table_cam_segmentation", "table_cam_normals", "table_cam_depth"):
        if hasattr(env_cfg.observations.policy, unused):
            setattr(env_cfg.observations.policy, unused, None)

    # Dedicated higher-res demo camera, same vantage point as the policy's own
    # table_cam (which is deliberately tiny -- 200x200 -- to match training input
    # and looks grainy on playback). Read directly off the scene sensor, not through
    # the observation manager, so it's purely for recording.
    env_cfg.scene.demo_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/demo_cam",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.4), rot=(-0.61237, -0.61237, 0.35355, 0.35355), convention="ros"),
    )
    # Isaac RTX renderer settings are process-global and must match across every
    # Isaac RTX camera in the scene.
    if task == COSMOS_TASK:
        # The Cosmos cfg sets antialiasing_mode="Off"; measured, that leaves path-traced frames at
        # ~28 mean neighbor-pixel noise vs 1.72 in the original run's lossless policy-camera
        # capture. DLSS measures 1.55 -- closest match (DLAA: 1.48).
        for cam_name in ("table_cam", "wrist_cam", "demo_cam"):
            set_isaac_rtx_global_settings(
                getattr(env_cfg.scene, cam_name).renderer_cfg,
                dome_light_upper_lower_strategy=4,
                antialiasing_mode="DLSS",
                enable_dl_denoiser=True,
            )
    else:
        # Base visuomotor cfg already sets DLAA on table_cam/wrist_cam; match it on demo_cam.
        set_isaac_rtx_global_settings(env_cfg.scene.demo_cam.renderer_cfg, antialiasing_mode="DLAA")

    return env_cfg


def _table_cam_noise(env) -> tuple[float, list]:
    """Mean abs neighbor-pixel diff on a flat table patch (original lossless capture: 1.72)."""
    obs = None
    zero = torch.zeros((1, 7), device=env.unwrapped.device)
    zero[:, 6] = 1.0  # keep gripper open
    for _ in range(8):  # let temporal AA / denoisers accumulate
        with torch.inference_mode():
            obs, *_ = env.step(zero)
    img = obs["policy"]["table_cam"][0].to(torch.uint8).cpu().numpy().astype(float)
    patch = img[150:190, 10:60]
    return float(np.abs(np.diff(patch, axis=1)).mean()), patch.reshape(-1, 3).mean(0).round(1).tolist()


def _render_sweep(env) -> None:
    import carb

    s = carb.settings.get_settings()
    probe = ["/rtx/rendermode", "/rtx/post/aa/op", "/rtx-transient/dldenoiser/enabled", "/rtx/post/dlss/execMode",
             "/rtx/directLighting/sampledLighting/samplesPerPixel", "/rtx/rtpt/spp", "/rtx/pathtracing/spp",
             "/rtx/rtpt/denoiser/enabled"]
    print("[SWEEP] current:", {k: s.get(k) for k in probe}, flush=True)
    env.reset()
    configs = [
        ("baseline", {}),
        ("aa=TAA", {"/rtx/post/aa/op": 1}),
        ("aa=DLSS", {"/rtx/post/aa/op": 3}),
        ("aa=DLAA", {"/rtx/post/aa/op": 4}),
        ("dldenoiser", {"/rtx/post/aa/op": 0, "/rtx-transient/dldenoiser/enabled": True}),
        ("dldenoiser+DLAA", {"/rtx/post/aa/op": 4, "/rtx-transient/dldenoiser/enabled": True}),
        ("spp4", {"/rtx/directLighting/sampledLighting/samplesPerPixel": 4}),
        ("rtpt spp8", {"/rtx/rtpt/spp": 8, "/rtx/pathtracing/spp": 8}),
        ("rtpt denoiser", {"/rtx/rtpt/denoiser/enabled": True}),
    ]
    for name, cfg in configs:
        for k, v in cfg.items():
            s.set(k, v)
        noise, mean = _table_cam_noise(env)
        print(f"[SWEEP] {name:18s} noise={noise:6.2f} table_mean={mean}  (target 1.72, [78.7, 87.3, 79.0])", flush=True)


def obs_to_policy_input(obs: dict) -> dict:
    table_cam = obs["table_cam"][0].to(torch.uint8).cpu().numpy()  # raw 200x200, as in the original run
    wrist_cam = obs["wrist_cam"][0].to(torch.uint8).cpu().numpy()
    if not globals().get("_dumped_debug_imgs"):
        from PIL import Image

        Image.fromarray(table_cam).save("/tmp/debug_table_cam.png")
        Image.fromarray(wrist_cam).save("/tmp/debug_wrist_cam.png")
        globals()["_dumped_debug_imgs"] = True

    # Revised hypothesis, grounded in the checkpoint's own statistics.json (not a guess): raw
    # joint_pos satisfied the *dimension* (9) but not the actual value ranges -- real stats show
    # index 3 clustered near 1.0 and index 7 near -1.0, classic 6D continuous-rotation-representation
    # behavior (Zhou et al.), not joint angles. state.single_arm = eef_pos(3) + first two columns of
    # the eef rotation matrix(6) = 9D: a standard robot-learning pose encoding, not joint-space.
    eef_pos = obs["eef_pos"][0]  # (3,)
    # Isaac Lab 3.x quaternions are already (x, y, z, w) -- matrix_from_quat's native order.
    # A [1,2,3,0] reorder (assuming wxyz) misread the downward gripper as near-identity and put
    # state[7] at +1 vs training range [-1.0, -0.91], saturating every action channel.
    eef_quat = obs["eef_quat"][0]  # xyzw
    rotmat = matrix_from_quat(eef_quat.unsqueeze(0))[0]  # (3, 3)
    rot6d = rotmat[:, :2].T.reshape(-1)  # first two columns, flattened -> (6,)
    single_arm_state = torch.cat([eef_pos, rot6d]).cpu().numpy().astype(np.float32)  # (9,)
    if not globals().get("_dumped_state"):
        print(f"[STATE_DEBUG] raw eef_quat={eef_quat.tolist()} eef_pos={eef_pos.tolist()}", flush=True)
        print(f"[STATE_DEBUG] single_arm_state={single_arm_state.round(4).tolist()}", flush=True)
        print(f"[STATE_DEBUG] gripper_pos={obs['gripper_pos'][0].tolist()}", flush=True)
        globals()["_dumped_state"] = True

    # Same "raw proprioception, not a derived scalar" pattern as single_arm above --
    # confirmed via another boolean-dim mismatch (expected 2, got 1).
    gripper_state = obs["gripper_pos"][0].cpu().numpy().astype(np.float32)  # (2,)

    model_input = {
        "video": {
            "table_cam": table_cam[np.newaxis, np.newaxis, ...],
            "wrist_cam": wrist_cam[np.newaxis, np.newaxis, ...],
        },
        "state": {
            "single_arm": single_arm_state[np.newaxis, np.newaxis, ...],
            "gripper": gripper_state[np.newaxis, np.newaxis, ...],
        },
        "language": {"annotation.human.task_description": [[LANG_INSTRUCTION]]},
    }
    return model_input


def decode_action_chunk(chunk: dict, t: int, device, step: int) -> torch.Tensor:
    single_arm = np.asarray(chunk["single_arm"])[0][t]  # (6,)
    gripper = np.asarray(chunk["gripper"])[0][t]  # (1,)
    if single_arm.shape[-1] != 6:
        raise ValueError(f"Policy returned single_arm dim {single_arm.shape[-1]}, expected 6.")
    print(
        f"[ACTION] step={step} gripper={gripper.tolist()} "
        f"single_arm_norm={np.linalg.norm(single_arm):.4f} single_arm={single_arm.round(3).tolist()}",
        flush=True,
    )
    full = np.concatenate([single_arm, gripper], axis=0)  # (7,) matches arm_action(6)+gripper_action(1)
    return torch.tensor(full, dtype=torch.float32, device=device).unsqueeze(0)


def main() -> None:
    args_cli = _parse_args()
    env_cfg = build_env_cfg(args_cli.task, num_envs=1)

    try:
        validate(env_cfg)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid environment configuration: {exc}") from None

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        env = gym.make(args_cli.task, cfg=env_cfg)
        if args_cli.render_sweep:
            _render_sweep(env)
            return
        cleanup.callback(env.close)

        policy_client = PolicyClient(host=args_cli.policy_host, port=args_cli.policy_port)
        cleanup.callback(policy_client.close)

        print(f"[INFO]: Gym observation keys: {list(env.unwrapped.observation_space.keys())}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        results = []
        out_dir = os.path.dirname(args_cli.out) or "."
        out_stem, out_ext = os.path.splitext(os.path.basename(args_cli.out))
        os.makedirs(out_dir, exist_ok=True)

        for rollout_idx in range(args_cli.num_rollouts):
            # Written per-rollout, not accumulated across the whole batch: holding every
            # frame from every rollout in one list grows unboundedly (10 rollouts x 600
            # steps x 1280x720x3 bytes ~= 16.6GB) and was the real cause of a severe
            # slowdown in an earlier batch run, not model/server latency (confirmed via
            # per-call timing: ~0.14s/call, negligible).
            frames: list[np.ndarray] = []
            obs, _ = env.reset()
            policy_obs = {k: v for k, v in obs["policy"].items()}
            success = False
            steps = 0

            while steps < args_cli.max_steps:
                model_input = obs_to_policy_input(policy_obs)
                _t0 = time.monotonic()
                action_chunk, _info = policy_client.get_action(model_input)
                print(f"[TIMING] get_action call took {time.monotonic() - _t0:.2f}s at step {steps}", flush=True)
                any_key = next(iter(action_chunk.keys()))
                horizon = min(args_cli.action_horizon, np.asarray(action_chunk[any_key]).shape[1])
                if steps == 0:
                    full_gripper = np.asarray(action_chunk["gripper"])[0, :, 0]
                    print(f"[FULL_HORIZON_GRIPPER] {full_gripper.tolist()}", flush=True)

                for t in range(horizon):
                    if steps >= args_cli.max_steps:
                        break
                    action = decode_action_chunk(action_chunk, t, env.unwrapped.device, steps)
                    with torch.inference_mode():
                        obs, _reward, terminated, truncated, info = env.step(action)
                    policy_obs = {k: v for k, v in obs["policy"].items()}
                    steps += 1

                    demo_cam = env.unwrapped.scene.sensors["demo_cam"]
                    frame = demo_cam.data.output["rgb"][0].to(torch.uint8).cpu().numpy()
                    frames.append(frame)

                    term_mgr = env.unwrapped.termination_manager
                    if "success" in term_mgr.active_terms and term_mgr.get_term("success")[0]:
                        success = True
                        break
                    if bool(terminated[0]) or bool(truncated[0]):
                        break

                if success or bool(terminated[0]) or bool(truncated[0]):
                    break

            print(f"[RESULT] rollout {rollout_idx}: steps={steps} success={success}")
            results.append((rollout_idx, steps, success))

            if frames:
                import imageio.v2 as imageio

                suffix = "success" if success else "fail"
                rollout_out = os.path.join(out_dir, f"{out_stem}_{rollout_idx:02d}_{suffix}{out_ext}")
                imageio.mimwrite(rollout_out, frames, fps=20)
                print(f"[INFO]: Wrote {len(frames)} frames to {rollout_out}", flush=True)
            del frames

        print(f"Success rate: {sum(1 for _, _, s in results if s)}/{len(results)}")
        for idx, steps, success in results:
            print(f"  rollout {idx:02d}: steps={steps:4d} success={success}")


if __name__ == "__main__":
    main()
