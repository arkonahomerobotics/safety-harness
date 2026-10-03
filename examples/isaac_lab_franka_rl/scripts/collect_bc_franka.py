"""Collect behavior-cloning demos for the Franka cube-stacking PPO policy INSIDE the RL task env (RL obs, RL actions).

The scripted expert (expert_stack.py's phase machine, unchanged) runs inside the RL task, and its commands are
expressed in the RL task's action space:

* expert_stack.py drives the base task ``IsaacContrib-Stack-Cube-Franka-IK-Rel`` whose arm action is plain IK-Rel with
  ``scale = 0.5``: an expert xyz command ``a`` means an intended end-effector delta ``d = 0.5 * a`` (m per env step).
* The RL tasks use ``DistanceScaledIKRelAction`` (mdp/rl_actions_impl.py): delta = ``s(state) * a_rl`` with the scale
  ``s`` recomputed from the current state in ``process_actions``. So the RL action producing the expert's intended
  delta is ``a_rl = 0.5 * a / s(state)`` (rotation dims stay 0, the gripper stays +-1).
* ``s(state)`` is not re-derived here: before each step the env's own arm action term is asked for it by calling its
  ``process_actions`` on a dummy action and reading ``_scale``. ``env.step`` then calls ``process_actions`` again on
  the same (unchanged) state with the real action, so the scale is identical; ``--check_scale`` asserts it after the
  step, and the processed (applied) delta is checked against ``0.5 * a`` too.

Stage 1 (``--stage 1``): Robosuite-Snap-Sparse task with its expert start states turned off (normal starts); the
expert only does red (cube_2) onto blue (cube_1), releases, retreats and then holds still.
Stage 2 (``--stage 2``): Stage2Skill task, every episode starts from its expert snapshot distribution; the expert does
green (cube_3) onto red (cube_2). Its phase is inferred from the snapshot state at the first step after reset.

DART-style data: the EXECUTED action gets Gaussian noise (in RL action units, xyz only) and optional "shove"
perturbations; the recorded LABEL is always the clean expert action. Only successful episodes are kept, and only up
to ``--hold_steps`` steps after the expert finished (then the env is forced to time out, so no sim time is spent on
idle hold). ``--noise_std 0 --perturb_prob 0 --no_save`` measures the expert itself in the RL action space.

Output: {"obs": [N, 94] float32, "act": [N, 7] float32, "meta": {...}}. ``--template_out`` additionally writes a fresh
(iteration-0) rsl_rl checkpoint of the task's agent cfg, used by bc_train_franka.py as the checkpoint skeleton.
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--snapshot_path", type=str, default=None, help="stage 2: override the snapshot file")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--target_pairs", type=int, default=1_500_000)
parser.add_argument("--max_episodes", type=int, default=10**9, help="stop after this many finished episodes")
parser.add_argument("--noise_std", type=float, default=0.1, help="Gaussian noise std on executed xyz (RL action units)")
parser.add_argument("--perturb_prob", type=float, default=0.3, help="per-episode probability of shove perturbations")
parser.add_argument("--hold_steps", type=int, default=30, help="steps recorded after the expert is done")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--template_out", type=str, default=None)
parser.add_argument("--no_save", action="store_true")
parser.add_argument("--check_scale", action="store_true", help="assert the inversion every step (slower)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dagger_policy", type=str, default=None,
                    help="DAgger: an rsl_rl checkpoint whose (deterministic) actions are executed; the expert only labels")
parser.add_argument("--dagger_beta", type=float, default=0.0,
                    help="DAgger: per-episode probability that the expert drives instead (those episodes: success-only)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_expert import AMAX, BASE_IK_SCALE, CLOSE, OPEN, Expert, infer_phase, seated  # noqa: E402


def main():
    rng = np.random.default_rng(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if args_cli.stage == 1:
        task = args_cli.task or "Isaac-Stack-Cube-Franka-IK-Rel-RL-Robosuite-Snap-Sparse-v0"
        pick = [("cube_2", "cube_1")]
    else:
        task = args_cli.task or "Isaac-Stack-Cube-Franka-IK-Rel-RL-Stage2Skill-v0"
        pick = [("cube_3", "cube_2")]
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if args_cli.stage == 1:
        env_cfg.events.reset_from_snapshot = None  # normal starts
    elif args_cli.snapshot_path:
        env_cfg.events.reset_from_snapshot.params["snapshot_path"] = args_cli.snapshot_path
    env = gym.make(task, cfg=env_cfg).unwrapped
    n, dev = env.num_envs, env.device
    T = int(env.max_episode_length)
    robot = env.scene["robot"]
    finger_ids, _ = robot.find_joints("panda_finger_joint.*")
    origins = env.scene.env_origins
    arm = env.action_manager.get_term("arm_action")
    term_names = env.termination_manager.active_terms
    print(f"[collect] task={task} envs={n} T={T} action_dim={env.action_manager.total_action_dim} "
          f"arm term={type(arm).__name__} terminations={term_names}", flush=True)

    if args_cli.template_out:
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.utils import to_dict

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.template_out)), exist_ok=True)
        runner.save(args_cli.template_out)
        print(f"[collect] wrote fresh template checkpoint {args_cli.template_out} (clip_actions={agent_cfg.clip_actions})")
        del runner

    actor = None
    if args_cli.dagger_policy:
        sd = torch.load(args_cli.dagger_policy, map_location=dev, weights_only=False)["actor_state_dict"]
        lin = sorted({k.rsplit(".", 1)[0] for k in sd if k.startswith("mlp.") and k.endswith(".weight")},
                     key=lambda s_: int(s_.split(".")[1]))
        mu, sig = sd["obs_normalizer._mean"], sd["obs_normalizer._std"] + 1e-2

        def actor(x):  # rsl_rl MLPModel deterministic forward (EmpiricalNormalization -> MLP, ELU)
            h = (x - mu) / sig
            for j, k in enumerate(lin):
                h = h @ sd[f"{k}.weight"].T + sd[f"{k}.bias"]
                if j < len(lin) - 1:
                    h = torch.nn.functional.elu(h)
            return h

        print(f"[collect] DAgger: executing {args_cli.dagger_policy} (beta={args_cli.dagger_beta})", flush=True)
    pol_drive = np.zeros(n, dtype=bool)
    pol_out = {"success": 0, "fail": 0}

    def sample_driver():
        return actor is not None and rng.random() >= args_cli.dagger_beta

    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    obs_buf = torch.zeros(n, T + 2, obs_dim, device=dev)
    act_buf = torch.zeros(n, T + 2, 7, device=dev)
    rec_len = np.zeros(n, dtype=np.int64)

    def new_expert():
        return Expert(pick)

    experts = [new_expert() for _ in range(n)]
    fresh = np.ones(n, dtype=bool)  # first step after a reset: hold still, (stage 2) infer the phase next step
    starting = np.zeros(n, dtype=bool)
    judged = np.zeros(n, dtype=bool)
    start_phase = {}
    shoves = [[] for _ in range(n)]
    steps = np.zeros(n, dtype=np.int64)
    done_steps = np.full(n, -1, dtype=np.int64)
    kept_obs, kept_act = [], []
    n_pairs = 0
    outcomes = {"success": 0, "fail_terminated": 0, "fail_timeout": 0, "fail_not_stacked": 0}
    fail_phase = {}
    succ_len = []
    scale_err_max, delta_err_max = 0.0, 0.0
    label_abs = []

    def sample_shoves():
        out = []
        if rng.random() < args_cli.perturb_prob:
            for _ in range(rng.integers(1, 3)):
                s0 = int(rng.integers(5, 250))
                d = rng.normal(size=3)
                d[2] = abs(d[2]) * 0.6
                d /= np.linalg.norm(d) + 1e-9
                out.append((s0, s0 + int(rng.integers(8, 20)), d * AMAX))
        return out

    for i in range(n):
        shoves[i] = sample_shoves()
        pol_drive[i] = sample_driver()

    t0 = time.time()
    total_steps = 0
    obs_dict, _ = env.reset()
    with torch.inference_mode():
        obs = obs_dict["policy"].clone()
        while n_pairs < args_cli.target_pairs and sum(outcomes.values()) + sum(pol_out.values()) < args_cli.max_episodes:
            eef = (env.scene["ee_frame"].data.target_pos_w.torch[:, 0, :] - origins).cpu().numpy().astype(np.float64)
            cubes = {k: (env.scene[k].data.root_pos_w.torch - origins).cpu().numpy().astype(np.float64)
                     for k in ("cube_1", "cube_2", "cube_3")}
            finger = robot.data.joint_pos.torch[:, finger_ids].mean(dim=1).cpu().numpy()

            # the RL action term's step scale for the current state (same call env.step makes, same state)
            arm.process_actions(torch.zeros(n, arm.action_dim, device=dev))
            scale = arm._scale[:, 0].clone()
            scale_np = scale.cpu().numpy().astype(np.float64)

            exp_cmd = np.zeros((n, 7), dtype=np.float64)
            shove_delta = np.zeros((n, 3), dtype=np.float64)
            record = np.zeros(n, dtype=bool)
            for i in range(n):
                if judged[i]:  # waiting for the forced time-out: hold still, record nothing
                    exp_cmd[i, 6] = OPEN
                    continue
                if fresh[i]:
                    # first step after a reset: the ee_frame pose can still be the pre-reset one, so hold still
                    # (keep the gripper as it is) and start the expert -- inferring its phase -- on the next step
                    fresh[i] = False
                    starting[i] = True
                    exp_cmd[i, 6] = CLOSE if finger[i] < 0.035 else OPEN
                    continue
                if starting[i]:
                    starting[i] = False
                    phase = "above" if args_cli.stage == 1 else infer_phase(
                        eef[i], {k: v[i] for k, v in cubes.items()}, float(finger[i]), *pick[0])
                    experts[i] = Expert(pick, phase=phase)
                    start_phase[phase] = start_phase.get(phase, 0) + 1
                a = experts[i].act(eef[i], {k: v[i] for k, v in cubes.items()}, float(finger[i]))
                exp_cmd[i] = a
                record[i] = done_steps[i] < 0 or steps[i] - done_steps[i] < args_cli.hold_steps
                for s0, s1, d in shoves[i]:
                    if s0 <= steps[i] < s1:
                        shove_delta[i] = d * BASE_IK_SCALE
                steps[i] += 1
                if experts[i].done and done_steps[i] < 0:
                    done_steps[i] = steps[i]

            # clean label in the RL action space: same intended EE delta as the expert's base-task command
            label = np.zeros((n, 7), dtype=np.float64)
            label[:, :3] = BASE_IK_SCALE * exp_cmd[:, :3] / scale_np[:, None]
            label[:, 6] = exp_cmd[:, 6]
            executed = label.copy()
            shoved = np.abs(shove_delta).sum(1) > 0
            executed[shoved, :3] = np.clip(shove_delta[shoved] / scale_np[shoved, None], -3.0, 3.0)
            if pol_drive.any():
                pa = actor(obs).double().cpu().numpy()
                executed[pol_drive] = pa[pol_drive]
                executed[pol_drive & shoved, :3] = np.clip(shove_delta[pol_drive & shoved] / scale_np[pol_drive & shoved, None], -3.0, 3.0)
            if args_cli.noise_std > 0:
                executed[:, :3] += rng.normal(scale=args_cli.noise_std, size=(n, 3))

            lab_t = torch.tensor(label, dtype=torch.float32, device=dev)
            rec_ids = np.nonzero(record)[0]
            if len(rec_ids):
                ri = torch.as_tensor(rec_ids, device=dev)
                li = torch.as_tensor(rec_len[rec_ids], device=dev)
                obs_buf[ri, li] = obs[ri]
                act_buf[ri, li] = lab_t[ri]
                rec_len[rec_ids] += 1
                if len(label_abs) < 2000:
                    label_abs.append(np.abs(label[rec_ids, :3]).max(1))

            obs_dict, _, terminated, truncated, _ = env.step(torch.tensor(executed, dtype=torch.float32, device=dev))
            total_steps += n
            if args_cli.check_scale:
                s_after = arm._scale[:, 0]
                scale_err_max = max(scale_err_max, float((s_after - scale).abs().max()))
                applied = arm.processed_actions[:, :3].cpu().numpy()  # = scale * executed xyz
                clean = ~shoved & (args_cli.noise_std == 0)
                if clean.any():
                    delta_err_max = max(delta_err_max, float(np.abs(applied[clean] - BASE_IK_SCALE * exp_cmd[clean, :3]).max()))
            obs = obs_dict["policy"].clone()

            done = (terminated | truncated).cpu().numpy()
            # expert finished and held long enough: judge success now, then force a time-out on the next step
            cubes_now = {k: (env.scene[k].data.root_pos_w.torch - origins).cpu().numpy() for k in ("cube_1", "cube_2", "cube_3")}
            finger_now = robot.data.joint_pos.torch[:, finger_ids].mean(dim=1).cpu().numpy()
            if args_cli.stage == 1:
                ok_now = seated(cubes_now["cube_2"], cubes_now["cube_1"]) & (finger_now > 0.035)
            else:
                ok_now = seated(cubes_now["cube_3"], cubes_now["cube_2"]) & seated(cubes_now["cube_2"], cubes_now["cube_1"]) & (finger_now > 0.035)
            finish = (~done) & (~judged) & (done_steps >= 0) & (steps - done_steps >= args_cli.hold_steps)
            for i in np.nonzero(done | finish)[0]:
                if pol_drive[i] and not (done[i] and judged[i]):
                    # DAgger episode: keep every labelled state the policy visited, whatever the outcome
                    pol_out["success" if (finish[i] and ok_now[i]) else "fail"] += 1
                    L = int(rec_len[i])
                    kept_obs.append(obs_buf[i, :L].cpu())
                    kept_act.append(act_buf[i, :L].cpu())
                    n_pairs += L
                elif done[i] and judged[i]:  # the forced time-out of an already-judged episode
                    pass
                elif finish[i] and ok_now[i]:
                    outcomes["success"] += 1
                    L = int(rec_len[i])
                    kept_obs.append(obs_buf[i, :L].cpu())
                    kept_act.append(act_buf[i, :L].cpu())
                    n_pairs += L
                    succ_len.append(int(steps[i]))
                else:
                    if finish[i]:
                        outcomes["fail_not_stacked"] += 1
                    elif truncated[i]:
                        outcomes["fail_timeout"] += 1
                    else:
                        outcomes["fail_terminated"] += 1
                    tag = f"{experts[i].phase}{'(done)' if experts[i].done else ''}"
                    fail_phase[tag] = fail_phase.get(tag, 0) + 1
                if finish[i]:
                    env.episode_length_buf[i] = env.max_episode_length  # times out (and resets) on the next step
                    judged[i] = True  # stop recording / re-judging until that reset happens
                else:
                    experts[i] = new_expert()
                    fresh[i] = True
                    judged[i] = False
                    shoves[i] = sample_shoves()
                    pol_drive[i] = sample_driver()
                    steps[i] = 0
                    done_steps[i] = -1
                    rec_len[i] = 0
            ep = sum(outcomes.values()) + sum(pol_out.values())
            if ep and total_steps % (n * 200) == 0:
                if actor is not None:
                    print(f"[collect] policy-driven episodes: {pol_out}", flush=True)
                print(f"[collect] steps/env={total_steps // n} episodes={ep} success={outcomes['success'] / ep:.3f} "
                      f"pairs={n_pairs} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    ep = sum(outcomes.values())
    print("=" * 90)
    print(f"COLLECT stage={args_cli.stage} task={task} envs={n} noise_std={args_cli.noise_std} perturb_prob={args_cli.perturb_prob}")
    print(f"COLLECT expert success (RL action space): {outcomes['success']}/{ep} = {outcomes['success'] / max(ep, 1):.3f}  {outcomes}")
    print("COLLECT expert start phases:", start_phase)
    if actor is not None:
        print(f"COLLECT DAgger policy-driven episodes: {pol_out}  "
              f"(policy success {pol_out['success'] / max(sum(pol_out.values()), 1):.3f})")
    if fail_phase:
        print("COLLECT failures by phase at end:", dict(sorted(fail_phase.items(), key=lambda kv: -kv[1])))
    if succ_len:
        print(f"COLLECT success length (steps to done+hold): median={np.median(succ_len):.0f} p90={np.percentile(succ_len, 90):.0f}")
    if label_abs:
        la = np.concatenate(label_abs)
        print(f"COLLECT |label xyz| max per step: median={np.median(la):.3f} p99={np.percentile(la, 99):.3f} max={la.max():.3f}")
    if args_cli.check_scale:
        print(f"COLLECT inversion check: max |scale(pre-step query) - scale(env.step)| = {scale_err_max:.2e}; "
              f"max |applied delta - 0.5*expert cmd| (clean steps) = {delta_err_max:.2e} m")
    print(f"COLLECT pairs={n_pairs}  wall {(time.time() - t0) / 60:.1f} min, {total_steps / (time.time() - t0):.0f} env-steps/s")
    print("=" * 90, flush=True)
    if not args_cli.no_save and kept_obs:
        O, A = torch.cat(kept_obs), torch.cat(kept_act)
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        torch.save({"obs": O, "act": A, "meta": {"task": task, "stage": args_cli.stage, "episodes": outcomes,
                                                  "noise_std": args_cli.noise_std, "perturb_prob": args_cli.perturb_prob,
                                                  "hold_steps": args_cli.hold_steps, "seed": args_cli.seed}}, args_cli.out)
        print(f"COLLECT saved {O.shape[0]} pairs -> {args_cli.out}", flush=True)


if __name__ == "__main__":
    main()
    # env.close() / simulation_app.close() hung for 10+ min after a 512-env run (GPU memory still held), so exit
    # hard once the data is on disk
    os._exit(0)
