"""Scripted privileged-state expert for Franka cube stacking, run across many parallel envs (Isaac Lab 3.x port).

The Expert phase machine is the recovered Brev ``expert_stack.py`` one, unchanged (cube_2 onto cube_1, then cube_3
onto cube_2, IK-Rel position-only actions, DART-style Gaussian noise and "shove" perturbations on the EXECUTED action
only). The capture_expert_snapshots*.py scripts carry the same phase machine.

Port changes vs the Brev script:
* default task is the camera-free ``IsaacContrib-Stack-Cube-Franka-IK-Rel`` (Isaac Lab 3.x id of the old
  ``Isaac-Stack-Cube-Franka-IK-Rel-v0``: same robot/cubes/table, IK-Rel arm action at scale 0.5);
* positions are read straight from the scene (ee_frame / cube root poses / finger joints) instead of camera-task
  observation keys, and the GR00T camera/HDF5 recording path is dropped (GR00T demos now come from
  /workspace/isaaclab/franka_scripted_expert.py). This script is for measuring the expert and sizing runs;
* each env runs exactly ``--episodes_per_env`` episodes and the outcome is read from the env's own ``success``
  termination (cubes_stacked: both stacked, gripper open, cubes at rest), time_out or a cube-drop termination.
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted expert for Franka cube stacking (vectorized over envs).")
parser.add_argument("--task", type=str, default="IsaacContrib-Stack-Cube-Franka-IK-Rel")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes_per_env", type=int, default=1)
parser.add_argument("--noise_std", type=float, default=0.0, help="Gaussian noise std on executed xyz actions.")
parser.add_argument("--perturb_prob", type=float, default=0.0, help="Per-episode probability of shove perturbations.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_phases", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

CUBE_H = 0.0468  # center-to-center height of stacked cubes (from cubes_stacked height_diff)
HOVER = 0.10
KP = 1.0
AMAX = 0.05
OPEN, CLOSE = 1.0, -1.0


class Expert:
    """Pick cube_2 -> place on cube_1, then pick cube_3 -> place on cube_2."""

    def __init__(self):
        self.tasks = [("cube_2", "cube_1"), ("cube_3", "cube_2")]
        self.task_idx = 0
        self.phase = "above"
        self.phase_steps = 0
        self.done = False

    def set_phase(self, phase):
        if args_cli.log_phases:
            print(f"        -> {self.tasks[min(self.task_idx, 1)][0]} {phase}", flush=True)
        self.phase = phase
        self.phase_steps = 0

    def act(self, eef, cubes, finger):
        """Return (7-dim clean action). eef/cubes are env-relative positions."""
        self.phase_steps += 1
        if self.done:
            return self._to(eef, eef + np.array([0, 0, 0.0]), OPEN)
        obj_name, dest_name = self.tasks[self.task_idx]
        obj, dest = cubes[obj_name], cubes[dest_name]
        grip = OPEN
        target = eef.copy()
        xy_err_obj = np.linalg.norm((obj - eef)[:2])
        holding = np.linalg.norm(obj - eef) < 0.03 and finger < 0.035

        if self.phase == "above":
            target = obj + np.array([0, 0, HOVER])
            if xy_err_obj < 0.006 and abs(target[2] - eef[2]) < 0.015:
                self.set_phase("descend")
        elif self.phase == "descend":
            target = obj + np.array([0, 0, 0.0])
            if xy_err_obj > 0.02:
                self.set_phase("above")
            elif np.linalg.norm(target - eef) < 0.008 or self.phase_steps > 60:
                self.set_phase("grasp")
            else:
                return self._to(eef, target, grip, z_max=0.02)
        elif self.phase == "grasp":
            target = obj.copy()
            grip = CLOSE
            if self.phase_steps > 12:
                self.set_phase("lift")
        elif self.phase == "lift":
            grip = CLOSE
            target = np.array([eef[0], eef[1], dest[2] + CUBE_H + HOVER])
            if not holding and self.phase_steps > 5:
                self.set_phase("above")
            elif abs(target[2] - eef[2]) < 0.015:
                self.set_phase("move")
        elif self.phase == "move":
            grip = CLOSE
            target = dest + np.array([0, 0, CUBE_H + HOVER])
            if not holding:
                self.set_phase("above")
            elif np.linalg.norm((target - eef)[:2]) < 0.006 and abs(target[2] - eef[2]) < 0.015:
                self.set_phase("lower")
        elif self.phase == "lower":
            grip = CLOSE
            target = dest + np.array([0, 0, CUBE_H + 0.004])
            if not holding:
                self.set_phase("above")
            elif np.linalg.norm((dest - eef)[:2]) > 0.02:
                self.set_phase("move")
            elif np.linalg.norm(target - eef) < 0.006 or self.phase_steps > 60:
                self.set_phase("release")
            else:
                return self._to(eef, target, grip, z_max=0.02)
        elif self.phase == "release":
            grip = OPEN
            target = eef.copy()
            if self.phase_steps > 10:
                self.set_phase("retreat")
        elif self.phase == "retreat":
            grip = OPEN
            target = np.array([eef[0], eef[1], dest[2] + CUBE_H + HOVER])
            if abs(target[2] - eef[2]) < 0.02:
                self.task_idx += 1
                if self.task_idx >= len(self.tasks):
                    self.done = True
                else:
                    self.set_phase("above")
        return self._to(eef, target, grip)

    @staticmethod
    def _to(eef, target, grip, z_max=AMAX):
        pos = np.clip(KP * (target - eef), -AMAX, AMAX)
        pos[2] = np.clip(pos[2], -z_max, z_max)
        return np.array([*pos, 0.0, 0.0, 0.0, grip], dtype=np.float32)


def sample_shoves(rng):
    shoves = []
    if rng.random() < args_cli.perturb_prob:
        for _ in range(rng.integers(1, 3)):
            start = int(rng.integers(20, 350))
            length = int(rng.integers(8, 20))
            d = rng.normal(size=3)
            d[2] = abs(d[2]) * 0.6
            d /= np.linalg.norm(d) + 1e-9
            shoves.append((start, start + length, d * AMAX))
    return shoves


def main():
    rng = np.random.default_rng(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    n = env.num_envs
    robot = env.scene["robot"]
    finger_ids, _ = robot.find_joints("panda_finger_joint.*")
    origins = env.scene.env_origins
    term_names = env.termination_manager.active_terms

    experts = [Expert() for _ in range(n)]
    shoves = [sample_shoves(rng) for _ in range(n)]
    steps = np.zeros(n, dtype=np.int64)
    episodes_done = np.zeros(n, dtype=np.int64)
    outcomes = {"success": 0, "time_out": 0, "dropped": 0}
    fail_phase = {}
    lengths = []
    t_start = time.time()
    total_steps = 0

    with torch.inference_mode():
        env.reset()
        # the RL env's reset leaves episode_length_buf at 0, so every env runs a full episode from here
        while (episodes_done < args_cli.episodes_per_env).any():
            eef = (env.scene["ee_frame"].data.target_pos_w.torch[:, 0, :] - origins).cpu().numpy().astype(np.float64)
            cubes = {
                k: (env.scene[k].data.root_pos_w.torch - origins).cpu().numpy().astype(np.float64)
                for k in ("cube_1", "cube_2", "cube_3")
            }
            finger = robot.data.joint_pos.torch[:, finger_ids].mean(dim=1).cpu().numpy()

            executed = np.zeros((n, 7), dtype=np.float32)
            for i in range(n):
                action = experts[i].act(eef[i], {k: v[i] for k, v in cubes.items()}, float(finger[i]))
                ex = action.copy()
                for s0, s1, d in shoves[i]:
                    if s0 <= steps[i] < s1:
                        ex[:3] = d
                if args_cli.noise_std > 0:
                    ex[:3] += rng.normal(scale=args_cli.noise_std, size=3)
                executed[i] = ex
                steps[i] += 1

            _, _, terminated, truncated, _ = env.step(torch.tensor(executed, device=env.device))
            total_steps += n
            done = (terminated | truncated).cpu().numpy()
            if not done.any():
                continue
            terms = {name: env.termination_manager.get_term(name).cpu().numpy() for name in term_names}
            for i in np.nonzero(done)[0]:
                if episodes_done[i] < args_cli.episodes_per_env:
                    if terms["success"][i]:
                        outcomes["success"] += 1
                        lengths.append(int(steps[i]))
                    else:
                        key = "time_out" if terms.get("time_out", np.zeros(n, bool))[i] else "dropped"
                        outcomes[key] += 1
                        tag = f"task{experts[i].task_idx}:{experts[i].phase}"
                        fail_phase[tag] = fail_phase.get(tag, 0) + 1
                    episodes_done[i] += 1
                experts[i] = Expert()
                shoves[i] = sample_shoves(rng)
                steps[i] = 0

    attempts = int(episodes_done.sum())
    rate = total_steps / (time.time() - t_start)
    print("=" * 80)
    print(f"EXPERT task={args_cli.task} envs={n} episodes={attempts} noise_std={args_cli.noise_std} "
          f"perturb_prob={args_cli.perturb_prob}")
    print(f"EXPERT success: {outcomes['success']}/{attempts} = {outcomes['success'] / max(attempts, 1):.3f}   "
          f"time_out={outcomes['time_out']} dropped={outcomes['dropped']}")
    if lengths:
        print(f"EXPERT success episode length: median={np.median(lengths):.0f} p90={np.percentile(lengths, 90):.0f} steps")
    if fail_phase:
        print("EXPERT failures by (task, phase at end):", dict(sorted(fail_phase.items(), key=lambda kv: -kv[1])))
    print(f"EXPERT throughput {rate:.0f} env-steps/s, wall {(time.time() - t_start) / 60:.1f} min")
    print("=" * 80)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
