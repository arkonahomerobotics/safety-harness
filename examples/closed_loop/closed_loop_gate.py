"""Closed-loop measurement: a working Franka cube-stacking controller, gated vs ungated.

The controller is the scripted privileged-state expert (expert_stack.py's phase machine, unchanged
logic, 92-94% success on its own). Every 20 Hz control step for every env is translated into the
harness decision it represents -- ``grasp`` on the gripper-closing edge, ``place`` on the opening
edge while holding, ``reach`` otherwise, with the real commanded Cartesian speed -- and passed
through the real IsaacLabCubeStackPerceptionAdapter / DynamicsAdapter / ActuatorGate (pinned example
config). What executes is DecisionWatchdog.command(): the permitted action if fresh and
bit-identical, else freeze (zero arm motion, gripper held). Nothing is injected in the nominal
condition. The human condition adds a PHYSICAL kinematic capsule (a human proxy) that walks up to
the table, stays, and leaves -- tracked from its real simulated pose and velocity.

Answers: does the gate get in the way of a working robot (task success and false-block rate,
gated vs ungated), and does it keep the arm away from a person who actually approaches.
"""

import argparse
import json
import math
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--gated", type=int, default=1)
parser.add_argument("--human", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episode_s", type=float, default=60.0, help="long enough that a correct freeze for a present human doesn't turn into a timeout")
parser.add_argument("--harness", type=str, default="/workspace/safety_harness")
parser.add_argument("--out", type=str, default="/tmp/closed_loop.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

sys.path.insert(0, args_cli.harness)
from safety_harness import ActionSchemaRegistry, ActuatorGate, DecisionWatchdog  # noqa: E402
from safety_harness.adapters import FreezeInPlaceFallback, InMemoryLogger  # noqa: E402
from safety_harness.adapters.isaac_lab import (  # noqa: E402
    IsaacLabCubeStackDynamicsAdapter, IsaacLabCubeStackPerceptionAdapter,
)
from safety_harness.integrity import read_digest_file  # noqa: E402
from safety_harness.schema import Action, AgentCategory, Pose, TrackedAgent  # noqa: E402

# ---- the expert: expert_stack.py's phase machine, logic unchanged -------------------------------
CUBE_H, HOVER, KP, AMAX = 0.0468, 0.10, 1.0, 0.05
OPEN, CLOSE = 1.0, -1.0


class Expert:
    def __init__(self):
        self.tasks = [("cube_2", "cube_1"), ("cube_3", "cube_2")]
        self.task_idx, self.phase, self.phase_steps, self.done = 0, "above", 0, False
        self.holding = False

    def set_phase(self, p):
        self.phase, self.phase_steps = p, 0

    def act(self, eef, cubes, finger):
        self.phase_steps += 1
        if self.done:
            return self._to(eef, eef, OPEN)
        obj_name, dest_name = self.tasks[self.task_idx]
        obj, dest = cubes[obj_name], cubes[dest_name]
        grip, target = OPEN, eef.copy()
        xy_err_obj = np.linalg.norm((obj - eef)[:2])
        holding = np.linalg.norm(obj - eef) < 0.03 and finger < 0.035
        self.holding = holding
        if self.phase == "above":
            target = obj + np.array([0, 0, HOVER])
            if xy_err_obj < 0.006 and abs(target[2] - eef[2]) < 0.015:
                self.set_phase("descend")
        elif self.phase == "descend":
            target = obj.copy()
            if xy_err_obj > 0.02:
                self.set_phase("above")
            elif np.linalg.norm(target - eef) < 0.008 or self.phase_steps > 60:
                self.set_phase("grasp")
            else:
                return self._to(eef, target, grip, z_max=0.02)
        elif self.phase == "grasp":
            target, grip = obj.copy(), CLOSE
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
            grip, target = OPEN, eef.copy()
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


# ---- environment (+ physical human proxy) -----------------------------------------------------------
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed
env_cfg.episode_length_s = args_cli.episode_s
HUMAN_PARK = (3.0, 0.0, 0.6)
if args_cli.human:
    env_cfg.scene.human = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HumanProxy",
        init_state=RigidObjectCfg.InitialStateCfg(pos=HUMAN_PARK, rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CapsuleCfg(
            radius=0.18, height=0.9, axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.6, 0.2)),
        ),
    )
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
N = env.num_envs
dev = env.device
origins = env.scene.env_origins
robot = env.scene["robot"]
finger_ids, _ = robot.find_joints("panda_finger_joint.*")
arm_term = env.action_manager._terms[env.action_manager.active_terms[0]]
POS_SCALE = float(arm_term.cfg.scale if not isinstance(arm_term.cfg.scale, (tuple, list)) else arm_term.cfg.scale[0])
STEP_DT = float(env.step_dt)
print(f"arm action scale {POS_SCALE}, control dt {STEP_DT}s, max commanded speed {AMAX*POS_SCALE/STEP_DT:.2f} m/s")

CFG = f"{args_cli.harness}/configs/example_action_schema.yaml"
PIN = read_digest_file(CFG + ".sha256")
human_now = {"pos": None, "vel": None}


class Perception(IsaacLabCubeStackPerceptionAdapter):
    """The real adapter; in the human condition, also tracks the physical proxy from its sim pose."""

    def get_world_state(self):
        s = super().get_world_state()
        if human_now["pos"] is None:
            return s
        p, v = human_now["pos"][self._env_index], human_now["vel"][self._env_index]
        if p[0] > 2.5:  # parked out of the room
            return s
        a = TrackedAgent("human_proxy", Pose(tuple(p)), velocity=tuple(v), tracking_confidence=0.95,
                         worst_case_speed_mps=1.5, category=AgentCategory.ADULT, stature_m=1.75)
        import dataclasses
        return dataclasses.replace(s, agents=(a,))


gates, watchdogs = [], []
if args_cli.gated:
    schema = ActionSchemaRegistry.from_yaml(CFG, expected_digest=PIN)
    for i in range(N):
        wd = DecisionWatchdog(FreezeInPlaceFallback(), deadline_s=0.2)
        watchdogs.append(wd)
        gates.append(ActuatorGate(Perception(env, env_index=i), IsaacLabCubeStackDynamicsAdapter(),
                                  FreezeInPlaceFallback(), InMemoryLogger(), schema, watchdog=wd))

# human path (per env, same timing): approach from +x to the table edge, stay, leave
T_APPROACH, T_ARRIVE, T_LEAVE, T_GONE = 60, 100, 200, 240  # control steps (20 Hz: 3s, 5s, 10s, 12s)
NEAR = (0.95, 0.0, 0.55)  # just beyond the far table edge, ~0.45m from the cubes


def human_pos(step):
    far = np.array([2.2, 0.0, 0.55])
    near = np.array(NEAR)
    if step < T_APPROACH or step >= T_GONE:
        return np.array(HUMAN_PARK)
    if step < T_ARRIVE:
        f = (step - T_APPROACH) / (T_ARRIVE - T_APPROACH)
        return far + f * (near - far)
    if step < T_LEAVE:
        return near
    f = (step - T_LEAVE) / (T_GONE - T_LEAVE)
    return near + f * (far - near)


# ---- run: exactly one episode per env ------------------------------------------------------------------
stats = {"condition": {"gated": bool(args_cli.gated), "human": bool(args_cli.human), "num_envs": N, "seed": args_cli.seed},
         "decisions": {}, "blocked_by": {}, "gate_ms": []}
with torch.inference_mode():
    env.reset()
    env.episode_length_buf[:] = 0
    experts = [Expert() for _ in range(N)]
    prev_grip = np.full(N, OPEN)
    outcome = np.full(N, -1)  # -1 running, 1 success, 0 failure/timeout
    ep_len = np.zeros(N, dtype=int)
    min_hand_human = np.full(N, 9.9)
    moved_near_human = np.zeros(N, dtype=int)  # steps the arm actually moved while the human was within 1.0m
    blocked_steps = np.zeros(N, dtype=int)
    step = 0
    cube_h = {k: env.scene[k] for k in ("cube_1", "cube_2", "cube_3")}
    last_hp = None
    while (outcome < 0).any() and step < env.max_episode_length + 5:
        if args_cli.human:
            hp = human_pos(step)
            pose = torch.zeros(N, 7, device=dev)
            pose[:, :3] = torch.tensor(hp, device=dev, dtype=torch.float32) + origins
            pose[:, 6] = 1.0
            env.scene["human"].write_root_pose_to_sim(pose)
            hv = (hp - last_hp) / STEP_DT if last_hp is not None and hp[0] < 2.5 and last_hp[0] < 2.5 else np.zeros(3)
            last_hp = hp
            human_now["pos"] = np.repeat(hp[None], N, 0)
            human_now["vel"] = np.repeat(hv[None], N, 0)
        eef = (env.scene["ee_frame"].data.target_pos_w.torch[:, 0] - origins).cpu().numpy().astype(np.float64)
        fing = robot.data.joint_pos.torch[:, finger_ids].mean(1).cpu().numpy()
        cubes = {k: (h.data.root_pos_w.torch - origins).cpu().numpy().astype(np.float64) for k, h in cube_h.items()}
        exe = np.zeros((N, 7), dtype=np.float32)
        for i in range(N):
            ex = experts[i]
            a = ex.act(eef[i], {k: v[i] for k, v in cubes.items()}, float(fing[i]))
            if outcome[i] >= 0:
                exe[i] = a * 0  # finished env: hold still
                exe[i, 6] = prev_grip[i]
                continue
            obj_name, dest_name = ex.tasks[min(ex.task_idx, 1)]
            dpos = a[:3].astype(np.float64) * POS_SCALE
            speed = float(np.linalg.norm(dpos) / STEP_DT)
            if a[6] == CLOSE and prev_grip[i] == OPEN:
                act = Action("grasp", {"object_id": obj_name, "target_position": tuple(cubes[obj_name][i]),
                                       "grip_force_n": 5.0, "commanded_speed_mps": speed})
            elif a[6] == OPEN and prev_grip[i] == CLOSE and ex.holding:
                seat = cubes[dest_name][i] + np.array([0, 0, CUBE_H])
                act = Action("place", {"object_id": obj_name, "target_surface_id": dest_name,
                                       "target_position": tuple(seat), "commanded_speed_mps": speed})
            else:
                tgt = eef[i] + dpos * (1.0 / STEP_DT)  # where this velocity leads in 1s
                act = Action("reach", {"object_id": obj_name, "target_position": tuple(tgt), "commanded_speed_mps": speed})
            if args_cli.gated:
                t0 = time.perf_counter()
                d = gates[i].gate(act)
                stats["gate_ms"].append((time.perf_counter() - t0) * 1000)
                cmd = watchdogs[i].command()
                ok = cmd.action_type != "freeze"
                key = f"{act.action_type}:{'permit' if ok else 'block'}"
                if not ok:
                    blocked_steps[i] += 1
                    for r in d.precondition_results:
                        if not r.satisfied:
                            stats["blocked_by"][r.name] = stats["blocked_by"].get(r.name, 0) + 1
                            # reason strings, normalized so they aggregate (numbers rounded away)
                            import re as _re
                            rk = r.name + ": " + _re.sub(r"-?\d+\.\d+", "#", r.reason)
                            stats.setdefault("reasons", {})
                            stats["reasons"][rk] = stats["reasons"].get(rk, 0) + 1
                    exe[i, :6] = 0.0
                    exe[i, 6] = prev_grip[i]
                else:
                    exe[i] = a
            else:
                key = f"{act.action_type}:ungated"
                exe[i] = a
            stats["decisions"][key] = stats["decisions"].get(key, 0) + 1
        moving = np.linalg.norm(exe[:, :3], axis=1) > 1e-6
        if args_cli.human and human_now["pos"][0][0] < 2.5:
            d_h = np.linalg.norm(eef - human_now["pos"], axis=1)
            min_hand_human = np.minimum(min_hand_human, d_h)
            moved_near_human += ((d_h < 1.0) & moving & (outcome < 0)).astype(int)
        prev_grip = np.where(outcome < 0, exe[:, 6], prev_grip)
        _, _, term, trunc, _ = env.step(torch.tensor(exe, device=dev))
        step += 1
        succ = env.termination_manager.get_term("success").cpu().numpy()
        done = (term | trunc).cpu().numpy() & (outcome < 0)
        for i in np.nonzero(done)[0]:
            outcome[i] = 1 if succ[i] else 0
            ep_len[i] = step
    outcome[outcome < 0] = 0

gm = np.array(stats["gate_ms"]) if stats["gate_ms"] else np.zeros(1)
n_dec = sum(stats["decisions"].values())
n_blk = sum(v for k, v in stats["decisions"].items() if k.endswith(":block"))
res = {
    **stats["condition"],
    "success": int(outcome.sum()), "success_rate": float(outcome.mean()),
    "mean_ep_len_success": float(ep_len[outcome == 1].mean()) if (outcome == 1).any() else None,
    "decisions": stats["decisions"], "decisions_total": n_dec,
    "blocked_decisions": n_blk, "block_rate": (n_blk / n_dec) if n_dec else 0.0,
    "blocked_by": stats["blocked_by"],
    "reasons": dict(sorted(stats.get("reasons", {}).items(), key=lambda kv: -kv[1])[:15]),
    "gate_ms_mean": float(gm.mean()), "gate_ms_p99": float(np.percentile(gm, 99)), "gate_ms_max": float(gm.max()),
}
if args_cli.human:
    res["min_hand_to_human_m"] = float(min_hand_human.min())
    res["mean_min_hand_to_human_m"] = float(min_hand_human.mean())
    res["arm_moving_steps_within_1m_of_human"] = int(moved_near_human.sum())
print("CLOSED_LOOP_RESULT " + json.dumps(res))
json.dump(res, open(args_cli.out, "w"), indent=1)
simulation_app.close()
