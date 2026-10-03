"""Full three-cube tower from normal starts by skill chaining: the stage-1 policy (red onto blue) runs until
red has stood stacked and released for --handoff_steps consecutive steps, then that env switches to the
stage-2 skill policy, which sees the rebound cube roles (cube_1 := red, cube_2 := green, cube_3 := blue).
The switch is latched. Expert start states are disabled.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-RL-FullStack-Snap-Sparse-v0")
parser.add_argument("--stage1", type=str, required=True)
parser.add_argument("--stage2", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--handoff_steps", type=int, default=10)
parser.add_argument("--episode_s", type=float, default=0.0, help="override episode length (0 = task default)")
parser.add_argument("--stochastic", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import copy

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.stack.mdp.robosuite_rewards import _in_contact
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.events.reset_from_snapshot = None
if args_cli.episode_s > 0:
    env_cfg.episode_length_s = args_cli.episode_s
s2 = copy.deepcopy(env_cfg.observations.policy)
roles = {"cube_1_cfg": SceneEntityCfg("cube_2"), "cube_2_cfg": SceneEntityCfg("cube_3"), "cube_3_cfg": SceneEntityCfg("cube_1")}
for term in ("object", "cube_positions", "cube_orientations"):
    getattr(s2, term).params.update(roles)
env_cfg.observations.policy_s2 = s2
# extra contact sensors for failure attribution (what touches the red-on-blue stack when it falls)
from isaaclab.sensors import ContactSensorCfg  # noqa: E402

env_cfg.scene.cube_1.spawn = env_cfg.scene.cube_1.spawn.replace(activate_contact_sensors=True)
# robot link names resolve via the sensor's subtree search (links are nested under Robot/Geometry/... in 3.x)
from isaaclab_tasks.contrib.stack.config.franka.stack_ik_rel_rl_env_cfg import _BLUE, _GREEN, _RED, _ROBOT_LINK  # noqa: E402

_R = _ROBOT_LINK
_C = {1: _BLUE, 2: _RED, 3: _GREEN}
env_cfg.scene.x_green_blue = ContactSensorCfg(prim_path=_C[3], filter_prim_paths_expr=[_C[1]])
env_cfg.scene.x_hand_red = ContactSensorCfg(prim_path=_R + "panda_hand", filter_prim_paths_expr=[_C[2]])
env_cfg.scene.x_hand_blue = ContactSensorCfg(prim_path=_R + "panda_hand", filter_prim_paths_expr=[_C[1]])
env_cfg.scene.x_lf_blue = ContactSensorCfg(prim_path=_R + "panda_leftfinger", filter_prim_paths_expr=[_C[1]])
env_cfg.scene.x_rf_blue = ContactSensorCfg(prim_path=_R + "panda_rightfinger", filter_prim_paths_expr=[_C[1]])
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
pols = []
for ck in (args_cli.stage1, args_cli.stage2):
    r = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    r.load(ck, map_location=agent_cfg.device)
    pols.append(r.get_inference_policy(device=u.device))
LIFT_Z, THR = 0.0403, 0.01


def states():
    global aux
    oz = u.scene.env_origins[:, 2]
    red_z = u.scene["cube_2"].data.root_pos_w.torch[:, 2] - oz
    green_z = u.scene["cube_3"].data.root_pos_w.torch[:, 2] - oz
    g_red = _in_contact(u, "left_finger_red_contact", THR) & _in_contact(u, "right_finger_red_contact", THR)
    g_green = _in_contact(u, "left_finger_green_contact", THR) & _in_contact(u, "right_finger_green_contact", THR)
    s1 = (red_z > LIFT_Z) & _in_contact(u, "red_blue_contact", THR) & ~g_red
    tower = s1 & (green_z > LIFT_Z) & _in_contact(u, "green_red_contact", THR) & ~g_green
    ee_z = u.scene["ee_frame"].data.target_pos_w.torch[:, 0, 2] - oz
    aux = dict(g_green=g_green, g_red=g_red, green_z=green_z, red_z=red_z, ee_z=ee_z,
               g_red_any=_in_contact(u, "left_finger_red_contact", THR) | _in_contact(u, "right_finger_red_contact", THR),
               green_on_red=(green_z > LIFT_Z) & _in_contact(u, "green_red_contact", THR),
               x_green_red=_in_contact(u, "green_red_contact", THR), x_green_blue=_in_contact(u, "x_green_blue", THR),
               x_hand_red=_in_contact(u, "x_hand_red", THR), x_hand_blue=_in_contact(u, "x_hand_blue", THR),
               x_finger_blue=_in_contact(u, "x_lf_blue", THR) | _in_contact(u, "x_rf_blue", THR),
               red_xy_off=torch.linalg.norm(u.scene["cube_2"].data.root_pos_w.torch[:, :2] - u.scene["cube_1"].data.root_pos_w.torch[:, :2], dim=1),
               green_blue_xy=torch.linalg.norm(u.scene["cube_3"].data.root_pos_w.torch[:, :2] - u.scene["cube_1"].data.root_pos_w.torch[:, :2], dim=1))
    return s1, tower


n, dev = u.num_envs, u.device
T = int(u.max_episode_length)
finished = torch.zeros(n, dtype=torch.bool, device=dev)
early = torch.zeros(n, dtype=torch.bool, device=dev)
s1_ever = torch.zeros(n, dtype=torch.bool, device=dev)
tower_ever = torch.zeros(n, dtype=torch.bool, device=dev)
s1_last = torch.zeros(n, dtype=torch.bool, device=dev)
tower_last = torch.zeros(n, dtype=torch.bool, device=dev)
s1_run = torch.zeros(n, dtype=torch.long, device=dev)
handed = torch.zeros(n, dtype=torch.bool, device=dev)
hand_t = torch.full((n,), -1, dtype=torch.long, device=dev)
first_tower = torch.full((n,), -1, dtype=torch.long, device=dev)
# failure bookkeeping: first loss of red-on-blue after handoff, and what the gripper was doing then
lost_t = torch.full((n,), -1, dtype=torch.long, device=dev)
lost_green_held = torch.zeros(n, dtype=torch.bool, device=dev)
lost_finger_on_red = torch.zeros(n, dtype=torch.bool, device=dev)
lost_green_z = torch.zeros(n, device=dev)
end_green_held = torch.zeros(n, dtype=torch.bool, device=dev)
end_green_on_red = torch.zeros(n, dtype=torch.bool, device=dev)
end_red_z = torch.zeros(n, device=dev)
prev_s1 = torch.zeros(n, dtype=torch.bool, device=dev)
prev_aux = None
TOUCH = ("g_red_any", "x_green_red", "x_green_blue", "x_hand_red", "x_hand_blue", "x_finger_blue")
touch_win = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in TOUCH}  # any contact in the 3 steps before loss
hist = []
hand_red_off = torch.zeros(n, device=dev)
hand_green_blue = torch.zeros(n, device=dev)
tail = []

obs = env.get_observations()
with torch.inference_mode():
    for t in range(T):
        s1, tower = states()
        s1, tower = s1 & ~finished, tower & ~finished
        s1_run = torch.where(s1, s1_run + 1, torch.zeros_like(s1_run))
        new = ~handed & (s1_run >= args_cli.handoff_steps) & ~finished
        hand_t = torch.where(new, torch.full_like(hand_t, t), hand_t)
        handed |= new
        hand_red_off = torch.where(new, aux["red_xy_off"], hand_red_off)
        hand_green_blue = torch.where(new, aux["green_blue_xy"], hand_green_blue)
        hist = (hist + [{k: aux[k] for k in TOUCH}])[-4:]
        first_tower = torch.where(tower & ~tower_ever, torch.full_like(first_tower, t), first_tower)
        s1_ever |= s1
        tower_ever |= tower
        lost = handed & prev_s1 & ~s1 & ~aux["g_red"] & (lost_t < 0) & ~finished & (hand_t < t)
        if prev_aux is not None:
            lost_green_held = torch.where(lost, prev_aux["g_green"], lost_green_held)
            lost_finger_on_red = torch.where(lost, prev_aux["g_red_any"], lost_finger_on_red)
            lost_green_z = torch.where(lost, prev_aux["green_z"], lost_green_z)
        for k in TOUCH:
            anyk = torch.stack([h[k] for h in hist]).any(0)
            touch_win[k] = torch.where(lost, anyk, touch_win[k])
        lost_t = torch.where(lost, t - hand_t, lost_t)
        prev_s1, prev_aux = s1, aux
        live = ~finished
        end_green_held = torch.where(live, aux["g_green"], end_green_held)
        end_green_on_red = torch.where(live, aux["green_on_red"], end_green_on_red)
        end_red_z = torch.where(live, aux["red_z"], end_red_z)
        s1_last = torch.where(finished, s1_last, s1)
        tower_last = torch.where(finished, tower_last, tower)
        if t >= T - 50:
            tail.append(tower.float())
        obs2 = obs.clone()
        obs2["policy"] = obs["policy_s2"]
        if args_cli.stochastic:
            a1, a2 = pols[0](obs, stochastic_output=True), pols[1](obs2, stochastic_output=True)
        else:
            a1, a2 = pols[0](obs), pols[1](obs2)
        actions = torch.where(handed.unsqueeze(1), a2, a1)
        obs, _, dones, extras = env.step(actions)
        for p in pols:
            p.reset(dones)
        done = dones.bool() & ~finished
        early |= done & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
        finished |= done

ft = first_tower[first_tower >= 0].float()
ht = hand_t[hand_t >= 0].float()
print("=" * 90)
print(f"CHAIN stage1={args_cli.stage1}")
print(f"CHAIN stage2={args_cli.stage2}")
print(f"CHAIN envs={n} episode_len={T} normal starts, {'stochastic' if args_cli.stochastic else 'deterministic'}, handoff after {args_cli.handoff_steps} stacked steps")
print(f"CHAIN handed off to stage 2: {handed.float().mean():.3f}" + (f" (step median {ht.median():.0f}, p90 {ht.quantile(0.9):.0f})" if len(ht) else ""))
print(f"CHAIN red on blue at end: {s1_last.float().mean():.3f}   ever: {s1_ever.float().mean():.3f}")
print(f"CHAIN TOWER at end:       {tower_last.float().mean():.3f}   ever: {tower_ever.float().mean():.3f}")
print(f"CHAIN share of last 50 steps with tower standing: {torch.stack(tail).mean():.3f}")
if len(ft):
    print(f"CHAIN first tower step: median={ft.median():.0f} p90={ft.quantile(0.9):.0f}")
print(f"CHAIN early terminations: {early.float().mean():.3f}")
fail = ~tower_last
red_off = fail & ~s1_last
print(f"CHAIN FAILURES (share of all envs): no tower at end {fail.float().mean():.3f}")
print(f"CHAIN   red not on blue at end:                 {red_off.float().mean():.3f}"
      f"  (red on table: {(red_off & (end_red_z < LIFT_Z)).float().mean():.3f}; green then stacked on that red: {(red_off & end_green_on_red).float().mean():.3f})")
print(f"CHAIN   red on blue, green on red but still held: {(fail & s1_last & end_green_on_red & end_green_held).float().mean():.3f}")
print(f"CHAIN   red on blue, green held elsewhere:        {(fail & s1_last & ~end_green_on_red & end_green_held).float().mean():.3f}")
print(f"CHAIN   red on blue, green loose not on red:      {(fail & s1_last & ~end_green_on_red & ~end_green_held).float().mean():.3f}")
print(f"CHAIN   never handed off:                          {(fail & ~handed).float().mean():.3f}")
lt = lost_t >= 0
if lt.any():
    print(f"CHAIN RED KNOCKED OFF after handoff in {lt.float().mean():.3f} of envs; steps after handoff median {lost_t[lt].float().median():.0f}")
    print(f"CHAIN   at that moment: green held {lost_green_held[lt].float().mean():.2f}, a finger touching red {lost_finger_on_red[lt].float().mean():.2f}, "
          f"green height above seated-on-red level median {(lost_green_z[lt] - 0.0203 - 2 * 0.0468).median() * 100:+.1f}cm")
    print("CHAIN   contacts in the 3 steps before red fell: " + ", ".join(f"{k}={touch_win[k][lt].float().mean():.2f}" for k in TOUCH))
    ok = handed & ~lt
    print(f"CHAIN   at handoff, red xy offset on blue: knocked-off envs median {hand_red_off[lt].median() * 100:.2f}cm vs others {hand_red_off[ok].median() * 100:.2f}cm")
    print(f"CHAIN   at handoff, green-to-blue xy distance: knocked-off envs median {hand_green_blue[lt].median() * 100:.1f}cm vs others {hand_green_blue[ok].median() * 100:.1f}cm")
    for lo, hi in ((0, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 1.0)):
        b = handed & (hand_green_blue >= lo) & (hand_green_blue < hi)
        if b.any():
            print(f"CHAIN     green {lo * 100:.0f}-{hi * 100:.0f}cm from blue at handoff: n={int(b.sum())} knocked-off {lt[b].float().mean():.2f} tower-at-end {tower_last[b].float().mean():.2f}")
print("=" * 90)
env.close()
simulation_app.close()
