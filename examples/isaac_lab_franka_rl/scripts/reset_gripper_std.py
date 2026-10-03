"""Copy an rsl_rl checkpoint with the gripper action's exploration std reset (and optionally a fresh optimizer).

usage: reset_gripper_std.py <src.pt> <dst.pt> [std] [--dims -1] [--reset_optimizer] [--lr 5e-5]

Brev workflow, ported to rsl-rl-lib 5.5.1 (checkpoint layout verified on this box: top-level keys actor_state_dict /
critic_state_dict / optimizer_state_dict / iter / infos; the Gaussian std lives in
actor_state_dict["distribution.std_param"] (std_type "scalar", our cfg) or "distribution.log_std_param" (std_type
"log")). The last action dim is the gripper (BinaryJointPositionAction).

--reset_optimizer replaces the Brev ``train_reset_lr.py`` (upstream train.py with
``runner.load(..., load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True})``). The unified
Isaac Lab 3.x train.py always calls ``runner.load(path)`` with everything enabled, so instead the checkpoint itself is
rewritten: Adam's per-parameter state (moments, step counts) is emptied and every param group's lr is set to --lr.
Loading that is exactly a freshly constructed optimizer at the configured LR (Adam re-initialises empty state lazily),
and rsl_rl 5.5.1's PPO.load() copies the param-group lr into ``alg.learning_rate`` (the issue-#221 sync the Brev box
had to patch in by hand is upstream now). Iteration count, weights and obs-normalizer statistics are kept.
Pass --lr equal to the agent cfg's learning_rate (5e-5 for StackCubePPORunnerCfg).
"""

import argparse
import math
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
parser.add_argument("std", nargs="?", type=float, default=None, help="new std for --dims (Brev default was 1.0)")
parser.add_argument("--dims", type=str, default="-1", help="comma list of action dims to reset (default: gripper)")
parser.add_argument("--reset_optimizer", action="store_true")
parser.add_argument("--lr", type=float, default=5.0e-5)
args = parser.parse_args()
if args.std is None and not args.reset_optimizer:
    args.std = 1.0  # Brev CLI: "reset_gripper_std.py src dst" meant std 1.0

ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
actor = ckpt["actor_state_dict"]
if args.std is not None:
    dims = [int(d) for d in args.dims.split(",")]
    if "distribution.std_param" in actor:
        param, value = actor["distribution.std_param"], args.std
        show = lambda p: [round(float(v), 3) for v in p]  # noqa: E731
    elif "distribution.log_std_param" in actor:
        param, value = actor["distribution.log_std_param"], math.log(args.std)
        show = lambda p: [round(float(v), 3) for v in p.exp()]  # noqa: E731
    else:
        raise KeyError(f"no distribution std parameter in actor_state_dict: {list(actor)}")
    print("std before:", show(param))
    for d in dims:
        param[d] = value
    print("std after: ", show(param))
if args.reset_optimizer:
    opt = ckpt["optimizer_state_dict"]
    print(f"optimizer: dropping state for {len(opt['state'])} params, lr {[g['lr'] for g in opt['param_groups']]} -> {args.lr}")
    opt["state"] = {}
    for g in opt["param_groups"]:
        g["lr"] = args.lr
print("iteration kept:", ckpt.get("iter"))
os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
torch.save(ckpt, args.dst)
print("saved", args.dst)
