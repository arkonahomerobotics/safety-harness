"""Behavior-clone the scripted Franka stacking expert into the rsl_rl PPO actor and write an rsl_rl 5.5.1 checkpoint
that ``train.py --checkpoint`` loads as a warm start (the final policy is still trained by PPO).

Data: collect_bc_franka.py output(s) -- (RL-task policy obs [94], clean expert action in the RL task's action space
[7]), DART-noised execution, successful episodes only.

Network = exactly the PPO actor of StackCubePPORunnerCfg: EmpiricalNormalization (rsl_rl's formula,
(x - mean) / (sqrt(var) + 1e-2)) -> MLP [256, 128, 64] ELU -> 7 action means. The checkpoint skeleton is a FRESH
iteration-0 checkpoint of the same agent cfg (collect_bc_franka.py --template_out), so the critic stays at its init and
the optimizer state is empty at the cfg LR; only the actor's normalizer, MLP and std are replaced. The critic gets the
same normalizer statistics so its inputs are sane from step 0 (its weights are untouched). Normalizer ``count`` is set
to 100x the dataset size so PPO's running updates barely move it (a shifting normalizer would scramble the cloned
mapping -- the G1 lesson).

Runs on CPU by default (the GPU is shared with live training).
"""

import argparse
import glob

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="glob of collect_bc_franka.py outputs")
parser.add_argument("--template", required=True, help="fresh iteration-0 checkpoint of the target agent cfg")
parser.add_argument("--out", required=True)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--std_arm", type=float, default=0.3)
parser.add_argument("--std_grip", type=float, default=0.3)
parser.add_argument("--grip_weight", type=float, default=2.0)
parser.add_argument("--shuffle_last_action", type=float, default=1.0,
                    help="probability of replacing the last-action obs dims (0..6) with another sample's during training")
parser.add_argument("--device", default="cpu")
parser.add_argument("--threads", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.set_num_threads(args.threads)
dev = args.device

files = sorted(glob.glob(args.data))
O = torch.cat([torch.load(f, weights_only=False)["obs"] for f in files]).float()
A = torch.cat([torch.load(f, weights_only=False)["act"] for f in files]).float()
print(f"{len(files)} files, {O.shape[0]} pairs, obs {O.shape[1]}, act {A.shape[1]}")
print("label |mean| per dim:", [round(float(v), 3) for v in A.abs().mean(0)],
      " max:", [round(float(v), 2) for v in A.abs().max(0).values])
print("gripper label: close share", round(float((A[:, -1] < 0).float().mean()), 3))

ck = torch.load(args.template, map_location="cpu", weights_only=False)
actor = ck["actor_state_dict"]
lin_keys = sorted({k.rsplit(".", 1)[0] for k in actor if k.startswith("mlp.") and k.endswith(".weight")},
                  key=lambda s: int(s.split(".")[1]))
shapes = [tuple(actor[f"{k}.weight"].shape) for k in lin_keys]
print("template actor MLP:", list(zip(lin_keys, shapes)))
assert shapes[0][1] == O.shape[1] and shapes[-1][0] == A.shape[1], "template does not match data dims"

LAST_ACT = A.shape[1]  # policy obs starts with mdp.last_action (the ObservationsCfg.PolicyCfg term order)
lag = (O[1:, :3] - A[:-1, :3]).abs().mean()
print(f"obs[:, :3] vs previous label xyz: mean |diff| {float(lag):.3f} (~noise if the layout is right), "
      f"vs same-step label {float((O[:, :3] - A[:, :3]).abs().mean()):.3f}")

perm = torch.randperm(O.shape[0])
n_val = O.shape[0] // 20
val, tr = perm[:n_val], perm[n_val:]
mean = O[tr].mean(0, keepdim=True)
var = O[tr].var(0, unbiased=False, keepdim=True)
std = var.sqrt()
denom = std + 1e-2  # rsl_rl EmpiricalNormalization.forward, exactly

layers = []
for i, (out_d, in_d) in enumerate(shapes):
    layers.append(nn.Linear(in_d, out_d))
    if i < len(shapes) - 1:
        layers.append(nn.ELU())
mlp = nn.Sequential(*layers).to(dev)

Ot, At = ((O - mean) / denom).to(dev), A.to(dev)
w = torch.ones(A.shape[1], device=dev)
w[-1] = args.grip_weight
opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
tr_d, val_d = tr.to(dev), val.to(dev)
for ep in range(args.epochs):
    mlp.train()
    idx = tr_d[torch.randperm(len(tr_d), device=dev)]
    tot = 0.0
    for b in idx.split(args.batch):
        x = Ot[b]
        if args.shuffle_last_action > 0:
            # anti-copycat: the obs carries the previous (executed) action, and labels are smooth in time, so a
            # cloned net learns "repeat the last action" and never starts the lift / release (diag_bc_franka.py:
            # v1 grasped in 100% of envs and lifted in 8%). Swapping these dims with another sample's makes the net
            # predict from the physical state; PPO can still learn to use them afterwards.
            x = x.clone()
            swap = torch.rand(len(b), device=dev) < args.shuffle_last_action
            donor = tr_d[torch.randint(len(tr_d), (int(swap.sum()),), device=dev)]
            x[swap, :LAST_ACT] = Ot[donor, :LAST_ACT]
        loss = (((mlp(x) - At[b]) ** 2) * w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item() * len(b)
    sched.step()
    if ep % 5 == 4 or ep == args.epochs - 1:
        mlp.eval()
        with torch.no_grad():
            pv = torch.cat([mlp(Ot[b]) for b in val_d.split(65536)])
            av = At[val_d]
            per_dim = ((pv - av) ** 2).mean(0)
            grip_acc = ((pv[:, -1] > 0) == (av[:, -1] > 0)).float().mean().item()
        print(f"epoch {ep + 1:3d} train {tot / len(tr_d):.5f} val {(per_dim * w).mean():.5f} "
              f"val mse xyz {[round(float(v), 5) for v in per_dim[:3]]} rot {float(per_dim[3:6].mean()):.6f} "
              f"grip {float(per_dim[6]):.4f} grip-sign acc {grip_acc:.4f}", flush=True)

actor["obs_normalizer._mean"] = mean.clone()
actor["obs_normalizer._var"] = var.clone()
actor["obs_normalizer._std"] = std.clone()
actor["obs_normalizer.count"] = torch.tensor(O.shape[0] * 100, dtype=torch.long)
lin = [m for m in mlp if isinstance(m, nn.Linear)]
for k, m in zip(lin_keys, lin):
    assert actor[f"{k}.weight"].shape == m.weight.shape
    actor[f"{k}.weight"] = m.weight.detach().cpu().clone()
    actor[f"{k}.bias"] = m.bias.detach().cpu().clone()
std_p = torch.full_like(actor["distribution.std_param"], args.std_arm)
std_p[-1] = args.std_grip
actor["distribution.std_param"] = std_p
for k in ("_mean", "_var", "_std", "count"):
    ck["critic_state_dict"][f"obs_normalizer.{k}"] = actor[f"obs_normalizer.{k}"].clone()
ck["iter"] = 0
assert not ck["optimizer_state_dict"]["state"], "template optimizer is not fresh"

# sanity: rsl_rl-style forward from the written state dict reproduces our network
with torch.no_grad():
    x = O[val[:2000]]
    ours = mlp.cpu()((x - mean) / denom)
    h = (x - actor["obs_normalizer._mean"]) / (actor["obs_normalizer._std"] + 1e-2)
    for i, k in enumerate(lin_keys):
        h = h @ actor[f"{k}.weight"].T + actor[f"{k}.bias"]
        if i < len(lin_keys) - 1:
            h = torch.nn.functional.elu(h)
    print("state-dict forward check: max |diff| =", float((ours - h).abs().max()))
print("std_param:", [round(float(v), 3) for v in actor["distribution.std_param"]])
torch.save(ck, args.out)
print("wrote", args.out)
