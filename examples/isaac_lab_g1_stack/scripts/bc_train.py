"""Behavior-clone the scripted G1 stacking expert into an RSL-RL-shaped actor, then write it into a
PPO checkpoint so PPO fine-tunes from it instead of from scratch (from-scratch PPO never found the
Dex3 palm-pocket grasp: 0% lift after 250 iterations).

Data: (policy obs, clean expert action) pairs from stack_expert_rl.py --record, DART-style (executed
actions were noised, labels are clean), successful episodes only.

Output checkpoint = a template checkpoint of the target run config (same shapes) with the actor's
obs normalizer, MLP and action std replaced; the critic and optimizer are left at their fresh init.
"""

import argparse
import glob

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="/workspace/isaaclab/bc_data/demo_*.pt")
parser.add_argument("--template", required=True, help="a model_0.pt from a run with the target network shape")
parser.add_argument("--out", required=True)
parser.add_argument("--hidden", default="512,256,128")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--init_std", type=float, default=0.15)
parser.add_argument("--grip_weight", type=float, default=4.0)
args = parser.parse_args()

dev = "cuda"
files = sorted(glob.glob(args.data))
O = torch.cat([torch.load(f)["obs"] for f in files]).float()
A = torch.cat([torch.load(f)["act"] for f in files]).float()
print(f"{len(files)} files, {O.shape[0]} pairs, obs {O.shape[1]}, act {A.shape[1]}")

perm = torch.randperm(O.shape[0])
n_val = O.shape[0] // 20
val, tr = perm[:n_val], perm[n_val:]
mean = O[tr].mean(0, keepdim=True)
var = O[tr].var(0, unbiased=False, keepdim=True)
std = var.sqrt()
# must match rsl_rl EmpiricalNormalization EXACTLY: forward is (x - mean) / (sqrt(var) + 1e-2), and
# every update recomputes _std = sqrt(_var). An earlier version normalized with sqrt(var + 1e-2) -- ~10x
# off on near-constant dims -- and the first PPO normalizer update scrambled the cloned policy (49% -> 0%).
denom = std + 1e-2

hidden = [int(h) for h in args.hidden.split(",")]
layers, d = [], O.shape[1]
for h in hidden:
    layers += [nn.Linear(d, h), nn.ELU()]
    d = h
layers.append(nn.Linear(d, A.shape[1]))
mlp = nn.Sequential(*layers).to(dev)

Ot, At = ((O - mean) / denom).to(dev), A.to(dev)
w = torch.ones(A.shape[1], device=dev)
w[-1] = args.grip_weight  # the grip decision (close / open) matters more than a millimeter of arm motion
opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
tr_d, val_d = tr.to(dev), val.to(dev)
for ep in range(args.epochs):
    mlp.train()
    idx = tr_d[torch.randperm(len(tr_d), device=dev)]
    tot = 0.0
    for b in idx.split(4096):
        loss = (((mlp(Ot[b]) - At[b]) ** 2) * w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item() * len(b)
    sched.step()
    if ep % 5 == 4 or ep == args.epochs - 1:
        mlp.eval()
        with torch.no_grad():
            pv = mlp(Ot[val_d])
            vl = (((pv - At[val_d]) ** 2) * w).mean().item()
            grip_acc = ((pv[:, -1] > 0) == (At[val_d][:, -1] > 0)).float().mean().item()
        print(f"epoch {ep+1:3d} train {tot/len(tr_d):.4f} val {vl:.4f} grip-sign acc {grip_acc:.4f}")

ck = torch.load(args.template, map_location="cpu", weights_only=False)
actor = ck["actor_state_dict"]
actor["obs_normalizer._mean"] = mean.clone()
actor["obs_normalizer._var"] = var.clone()
actor["obs_normalizer._std"] = std.clone()
actor["obs_normalizer.count"] = torch.tensor(O.shape[0] * 100, dtype=torch.long)  # heavy prior: PPO shouldn't drift it much
lin = [m for m in mlp if isinstance(m, nn.Linear)]
for i, m in enumerate(lin):
    k = 2 * i
    assert actor[f"mlp.{k}.weight"].shape == m.weight.shape, (k, actor[f"mlp.{k}.weight"].shape, m.weight.shape)
    actor[f"mlp.{k}.weight"] = m.weight.detach().cpu().clone()
    actor[f"mlp.{k}.bias"] = m.bias.detach().cpu().clone()
actor["distribution.std_param"] = torch.full_like(actor["distribution.std_param"], args.init_std)
print("normalizer check: max |our input - rsl_rl input| =",
      float(((O[:1000] - mean) / denom - (O[:1000] - actor["obs_normalizer._mean"]) / (actor["obs_normalizer._std"] + 1e-2)).abs().max()))
# critic sees the same obs: give it the same normalizer so its inputs are sane from step 0
for k in ("_mean", "_var", "_std"):
    ck["critic_state_dict"][f"obs_normalizer.{k}"] = actor[f"obs_normalizer.{k}"].clone()
ck["critic_state_dict"]["obs_normalizer.count"] = actor["obs_normalizer.count"].clone()
ck["iter"] = 0
torch.save(ck, args.out)
print("wrote", args.out)
