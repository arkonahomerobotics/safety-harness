"""Keep only near-goal stage-2 snapshots (green just above red, centered) and mix with some stage-1 snapshots.

usage: filter_snapshots.py <stage2.pt> <stage1.pt> <out.pt> <max_gap_cm> <max_xy_cm> <n_stage1>
Heights use the true 4.68 cm cube (seated green center = red center + 4.68 cm).
"""

import sys

import torch

s2, s1 = torch.load(sys.argv[1], weights_only=True), torch.load(sys.argv[2], weights_only=True)
out, max_gap, max_xy, n1 = sys.argv[3], float(sys.argv[4]) / 100, float(sys.argv[5]) / 100, int(sys.argv[6])
cp = s2["cube_pose"]
gap = cp[:, 2, 2] - cp[:, 1, 2] - 0.0468
xy = torch.linalg.norm(cp[:, 2, :2] - cp[:, 1, :2], dim=1)
red_seated = ((cp[:, 1, 2] - cp[:, 0, 2] - 0.0468).abs() < 0.004) & (torch.linalg.norm(cp[:, 1, :2] - cp[:, 0, :2], dim=1) < 0.01)
keep = (gap > -0.002) & (gap < max_gap) & (xy < max_xy) & red_seated
near = {k: v[keep] for k, v in s2.items()}
idx = torch.randperm(len(s1["joint_pos"]))[:n1]
far = {k: v[idx] for k, v in s1.items() if k in near}
merged = {k: torch.cat([near[k], far[k]]) for k in near}
merged["stage"] = torch.cat([torch.ones(len(near["joint_pos"]), dtype=torch.long), torch.zeros(len(idx), dtype=torch.long)])
torch.save(merged, out)
g = gap[keep] * 100
print(f"FILTERED near-goal stage-2 kept={int(keep.sum())}/{len(keep)} (gap {g.min():.2f}..{g.max():.2f} cm, median {g.median():.2f}) + stage-1 {len(idx)} -> {out}")
