"""Mix two snapshot files into one. usage: merge_snapshots.py <a.pt> <b.pt> <out.pt> <n_a> <n_b>"""

import sys

import torch

a, b = torch.load(sys.argv[1], weights_only=True), torch.load(sys.argv[2], weights_only=True)
out, n_a, n_b = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])


def take(s, k):
    idx = torch.randperm(len(s["joint_pos"]))[: min(k, len(s["joint_pos"]))]
    return {key: v[idx] for key, v in s.items()}


a, b = take(a, n_a), take(b, n_b)
merged = {key: torch.cat([a[key], b[key]]) for key in ("joint_pos", "cube_pose", "phase")}
merged["stage"] = torch.cat([torch.zeros(len(a["joint_pos"]), dtype=torch.long), torch.ones(len(b["joint_pos"]), dtype=torch.long)])
torch.save(merged, out)
print(f"MERGED {len(a['joint_pos'])} + {len(b['joint_pos'])} -> {out}")
