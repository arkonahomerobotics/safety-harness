"""Merge the all-phases stage-2 expert snapshots with the stage-2 (stage==1) part of the near-goal mix."""
import sys

import torch

D = "/workspace/isaaclab/snapshots"
# optional overrides (port addition): merge_stage2.py [allphases.pt neargoal_mix.pt out.pt]
A, B, OUT = sys.argv[1:4] if len(sys.argv) > 3 else (f"{D}/expert_stage2_allphases.pt", f"{D}/stage2_neargoal_mix.pt", f"{D}/stage2_allphases_neargoal.pt")
a = torch.load(A, weights_only=True)
b = torch.load(B, weights_only=True)
keep = b["stage"] == 1
out = {k: torch.cat([a[k], b[k][keep]]) for k in ("joint_pos", "cube_pose")}
torch.save(out, OUT)
print(f"MERGED allphases={len(a['joint_pos'])} neargoal={int(keep.sum())} total={len(out['joint_pos'])}")
