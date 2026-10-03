"""stage2_v3: handoff-heavy mix (handoff states x10 + expert stage-2 states)."""
import sys

import torch

D = "/workspace/isaaclab/snapshots"
# optional overrides (port addition): merge_stage2_v3.py [handoff.pt stage2_allphases_neargoal.pt out.pt]
H, A, OUT = sys.argv[1:4] if len(sys.argv) > 3 else (f"{D}/handoff_G4100.pt", f"{D}/stage2_allphases_neargoal.pt", f"{D}/stage2_v3.pt")
h = torch.load(H, weights_only=True)
a = torch.load(A, weights_only=True)
cp = h["cube_pose"]
ok = ((cp[:, 1, 2] - cp[:, 0, 2] - 0.0468).abs() < 0.004) & (torch.linalg.norm(cp[:, 1, :2] - cp[:, 0, :2], dim=1) < 0.015)
out = {k: torch.cat([h[k][ok].repeat(10, *[1] * (h[k].dim() - 1)), a[k]]) for k in ("joint_pos", "cube_pose")}
torch.save(out, OUT)
print(f"MERGED handoff {int(ok.sum())} x10 + expert {len(a['joint_pos'])} = {len(out['joint_pos'])}")
