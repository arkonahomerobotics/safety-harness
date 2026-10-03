# Franka chained-RL 20-episode acceptance protocol

What `accept_20.py` runs, why each piece of the protocol is there, and how to read its output.
Written so the eventual `ACCEPT k/20` claim is reproducible by someone who wasn't in the room when
it was produced, not just printed and trusted.

**Status: not yet run against a real checkpoint.** No stage-1/stage-2 checkpoint exists for this
port yet (see the parent `README.md`'s "Current status") — this is the gate to run the moment one
does, not a result.

## Protocol

- **20 episodes, sequential, one environment (`num_envs=1`).** Not 1024-env aggregate statistics
  (that's what `eval_chain.py` is for) — this produces the same kind of individually-inspectable,
  individually-reproducible proof run as the G1 example's hazard campaign and hand-picked demo clips.
- **A different seed per episode** (`--seed_base`, default 100, so episode `i` uses seed
  `100 + i`), via `torch.manual_seed(seed)` immediately before an explicit `env.reset()` for that
  episode. Not relying on Isaac Lab's own auto-reset-on-done for episode-to-episode variation,
  because that reset's randomness is drawn *before* this script gets control back — there's no way
  to inject a specific seed into a reset that already happened inside the previous episode's last
  `env.step()` call.
- **Deterministic actions.** Both policies are called without `stochastic_output=True` — the same
  default `eval_chain.py` and `record_chain.py` use.
- **Natural resets only** (`env_cfg.events.reset_from_snapshot = None`), same as `eval_chain.py` —
  no expert start states, no reverse-curriculum snapshots.
- **The exact chained stage-1 → stage-2 handoff** from `eval_chain.py`: stage 1 runs until red has
  stood stacked on blue, ungripped, for `--handoff_steps` (default 10) consecutive steps; from then
  on the episode runs stage 2 with the rebound cube-role observation (`cube_1 := red, cube_2 :=
  green, cube_3 := blue`).
- **Full episode length.** No `--episode_s` override exists in this script at all (unlike
  `eval_chain.py`, which has one) — there's nothing to accidentally set it, which also means this
  protocol can never be quietly made easier by shortening the episode.

## The one thing this file exists to get right

Isaac Lab auto-resets an environment **inside** the very `step()` call whose `dones` comes back
`True` — not on some later call. Reading simulation state (`u.scene[...]`) on *any* iteration after
that is already reading the *next* episode's start state, not the one that just ended. This is
exactly the bug that produced a bogus 0/N result on the `isaac_lab_g1_stack` example before it was
caught and fixed.

`eval_chain.py` already avoids this correctly by computing success/failure state (`states()`) at the
*top* of every loop iteration — i.e. always before that iteration's own `env.step()` — and only ever
latching its "last" / failure-attribution values from that pre-step reading. `accept_20.py` copies
that discipline, with one further simplification specific to running one episode at a time instead
of many envs in parallel: the instant `dones.item()` is `True`, the step loop `break`s immediately,
*before* ever calling `states()` again. Whatever was latched in that same iteration — right before
the terminating action's effect was committed — is the judged final state. The result: no
`torch.where(finished, ...)` masking is needed anywhere in this file (unlike `eval_chain.py`, which
needs it to keep already-finished envs from corrupting the batch), because the loop simply never
reaches another `states()` call for a finished episode at all.

## Success definition

"Tower standing" — the same definition `eval_chain.py` reports as `CHAIN TOWER at end`: red stacked
on blue (lifted above `LIFT_Z`, in contact, fingers off it) **and** green stacked on red the same
way, judged on the last pre-reset reading as described above. This is "the env's own success
definition" in the sense that it's the one quantity this whole project already reports as the
chained-task success rate (the Brev box's 92.3% in the parent README's "Brev results" table was
computed by this exact same `states()` logic in `eval_chain.py`) — not a separate, newly-invented
criterion.

## Output

- One `EPISODE k/20 seed=... -> PASS|FAIL: <reason>` line per episode, printed as it finishes.
- A final table: per-episode seed, PASS/FAIL, whether a tower was ever formed at any point in the
  episode (even if it didn't survive to the end — eval_chain.py's `ever` vs. `last` distinction),
  whether/when the stage-2 handoff happened, and the failure-attribution reason.
- Two summary lines: `Tower standing at episode end: k/20   Tower achieved at any point: k_ever/20`,
  then the final `ACCEPT k/20` line.
- **Failure attribution** (`classify()` in `accept_20.py`), checked in this order: never handed off
  to stage 2; red knocked off blue after handoff (with whether green was already held at that
  moment); red never made it onto blue at all (on the table, or displaced by green landing on it
  instead); red on blue but green still held, or held somewhere else, or loose but not placed, at
  episode end.

## Video (`--video`)

One mp4 per episode (`<video_dir>/episode_NN.mp4`), not one combined clip — so a specific episode's
proof can be sent on its own. Same 1280×720 demo camera, at the same Brev viewer pose, as
`record_chain.py`. Every frame carries a burned-in caption, "Franka 3-cube RL policy, episode k/20,
SUCCESS|FAIL", colored green or red by the episode's own result — computed once the episode's
outcome is known and applied to that episode's buffered frames before writing, not decided live
frame-by-frame.

## Running it

```bash
./isaaclab.sh -p accept_20.py --stage1 <stage1_checkpoint>.pt --stage2 <stage2_checkpoint>.pt
# with per-episode video:
./isaaclab.sh -p accept_20.py --stage1 <stage1_checkpoint>.pt --stage2 <stage2_checkpoint>.pt \
    --video --video_dir videos/accept_20
```

`--episodes` and `--seed_base` are both overridable, but the 20-episode, seed-100-based run is the
one this protocol document (and the `ACCEPT k/20` claim it produces) describes — change them only to
investigate a specific failure, not to produce the headline number.
