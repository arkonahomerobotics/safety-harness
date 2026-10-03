# Security Policy

This is a small, reference-implementation project maintained by one person, not a company with a
dedicated security team -- said plainly here rather than implied, the same way the README and
[design doc](docs/design.md) disclose every other limitation of this project. What follows is what
that one maintainer can actually commit to, not a template response-time promise.

## What counts as a vulnerability here

`safety-harness` is a default-deny precondition gate that a robot's actuators are meant to trust:
its entire job is to `BLOCK` an unsafe action rather than `PERMIT` it. Two different kinds of bugs
both matter here, and both should be reported through this process:

- **Conventional software vulnerabilities** -- anything that could let untrusted input execute
  code, corrupt memory, exfiltrate data, or otherwise compromise a system running this package
  (dependency vulnerabilities included -- see [`examples/sbom/`](examples/sbom/) for the current
  dependency list, currently just `pyyaml`).
- **Safety-logic defects that cause a wrong `PERMIT`.** A precondition check that silently passes
  on bad input is, for this project, at least as serious as a conventional vulnerability -- it's
  the failure mode the entire design exists to prevent. If you find a `WorldState` or `Action`
  input that makes `ActuatorGate.gate()` return `PERMIT` when a reasonable reading of the
  registered checks says it should `BLOCK`, that's a security report, not just a bug report.

There's real precedent for the second category: the design doc's ["NaN-Sensor Stress
Test"](docs/design.md#nan-sensor-stress-test-2026-09-27-a-comparison-based-default-deny-bypass)
section documents a systemic default-deny bypass this project found in its own adversarial
testing -- IEEE-754 comparisons against a NaN sensor value silently pass a `<`/`>` guard that was
supposed to fail closed, letting several checks (`mass_within_force_budget`,
`object_hazard_confirmed`, `object_pose_confirmed`, and compounding into a full `gate()` PERMIT
next to an under-tracked agent) report `satisfied=True` on corrupted input. It was fixed in
`preconditions.py`, validated three separate ways (the full test suite locally, the same checkout
against a remote GPU box's own Python, and five live fault-injection scenarios in a running Isaac
Sim environment), documented in the open rather than glossed over, and released as
[v0.2.2](https://github.com/arkonahomerobotics/safety-harness/releases/tag/v0.2.2) within the same
development session it was found. That was an internally-found issue, not an external report, but
it's real evidence of how this project actually handles a safety-relevant defect once one surfaces:
seriously, fixed, tested harder than the original bug required, and disclosed openly in the design
doc rather than fixed silently.

## Reporting a vulnerability

Email **kaoru.naganuma@gmail.com** with:

- A description of the issue and, if it's a safety-logic defect, the specific `WorldState`/`Action`
  values (or a minimal reproduction against `tests/fixtures.py`-style fixtures) that trigger it.
- Which check, module, or dependency is affected.
- Your assessment of severity/impact if you have one -- helpful, not required.

Please report privately first rather than opening a public GitHub issue, for the usual reason: this
project has real users building robot integrations against it, and a public issue naming an
unpatched default-deny bypass is itself a hazard notice for anyone running it unpatched.

GitHub's private vulnerability reporting (Security tab → "Report a vulnerability") is not currently
enabled on this repository; email is the working channel until that changes.

## Response time

No SLA -- this is one person's spare-time-adjacent project, not a funded security team, and a
promise this project can't keep would be worse than an honest one. In practice: expect an
acknowledgment within **5 business days**. After that, how fast a fix ships depends on severity and
complexity the same way the NaN-sensor fix did (same day, once found) versus more structural
findings (may take longer, and may ship as a documented known-gap first -- see the README's "Not
yet independently verified" section and the design doc's Scope & Non-Goals for the project's
existing practice of disclosing a gap it hasn't fixed yet rather than staying silent about it).

## Supported versions

Pre-1.0 (`Development Status :: 3 - Alpha`) and single-maintainer: only the **latest published
release on PyPI** gets security fixes. There is no backport policy to older minor versions --
upgrading is the fix. Check [PyPI](https://pypi.org/project/safety-harness/) or the
[releases page](https://github.com/arkonahomerobotics/safety-harness/releases) for the current version.

## Coordinated disclosure

Standard coordinated disclosure, sized to this project's actual capacity rather than a generic
90-day figure imported from a much bigger project:

1. Report privately (above). You'll get an acknowledgment and, once triaged, an honest estimate of
   how long a fix will take.
2. Please hold public disclosure until a fix is released, or until 90 days have passed with no
   substantive response from the maintainer -- whichever comes first. If a fix is more involved
   than that window allows, the maintainer will say so and ask for more time rather than going
   quiet; you're not obligated to grant it, but the ask will be explicit.
3. Once a fix ships, it will be documented in the [design doc's Version History](docs/design.md)
   the same way every other change to this project is -- what was found, how it was confirmed, how
   it was fixed and validated -- and you're welcome to be credited by name if you'd like, or to stay
   anonymous.

## What this policy doesn't cover

This project has **not** been independently assessed against IEC 61508, ISO 13849, or ISO
10218/TS 15066 (see the README's Status section), doesn't yet address the data-protection
implications of tracking and logging real human positions (flagged, unaddressed, in the design
doc), and `config_integrity_verified`'s unkeyed digest mode detects corruption and
uncoordinated edits, not an attacker who can rewrite both a config and its pinned hash (README, same
section) -- an HMAC key closes that gap; using this project without one doesn't. None of that is a
vulnerability this process will "fix" on report, because it's an already-disclosed scope
limitation, not a defect -- but if you find a way an attacker could exploit one of these known gaps
in a way not already described above, that's still worth reporting the same way.
