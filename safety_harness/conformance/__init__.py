"""The real, installable third-party conformance fixture (Development Roadmap stage 5).

Where ``tests/test_blackbox.py`` and ``tests/test_fuzz.py`` prove the engine and preconditions are
adapter-interface-generic by feeding them hand-built or randomly generated ``WorldState``/
``PredictedTrajectory`` objects directly, nothing in ``tests/`` is packaged into the wheel (see
``pyproject.toml``'s ``[tool.setuptools.packages.find]``), and both files assume the reader already
knows this repository's fixture conventions. This package is the third-party-facing version: a
robot team that has implemented ``PerceptionAdapter``/``DynamicsAdapter`` for their own stack
installs ``safety-harness`` and runs ``ConformanceSuite`` (or the ``safety-harness-conformance`` CLI)
against their own adapter *instances* to get a structured, serializable pass/fail report, without
reading a line of this repository's test code.

It checks three distinct things, each for a different reason:

- **Structural conformance** (``structural.py``): does the adapter's own output actually satisfy the
  ``schema.py`` contract -- right types, finite numbers, confidences in ``[0, 1]``? An adapter that
  returns something structurally wrong can defeat every downstream check without ever producing a
  Python exception (see the design doc's "NaN-Sensor Stress Test").
- **Contract fuzzing** (``contract.py``): the same adapter-interface-generic properties
  ``tests/test_blackbox.py`` checks (never an undocumented exception, never a hang, an unrecognized
  action type never permits, ``PERMIT`` returns the original action, ``BLOCK`` always returns some
  action) -- run against the adapter's *real* ``get_world_state()``/``predict_trajectory()``, not a
  synthetic stand-in.
- **Mutation conformance** (``mutation.py``): the same curated single-field-mutation battery
  ``tests/test_fuzz.py`` runs against a hand-built golden fixture, run instead against one real
  snapshot captured from the adapter -- so the fixture doesn't need to invent a plausible "safe"
  ``WorldState`` for a robot it's never seen.

See ``docs/design.md``'s Development Roadmap and "Contributing an adapter" in the README for how
this fits into the rest of the project.
"""

from __future__ import annotations

from .report import CheckOutcome, ConformanceReport, Violation
from .suite import ConformanceSuite

__all__ = [
    "CheckOutcome",
    "ConformanceReport",
    "ConformanceSuite",
    "Violation",
]
