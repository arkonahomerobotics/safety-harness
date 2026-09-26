"""Minimal, framework-free FallbackController and Logger -- usable as-is for a first integration
(design doc: "Getting Started", step 5: "freeze in place is enough if your platform is fixed-base
or currently in a stable stance"), and as fixtures for the engine's own tests.
"""

from __future__ import annotations

from ..schema import Action, Decision, WorldState
from .base import FallbackController, Logger


class FreezeInPlaceFallback(FallbackController):
    """Commands zero motion at the current joint state. Only correct for a fixed-base platform, or
    a legged one already confirmed in a statically stable stance -- see the design doc's "Fallback
    Control and Recovery" for why this is not safe for a humanoid mid-stride."""

    def execute(self, state: WorldState) -> Action:
        joint_positions = state.robot.joint_positions if state.robot else ()
        return Action(action_type="freeze", params={"hold_joint_positions": joint_positions})


class InMemoryLogger(Logger):
    """Records every decision in order, for tests and for a first integration before wiring to a
    real telemetry sink (design doc: "Logging, Escalation and the Correction Loop")."""

    def __init__(self):
        self.records: list = []

    def record(self, decision: Decision, state: WorldState) -> None:
        self.records.append((decision, state))
