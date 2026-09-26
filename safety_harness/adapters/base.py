"""Adapter interfaces a robot team implements.

The engine (engine.py) never imports a robot-specific module -- only these four abstract
interfaces. See the design doc's "Reference Module Design" and "Stack-by-Stack Integration"
sections for what each wraps on a given robot stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import Action, Decision, PredictedTrajectory, WorldState


class PerceptionAdapter(ABC):
    @abstractmethod
    def get_world_state(self) -> WorldState:
        """Read whatever perception this robot already runs and return it in the fixed schema."""


class DynamicsAdapter(ABC):
    @abstractmethod
    def predict_trajectory(self, state: WorldState, action: Action, horizon_s: float) -> PredictedTrajectory:
        """Forward-simulate the proposed action using this robot's own kinematics/physics model."""


class FallbackController(ABC):
    @abstractmethod
    def execute(self, state: WorldState) -> Action:
        """Return this robot's safe fallback action given the current state (freeze, retract, ...)."""


class Logger(ABC):
    @abstractmethod
    def record(self, decision: Decision, state: WorldState) -> None:
        """Write the decision to whatever telemetry sink this robot already uses."""
