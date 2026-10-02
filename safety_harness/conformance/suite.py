"""Ties the three stages together into one call a third party makes against their own adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..action_schema import ActionSchemaRegistry
from ..adapters.base import DynamicsAdapter, FallbackController, Logger, PerceptionAdapter
from ..schema import SCHEMA_VERSION, Action
from .contract import run_contract_fuzz
from .mutation import run_mutation_battery
from .report import CheckOutcome, ConformanceReport, Violation
from .structural import validate_trajectory, validate_world_state


@dataclass
class ConformanceSuite:
    """``perception``/``dynamics`` are the third party's own adapter *instances* -- already
    constructed, pointed at their robot or simulator. ``action_schema`` is the
    ``ActionSchemaRegistry`` they intend to actually run (their own YAML, not this project's
    example), since without one there is nothing for the contract-fuzz stage to check action types
    against. ``baseline_action`` is optional but required for the mutation-battery stage: one
    ``Action`` the caller believes is currently safe given their adapter's live state (e.g. a grasp
    of a known-cleared object) -- see ``mutation.py`` for why this can't be synthesized generically.
    ``fallback``/``logger`` default to the harness's own ``FreezeInPlaceFallback``/``InMemoryLogger``;
    override only if the real ones matter to what you're testing, since this suite is about
    perception/dynamics conformance, not fallback or logging behavior.
    """

    perception: PerceptionAdapter
    dynamics: DynamicsAdapter
    action_schema: ActionSchemaRegistry
    baseline_action: Optional[Action] = None
    fallback: Optional[FallbackController] = None
    logger: Optional[Logger] = None
    seed: int = 1
    n_fuzz_iterations: int = 200
    horizon_s: float = 1.0

    def run(self) -> ConformanceReport:
        outcomes = [
            self._run_structural_conformance(),
            self._run_contract_fuzz(),
            self._run_mutation_battery(),
        ]
        return ConformanceReport(outcomes=tuple(outcomes), meta={"schema_version": SCHEMA_VERSION, "seed": self.seed})

    def _run_structural_conformance(self) -> CheckOutcome:
        violations: list = []
        state = None
        try:
            state = self.perception.get_world_state()
            violations.extend(validate_world_state(state))
        except Exception as exc:  # noqa: BLE001
            violations.append(Violation(
                "structural", f"perception.get_world_state() raised {type(exc).__name__}: {exc}", {},
            ))

        if state is not None:
            probe_action = self.baseline_action
            if probe_action is None:
                registered_types = sorted(self.action_schema.effective_config().get("action_types", {}))
                probe_params = {"object_id": state.objects[0].object_id} if state.objects else {}
                probe_action = Action(action_type=registered_types[0] if registered_types else "reach", params=probe_params)
            try:
                trajectory = self.dynamics.predict_trajectory(state, probe_action, self.horizon_s)
                violations.extend(validate_trajectory(trajectory))
            except Exception as exc:  # noqa: BLE001
                violations.append(Violation(
                    "structural", f"dynamics.predict_trajectory() raised {type(exc).__name__}: {exc}",
                    {"probe_action": repr(probe_action)},
                ))

        return CheckOutcome(
            name="structural_conformance", ran=True, passed=not violations, violations=tuple(violations),
        )

    def _run_contract_fuzz(self) -> CheckOutcome:
        violations, histogram = run_contract_fuzz(
            self.perception, self.dynamics, self.action_schema,
            fallback=self.fallback, logger=self.logger, n_iterations=self.n_fuzz_iterations, seed=self.seed,
        )
        return CheckOutcome(
            name="contract_fuzz", ran=True, passed=not violations, violations=violations, detail={"histogram": histogram},
        )

    def _run_mutation_battery(self) -> CheckOutcome:
        if self.baseline_action is None:
            return CheckOutcome(
                name="mutation_battery", ran=False, passed=True,
                skip_reason="no baseline_action supplied -- pass an Action known to currently PERMIT to run this stage",
            )
        result = run_mutation_battery(
            self.perception, self.dynamics, self.action_schema, self.baseline_action,
            fallback=self.fallback, logger=self.logger, horizon_s=self.horizon_s,
        )
        if not result["baseline_permitted"]:
            return CheckOutcome(
                name="mutation_battery", ran=False, passed=True, skip_reason=result["skip_reason"], detail=result["detail"],
            )
        violations = result["violations"]
        return CheckOutcome(
            name="mutation_battery", ran=True, passed=not violations, violations=violations, detail=result["detail"],
        )
