"""The structured output the rest of this package produces -- a third party gets this back instead
of a pile of print statements or a pytest exit code, per the task this package exists to close (see
the package docstring)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Violation:
    """One concrete thing the adapter under test got wrong."""

    category: str  # e.g. "structural", "undocumented_exception", "mutation_not_blocked"
    message: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"category": self.category, "message": self.message, "detail": self.detail}


@dataclass(frozen=True)
class CheckOutcome:
    """One named stage of the suite (structural conformance, contract fuzz, mutation battery)."""

    name: str
    ran: bool  # False if the stage was skipped (e.g. no baseline_action supplied)
    passed: bool  # always True when ran=False -- a skipped stage is not a failure
    violations: tuple = ()
    skip_reason: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ran": self.ran,
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "skip_reason": self.skip_reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ConformanceReport:
    outcomes: tuple = ()
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(o.passed for o in self.outcomes)

    @property
    def violations(self) -> tuple:
        return tuple(v for o in self.outcomes for v in o.violations)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "meta": self.meta,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    def summary(self) -> str:
        lines = [f"Conformance: {'PASS' if self.ok else 'FAIL'}"]
        for o in self.outcomes:
            if not o.ran:
                lines.append(f"  [SKIP] {o.name} -- {o.skip_reason}")
                continue
            status = "PASS" if o.passed else "FAIL"
            lines.append(f"  [{status}] {o.name}" + (f" ({len(o.violations)} violation(s))" if o.violations else ""))
            for v in o.violations:
                lines.append(f"      - {v.category}: {v.message}")
        return "\n".join(lines)
