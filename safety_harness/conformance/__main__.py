"""Command-line entry point: ``safety-harness-conformance`` (installed via ``pyproject.toml``'s
``[project.scripts]``) or ``python -m safety_harness.conformance``.

A third party who doesn't want to write any Python at all points this at their own adapter module
and action schema:

    safety-harness-conformance \\
        --perception mystack.adapters:build_perception_adapter \\
        --dynamics mystack.adapters:build_dynamics_adapter \\
        --schema myconfig/action_schema.yaml \\
        --baseline-action '{"action_type": "grasp", "params": {"object_id": "known_safe_cube"}}'

``--perception``/``--dynamics`` each name a zero-argument factory callable (``module.path:name``)
rather than a class, since real adapters almost always need robot-specific constructor arguments
(a simulator handle, a ROS node, ...) this tool has no way to guess -- the factory is the third
party's own code, doing whatever construction their adapter needs, and returning the instance.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from ..action_schema import ActionSchemaRegistry
from ..integrity import read_digest_file
from ..schema import Action
from .suite import ConformanceSuite


def _load_factory(spec: str):
    if ":" not in spec:
        raise ValueError(f"expected 'module.path:factory_name', got {spec!r}")
    module_path, _, factory_name = spec.partition(":")
    module = importlib.import_module(module_path)
    factory = getattr(module, factory_name)
    return factory()


def _load_baseline_action(args) -> Action:
    if args.baseline_action_file:
        with open(args.baseline_action_file) as f:
            raw = json.load(f)
    elif args.baseline_action:
        raw = json.loads(args.baseline_action)
    else:
        return None
    return Action(action_type=raw["action_type"], params=raw.get("params", {}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety-harness-conformance",
        description="Run the third-party conformance fixture against your own PerceptionAdapter/DynamicsAdapter.",
    )
    parser.add_argument("--perception", required=True, help="module.path:factory_name returning a PerceptionAdapter instance")
    parser.add_argument("--dynamics", required=True, help="module.path:factory_name returning a DynamicsAdapter instance")
    parser.add_argument("--schema", required=True, help="path to your action schema YAML")
    parser.add_argument("--schema-digest", help="path to the schema's .sha256 pin file (omit to load unpinned)")
    parser.add_argument("--baseline-action", help='JSON, e.g. {"action_type": "grasp", "params": {"object_id": "cube_1"}}')
    parser.add_argument("--baseline-action-file", help="path to a JSON file with the same shape as --baseline-action")
    parser.add_argument("--iterations", type=int, default=200, help="contract-fuzz iteration count (default: 200)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", help="also write the full report as JSON to this path")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    perception = _load_factory(args.perception)
    dynamics = _load_factory(args.dynamics)
    expected_digest = read_digest_file(args.schema_digest) if args.schema_digest else None
    action_schema = ActionSchemaRegistry.from_yaml(args.schema, expected_digest=expected_digest)
    baseline_action = _load_baseline_action(args)

    suite = ConformanceSuite(
        perception=perception, dynamics=dynamics, action_schema=action_schema,
        baseline_action=baseline_action, seed=args.seed, n_fuzz_iterations=args.iterations,
    )
    report = suite.run()

    print(report.summary())
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nFull report written to {args.json}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
