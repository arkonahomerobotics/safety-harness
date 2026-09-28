"""Generate a CycloneDX 1.5 Software Bill of Materials for the installed ``safety-harness``
distribution -- stdlib only (``importlib.metadata``, ``json``, ``re``, ``datetime``), same
dependency-free-tooling preference as ``pin.py``.

**CycloneDX over SPDX.** Both are OWASP/ISO-recognized formats a Machinery Regulation Annex IV
technical file can point to; CycloneDX was picked here because its dependency-graph model
(``components`` + a ``dependencies`` array of ``ref``/``dependsOn`` edges) maps directly onto what
this script actually has to say -- one small package, one runtime dependency, no dependents -- with
less boilerplate than SPDX's tag-value or JSON forms for the same graph. Nothing here stops a later
SPDX export if a downstream integrator's toolchain requires it specifically.

**What this SBOM says, plainly stated rather than left to be inferred from an empty list:**
``safety-harness`` is *not* fully stdlib-only at the package level, despite ``integrity.py``'s own
docstring describing itself as following "the same dependency-free rule as the rest of this
package" -- that statement is true of ``integrity.py`` itself (``hashlib``/``hmac``/``struct``
only) but not, read literally, of the package as a whole. ``action_schema.py`` does an
unconditional ``import yaml`` to load the YAML action-schema config (``ActionSchemaRegistry.
from_yaml``, the primary entry point in the README's own usage example), and ``pyproject.toml``
correctly declares this as a runtime dependency (``pyyaml>=6.0``). So the honest SBOM for this
package has exactly one runtime third-party component, not zero -- and the test in
``tests/test_sbom.py`` pins that list to what ``pyproject.toml`` currently declares, so a future
dependency silently added to ``[project].dependencies`` fails the test instead of going unnoticed.

Dev/test-only tooling (``pytest``, used to run ``tests/``) is deliberately excluded: it isn't part
of the installed distribution a ``pip install safety-harness`` pulls in, and -- worth noting as its
own small finding -- it isn't declared anywhere in ``pyproject.toml`` either (no
``[project.optional-dependencies]`` group exists yet); it is just what ``.venv`` happens to have
installed. That absence is recorded as a ``metadata.properties`` entry below rather than silently
matching expectations.

**Reproducibility.** This script is a pure function of the installed package's metadata (which is
itself a pure function of ``pyproject.toml`` plus the resolved dependency versions in the
environment it runs in) except for one field: ``metadata.timestamp``. Set ``SOURCE_DATE_EPOCH``
(the standard reproducible-builds env var, unix seconds) to fix that field too, and two runs
against the same installed environment produce byte-identical output. No random ``serialNumber`` is
emitted (CycloneDX makes it optional) specifically so that determinism holds without extra
bookkeeping.

Run (after ``pip install -e .``, the same prerequisite the README's own "Install & test" section
already establishes for everything else here):

    python -m safety_harness.sbom > examples/sbom/results/safety-harness.cyclonedx.json

The checked-in file at that path is exactly this command's output, regenerated deliberately after a
reviewed dependency change -- same convention as ``configs/example_action_schema.yaml.sha256`` for
config pinning.
"""

from __future__ import annotations

import datetime
import importlib.metadata as _metadata
import json
import os
import re
import sys
from typing import Dict, List, Tuple

CYCLONEDX_SPEC_VERSION = "1.5"

# A small, deliberately non-exhaustive whitelist of SPDX license identifiers this script will
# recognize and encode as {"license": {"id": ...}}. Anything else -- including a license string
# importlib.metadata reports that isn't a clean SPDX id -- is encoded as {"license": {"name": ...}}
# instead, which CycloneDX also accepts. This is not a general SPDX-expression parser; it doesn't
# need to be for the one or two components this package actually has.
_KNOWN_SPDX_IDS = {
    "Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC",
    "MPL-2.0", "LGPL-3.0-only", "LGPL-3.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later",
    "PSF-2.0", "Unlicense", "0BSD",
}

_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


class SbomError(RuntimeError):
    """Raised when the environment can't support generating an honest SBOM -- e.g. a declared
    dependency isn't actually installed, so its real version/license can't be reported."""


def _canonicalize(name: str) -> str:
    """PEP 503 distribution-name canonicalization: lowercase, runs of -_. collapsed to a single -."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    match = _REQ_NAME_RE.match(requirement)
    if not match:
        raise SbomError(f"can't parse a distribution name out of requirement {requirement!r}")
    return match.group(1)


def _is_optional_extra(requirement: str) -> bool:
    """True for a requirement gated behind an extra (e.g. 'foo; extra == "dev"') -- not installed
    by a plain `pip install safety-harness`, so out of scope for a runtime SBOM."""
    return "extra ==" in requirement or "extra==" in requirement


def _resolve_dependency_graph(root_name: str) -> Tuple[List[str], Dict[str, _metadata.Distribution], Dict[str, List[str]]]:
    """Breadth-first walk of the *runtime* requirement graph starting at ``root_name``, using
    installed-distribution metadata as the source of truth. Returns (visit-order canonical keys,
    key -> Distribution, key -> [canonical keys it depends on])."""
    order: List[str] = []
    dists: Dict[str, _metadata.Distribution] = {}
    edges: Dict[str, List[str]] = {}
    queue = [root_name]
    while queue:
        name = queue.pop(0)
        key = _canonicalize(name)
        if key in dists:
            continue
        try:
            dist = _metadata.distribution(name)
        except _metadata.PackageNotFoundError as exc:
            raise SbomError(
                f"{name!r} is required but not installed in this environment. Run `pip install -e .` "
                "(see README, 'Install & test') before regenerating the SBOM -- an SBOM can only "
                "report what's actually resolvable, not what pyproject.toml merely declares."
            ) from exc
        dists[key] = dist
        order.append(key)
        deps: List[str] = []
        for req in dist.requires or ():
            if _is_optional_extra(req):
                continue
            dep_name = _requirement_name(req)
            deps.append(_canonicalize(dep_name))
            queue.append(dep_name)
        edges[key] = deps
    return order, dists, edges


def _purl(dist: _metadata.Distribution) -> str:
    name = _canonicalize(dist.metadata["Name"])
    version = dist.metadata["Version"]
    return f"pkg:pypi/{name}@{version}"


def _licenses_for(dist: _metadata.Distribution) -> List[dict]:
    meta = dist.metadata
    expr = meta.get("License-Expression")
    classic = meta.get("License")
    text = expr or (classic if classic and classic != "UNKNOWN" else None)
    if not text:
        return []
    if text in _KNOWN_SPDX_IDS:
        return [{"license": {"id": text}}]
    return [{"license": {"name": text}}]


def _component(dist: _metadata.Distribution, *, is_root: bool) -> dict:
    meta = dist.metadata
    purl = _purl(dist)
    component = {
        "type": "application" if is_root else "library",
        "bom-ref": purl,
        "name": meta["Name"],
        "version": meta["Version"],
        "purl": purl,
    }
    summary = meta.get("Summary")
    if summary:
        component["description"] = summary
    licenses = _licenses_for(dist)
    if licenses:
        component["licenses"] = licenses
    return component


def _timestamp() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        dt = datetime.datetime.fromtimestamp(int(epoch), tz=datetime.timezone.utc)
    else:
        dt = datetime.datetime.now(tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_sbom(*, root: str = "safety-harness") -> dict:
    """Build the CycloneDX document as a plain dict (JSON-serializable). Raises SbomError if the
    root package or any of its declared runtime dependencies isn't actually installed."""
    order, dists, edges = _resolve_dependency_graph(root)
    root_key = _canonicalize(root)
    root_component = _component(dists[root_key], is_root=True)

    components = [_component(dists[key], is_root=False) for key in order if key != root_key]

    dependencies = []
    for key in order:
        dependencies.append({
            "ref": _purl(dists[key]),
            "dependsOn": sorted(_purl(dists[dep_key]) for dep_key in edges[key]),
        })

    dev_note = (
        "Development/test-only tooling (pytest) is required to run tests/ but is not declared "
        "anywhere in pyproject.toml (no [project.optional-dependencies] group exists yet) and is "
        "not part of the installed runtime distribution; deliberately excluded from this SBOM's "
        "components."
    )
    philosophy_note = (
        "safety_harness/integrity.py describes itself as following 'the same dependency-free rule "
        "as the rest of this package'; read literally that overstates the package as a whole -- "
        "action_schema.py imports PyYAML unconditionally at runtime, and pyproject.toml correctly "
        "declares it. This SBOM lists that one real runtime dependency rather than asserting zero."
    )

    return {
        "bomFormat": "CycloneDX",
        "specVersion": CYCLONEDX_SPEC_VERSION,
        "version": 1,
        "metadata": {
            "timestamp": _timestamp(),
            "component": root_component,
            "properties": [
                {"name": "safety-harness:dev-tooling", "value": dev_note},
                {"name": "safety-harness:dependency-philosophy", "value": philosophy_note},
            ],
        },
        "components": components,
        "dependencies": dependencies,
    }


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print("usage: python -m safety_harness.sbom  (no arguments; writes CycloneDX JSON to stdout)", file=sys.stderr)
        return 2
    try:
        bom = generate_sbom()
    except SbomError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(bom, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
