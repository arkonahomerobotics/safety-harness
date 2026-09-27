"""Print the config digest to pin an action schema to (config_integrity_verified).

    python -m safety_harness.pin configs/example_action_schema.yaml > configs/example_action_schema.yaml.sha256

With ``SAFETY_HARNESS_CONFIG_KEY`` set in the environment, prints the HMAC-SHA256 digest under
that key instead -- pass the same key to ActionSchemaRegistry(hmac_key=...) when verifying. Run it
after a *reviewed* change only: re-pinning is how a change becomes the new validated baseline.
"""

from __future__ import annotations

import sys

from .action_schema import ActionSchemaRegistry
from .integrity import CONFIG_KEY_ENV, key_from_env


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m safety_harness.pin path/to/action_schema.yaml", file=sys.stderr)
        return 2
    registry = ActionSchemaRegistry.from_yaml(args[0], hmac_key=key_from_env(CONFIG_KEY_ENV))
    print(registry.config_digest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
