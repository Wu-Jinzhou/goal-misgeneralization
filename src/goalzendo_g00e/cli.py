"""Explicit wrapper CLI for the additive G00-E numerical bridge."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .bridge import (
    BridgeError,
    authenticate_legacy_and_bridge,
    create_v3_sidecar,
    install_exact_interval_correction,
    install_worker_bridge,
    verify_v3_sidecar,
)

_VALUE_OPTIONS = frozenset(
    {
        "--g00e-sidecar-output",
        "--g00e-sidecar",
        "--g00e-sidecar-sha256",
        "--g00e-manifest-sha256",
    }
)


def _extract_bridge_options(arguments: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    clean: list[str] = []
    observed: dict[str, str] = {}
    index = 0
    while index < len(arguments):
        token = arguments[index]
        option: str | None = None
        value: str | None = None
        if token in _VALUE_OPTIONS:
            option = token
            if index + 1 >= len(arguments):
                raise BridgeError(f"{token} requires one value")
            value = arguments[index + 1]
            index += 2
        else:
            for candidate in _VALUE_OPTIONS:
                prefix = f"{candidate}="
                if token.startswith(prefix):
                    option = candidate
                    value = token[len(prefix) :]
                    break
            if option is None:
                clean.append(token)
            index += 1
        if option is not None:
            if option in observed:
                raise BridgeError(f"duplicate G00-E wrapper option: {option}")
            if value is None or not value:
                raise BridgeError(f"{option} requires a nonempty value")
            observed[option] = value
    return clean, observed


def _require_options(options: dict[str, str], names: Sequence[str], command: str) -> None:
    missing = [name for name in names if name not in options]
    if missing:
        raise BridgeError(f"G00-E {command} requires explicit " + ", ".join(missing))
    allowed = set(names)
    unexpected = sorted(set(options) - allowed)
    if unexpected:
        raise BridgeError(f"G00-E {command} does not accept " + ", ".join(unexpected))


def _print_json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        clean, options = _extract_bridge_options(raw)

        if clean and clean[0] == "worker-preflight":
            _require_options(
                options,
                (
                    "--g00e-sidecar",
                    "--g00e-sidecar-sha256",
                    "--g00e-manifest-sha256",
                ),
                "worker preflight",
            )
            preflight = argparse.ArgumentParser(prog="goalzendo-g00e worker-preflight")
            preflight.add_argument("--repo", type=str, required=True)
            preflight.add_argument("--gate-artifact", type=str, required=True)
            parsed_preflight = preflight.parse_args(clean[1:])
            verified = verify_v3_sidecar(
                sidecar_path=options["--g00e-sidecar"],
                expected_sidecar_sha256=options["--g00e-sidecar-sha256"],
                gate_path=parsed_preflight.gate_artifact,
                repo=parsed_preflight.repo,
                expected_manifest_sha256=options["--g00e-manifest-sha256"],
            )
            verified.pop("authenticated_identity", None)
            _print_json({"g00e_v3_worker_preflight": verified})
            return 0

        import goalzendo.cli as legacy_cli

        parsed = legacy_cli.build_parser().parse_args(clean)
        repo = legacy_cli._repo_root(parsed.repo)
        if parsed.command == "gate":
            _require_options(
                options,
                ("--g00e-sidecar-output", "--g00e-manifest-sha256"),
                "gate",
            )
            identity = authenticate_legacy_and_bridge(
                repo=repo,
                expected_manifest_sha256=options["--g00e-manifest-sha256"],
            )
            install_exact_interval_correction(identity)
            result = legacy_cli.main(clean)
            if result != 0:
                return result
            assessment_path = parsed.assessment_output or parsed.output.with_name(
                f"{parsed.output.stem}-assessment.json"
            )
            sidecar = create_v3_sidecar(
                gate_path=parsed.output,
                assessment_path=assessment_path,
                output_path=options["--g00e-sidecar-output"],
                authenticated_identity=identity,
            )
            _print_json({"g00e_v3_sidecar": sidecar})
            return 0
        if parsed.command == "run":
            _require_options(
                options,
                (
                    "--g00e-sidecar",
                    "--g00e-sidecar-sha256",
                    "--g00e-manifest-sha256",
                ),
                "worker run",
            )
            if parsed.gate_artifact is None:
                raise BridgeError("G00-E worker run requires the legacy --gate-artifact")
            installed = install_worker_bridge(
                sidecar_path=options["--g00e-sidecar"],
                expected_sidecar_sha256=options["--g00e-sidecar-sha256"],
                gate_path=parsed.gate_artifact,
                repo=repo,
                expected_manifest_sha256=options["--g00e-manifest-sha256"],
            )
            _print_json({"g00e_v3_worker_bridge": installed})
            return legacy_cli.main(clean)
        if options:
            raise BridgeError("G00-E wrapper options are valid only for explicit gate or run commands")
        return legacy_cli.main(clean)
    except BridgeError as error:
        print(f"goalzendo-g00e: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
