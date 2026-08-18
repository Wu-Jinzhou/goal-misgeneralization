#!/usr/bin/env python3
"""Small, dependency-free helpers for the GoalZendo Runpod entrypoint.

This file deliberately does not import :mod:`goalzendo`.  It can therefore
validate a pod's local GPU layout and write crash-safe runtime metadata before
the project virtual environment has been installed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DEVICE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_FIELD_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_NO_GPU_SPECS = frozenset({"-1", "none", "void"})


class RuntimePlanError(ValueError):
    """Raised when a requested local shard plan is unsafe or impossible."""


@dataclass(frozen=True)
class ShardAssignment:
    """One stable hash shard assigned to one CUDA-visible device."""

    shard_index: int
    num_shards: int
    cuda_visible_devices: str


def _visible_device_tokens(detected_gpus: int, visible_devices: str | None) -> tuple[str, ...]:
    if isinstance(detected_gpus, bool) or detected_gpus < 0:
        raise RuntimePlanError("detected_gpus must be a non-negative integer")

    if visible_devices is None or visible_devices.strip().lower() == "all":
        return tuple(str(index) for index in range(detected_gpus))

    specification = visible_devices.strip()
    if specification.lower() in _NO_GPU_SPECS or not specification:
        return ()

    raw_tokens = specification.split(",")
    if any(not token.strip() for token in raw_tokens):
        raise RuntimePlanError("CUDA_VISIBLE_DEVICES contains an empty device token")
    tokens = tuple(token.strip() for token in raw_tokens)
    if any(_DEVICE_TOKEN.fullmatch(token) is None for token in tokens):
        raise RuntimePlanError("CUDA_VISIBLE_DEVICES contains an unsafe device token")
    if len(set(tokens)) != len(tokens):
        raise RuntimePlanError("CUDA_VISIBLE_DEVICES contains duplicate device tokens")
    if len(tokens) > detected_gpus:
        raise RuntimePlanError(
            "CUDA_VISIBLE_DEVICES names more devices than nvidia-smi reports inside the pod"
        )
    return tokens


def build_local_shard_plan(
    requested_shards: int,
    detected_gpus: int,
    visible_devices: str | None = None,
) -> tuple[ShardAssignment, ...]:
    """Return a deterministic one-shard-per-GPU execution plan.

    Device order follows an inherited ``CUDA_VISIBLE_DEVICES`` exactly.  When
    that variable is unset, CUDA's local ordinal order (``0, 1, ...``) is used.
    Only the first ``requested_shards`` visible devices are selected so a pod
    may intentionally leave spare GPUs idle.
    """

    if isinstance(requested_shards, bool) or requested_shards < 1:
        raise RuntimePlanError("requested_shards must be a positive integer")
    devices = _visible_device_tokens(detected_gpus, visible_devices)
    if requested_shards > len(devices):
        raise RuntimePlanError(
            f"requested {requested_shards} local GPU shards, but only {len(devices)} "
            "CUDA-visible GPUs are available"
        )
    return tuple(
        ShardAssignment(
            shard_index=index,
            num_shards=requested_shards,
            cuda_visible_devices=device,
        )
        for index, device in enumerate(devices[:requested_shards])
    )


def _parse_key_value(raw: str, option: str) -> tuple[str, str]:
    if "=" not in raw:
        raise RuntimePlanError(f"{option} values must have the form KEY=VALUE")
    key, value = raw.split("=", 1)
    if _FIELD_NAME.fullmatch(key) is None:
        raise RuntimePlanError(f"invalid JSON field name {key!r}")
    return key, value


def _strict_integer(raw: str, option: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimePlanError(f"{option} requires an integer value") from exc
    if str(value) != raw and str(value) != raw.lstrip("+"):
        raise RuntimePlanError(f"{option} requires a canonical base-10 integer")
    return value


def build_json_fields(
    strings: Sequence[str], integers: Sequence[str], booleans: Sequence[str]
) -> dict[str, Any]:
    """Build a strict JSON object from typed ``KEY=VALUE`` command arguments."""

    result: dict[str, Any] = {}
    for option, values in (
        ("--string", strings),
        ("--integer", integers),
        ("--boolean", booleans),
    ):
        for raw in values:
            key, value = _parse_key_value(raw, option)
            if key in result:
                raise RuntimePlanError(f"duplicate JSON field {key!r}")
            if option == "--integer":
                result[key] = _strict_integer(value, option)
            elif option == "--boolean":
                normalized = value.lower()
                if normalized not in {"true", "false"}:
                    raise RuntimePlanError("--boolean values must be true or false")
                result[key] = normalized == "true"
            else:
                result[key] = value
    return result


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    """Atomically replace a strict UTF-8 JSON object."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="dry-run a deterministic local GPU shard plan")
    plan.add_argument("--requested-shards", type=int, required=True)
    plan.add_argument("--detected-gpus", type=int, required=True)
    plan.add_argument(
        "--visible-devices",
        default=None,
        help="inherited CUDA_VISIBLE_DEVICES; omit when it is unset",
    )
    plan.add_argument("--format", choices=("json", "tsv"), default="json")

    write = subparsers.add_parser("write-json", help="atomically write typed runtime metadata")
    write.add_argument("--output", type=Path, required=True)
    write.add_argument("--string", action="append", default=[], metavar="KEY=VALUE")
    write.add_argument("--integer", action="append", default=[], metavar="KEY=VALUE")
    write.add_argument("--boolean", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            assignments = build_local_shard_plan(
                args.requested_shards,
                args.detected_gpus,
                args.visible_devices,
            )
            if args.format == "tsv":
                for assignment in assignments:
                    print(
                        f"{assignment.shard_index}\t{assignment.num_shards}\t"
                        f"{assignment.cuda_visible_devices}"
                    )
            else:
                payload = {
                    "schema": "goalzendo.runpod_local_gpu_plan",
                    "schema_version": 1,
                    "requested_shards": args.requested_shards,
                    "detected_gpus": args.detected_gpus,
                    "inherited_cuda_visible_devices": args.visible_devices,
                    "assignments": [asdict(assignment) for assignment in assignments],
                }
                print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
            return 0

        fields = build_json_fields(args.string, args.integer, args.boolean)
        atomic_write_json(args.output, fields)
        return 0
    except RuntimePlanError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
