"""Command-line inspection and verification for interactive GoalZendo."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ._json import dump_json
from .dialogue import dialogue_as_obj, render_dialogue
from .engine_audit import parse_engine_audit_report, verify_engineering_audit
from .environment import play_reference_episode
from .generation import parse_episode_bank
from .production_banks import (
    build_production_bank_plan,
    build_production_bank_qa_report,
    serialize_production_bank_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goalzendo-zendo",
        description="Inspect and verify the hidden-law interactive GoalZendo engine.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser(
        "bank-plan",
        help="print the non-authorizing prospective production-bank request plan",
    )
    plan.add_argument(
        "--with-qa",
        action="store_true",
        help="wrap the plan with its derived structural QA report",
    )

    example = commands.add_parser(
        "example",
        help="replay an exact reference trajectory from a stored episode bank",
    )
    example.add_argument("episode_bank", type=Path)
    example.add_argument("--episode-index", type=int, default=0)
    example.add_argument(
        "--include-hidden",
        action="store_true",
        help="include evaluator-only target/shadow rule identities",
    )

    audit = commands.add_parser(
        "verify-engine-audit",
        help="regenerate and verify a stored non-authorizing engineering audit",
    )
    audit.add_argument("report", type=Path)
    audit.add_argument("--repo", type=Path, default=Path.cwd())
    return parser


def _read_canonical_line(path: Path) -> str:
    payload = path.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError(f"{path} must end in exactly one newline")
    try:
        return payload[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} must be canonical ASCII JSON") from exc


def _bank_plan(*, with_qa: bool) -> str:
    plan = build_production_bank_plan()
    if not with_qa:
        return serialize_production_bank_plan(plan)
    report = build_production_bank_qa_report(plan)
    return dump_json({"plan": plan.as_obj(), "qa": report.as_obj()})


def _example(path: Path, *, episode_index: int, include_hidden: bool) -> str:
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError("episode index must be a non-negative integer")
    bank = parse_episode_bank(_read_canonical_line(path))
    if episode_index >= len(bank.episodes):
        raise ValueError(
            f"episode index {episode_index} lies outside bank size {len(bank.episodes)}"
        )
    episode = bank.episodes[episode_index]
    transcript = play_reference_episode(episode)
    result: dict[str, object] = {
        "scope": "deterministic_reference_example_not_model_result",
        "bank_id": bank.spec.bank_id,
        "bank_digest": bank.digest,
        "episode_index": episode_index,
        "episode_id": episode.episode_id,
        "episode_digest": episode.digest,
        "transcript_digest": transcript.digest,
        "dialogue": dialogue_as_obj(render_dialogue(episode, transcript)),
    }
    if include_hidden:
        result["evaluator_only"] = {
            "target_rule_id": episode.target.rule_id,
            "target_rule": episode.target.rule.as_obj(),
            "target_truth_digest": episode.target.truth_digest,
            "shadow_rule_id": episode.shadow.rule_id,
            "shadow_rule": episode.shadow.rule.as_obj(),
            "shadow_truth_digest": episode.shadow.truth_digest,
        }
    return dump_json(result)


def _verify_audit(path: Path, *, repo: Path) -> str:
    report = parse_engine_audit_report(_read_canonical_line(path))
    verified = verify_engineering_audit(report, repo.resolve())
    return dump_json(
        {
            "verified": True,
            "report_digest": verified.digest,
            "source_fingerprint": verified.source_fingerprint,
            "all_checks_passed": verified.all_checks_passed,
            "weight_updates_authorized": verified.weight_updates_authorized,
            "authorization_reasons": list(verified.authorization_reasons),
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "bank-plan":
            output = _bank_plan(with_qa=args.with_qa)
        elif args.command == "example":
            output = _example(
                args.episode_bank,
                episode_index=args.episode_index,
                include_hidden=args.include_hidden,
            )
        elif args.command == "verify-engine-audit":
            output = _verify_audit(args.report, repo=args.repo)
        else:  # pragma: no cover - argparse enforces the registered commands
            raise AssertionError(f"unknown command: {args.command!r}")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"goalzendo-zendo: {exc}\n")
    sys.stdout.write(output + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
