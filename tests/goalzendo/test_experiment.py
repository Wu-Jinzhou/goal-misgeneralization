from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

from goalzendo.artifacts import read_jsonl
from goalzendo.config import DEFAULT_CONFIG, deep_merge
from goalzendo.experiment import (
    ExperimentError,
    GoalZendoExperiment,
    materialize_banks,
    render_experiment,
    render_prompt_view,
    tokenization_metadata,
)
from goalzendo.interventions import audit_intervention, is_one_atom_law_boundary
from goalzendo.runner import build_plan, run_one


def _config(**overrides: object) -> dict[str, object]:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "experiment": {"id": "gztest", "name": "tiny_backend", "status": "exploratory"},
            "run": {
                "seeds": [17],
                "device": "cpu",
                "save_checkpoints": True,
                "checkpoint_steps": [1, 2],
                "resume": True,
            },
            "data": {
                "n_train": 8,
                "n_validation": 8,
                "n_eval_per_cell": 1,
                "feature_count": 9,
                "law_features": [0],
                "sage_features": [1],
                "distractor_features": [2, 3, 4, 5, 6, 7, 8],
                "rule_family": "literal",
                "sage_rule_family": "literal",
                "q_p": 0.75,
                "q_q": 0.75,
                "training_view": "full",
                "renderer": "natural",
                "train_renderers": ["natural_1"],
                "heldout_renderers": ["natural_2"],
            },
            "model": {
                "name": "local/tiny",
                "revision": "test-revision",
                "dtype": "float32",
                "chat_template": False,
                "action_labels": ["A", "B"],
            },
            "update": {"method": "full"},
            "train": {
                "algorithm": "sft",
                "steps": 2,
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.02,
                "weight_decay": 0.0,
                "warmup_ratio": 0.0,
                "grad_clip": 1.0,
                "eval_steps": [0, 1, 2],
                "max_sequence_length": 4096,
                "gradient_checkpointing": False,
                "kl_coefficient": 0.0,
            },
            "evaluation": {
                "prompt_views": ["full", "no_herald", "no_sage", "law_only"],
                "causal_prompt_views": ["full"],
                "causal_per_cell": 1,
                "final_causal_per_cell": 1,
                "final_eval_per_cell": 1,
                "batch_size": 2,
                "save_predictions": True,
            },
        },
    )
    return deep_merge(config, overrides)


class ByteTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259
    name_or_path = "local/byte-tokenizer"
    padding_side = "left"
    chat_template = None
    init_kwargs: ClassVar[dict[str, str]] = {"_commit_hash": "tokenizer-test-commit"}

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        values = [value + 3 for value in text.encode("utf-8")]
        return [self.bos_token_id, *values] if add_special_tokens else values


class TinyCausalLM(nn.Module):
    def __init__(self, vocabulary_size: int = 259, width: int = 12) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width)
        self.output = nn.Linear(width, vocabulary_size)
        self.config = SimpleNamespace(_commit_hash="model-test-commit", use_cache=True)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        logits_to_keep: int = 0,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        logits = self.output(self.embedding(input_ids))
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def _loader(
    model_config: object,
    update_config: object,
) -> tuple[TinyCausalLM, ByteTokenizer]:
    del model_config, update_config
    return TinyCausalLM(), ByteTokenizer()


def _repo(tmp_path: Path) -> Path:
    source = tmp_path / "repo" / "src" / "goalzendo"
    source.mkdir(parents=True)
    (source / "experiment_identity.py").write_text("VERSION = 1\n", encoding="utf-8")
    return tmp_path / "repo"


def test_symbolic_banks_are_deterministic_factorial_and_causally_capped() -> None:
    config = _config()
    seeds = build_plan(config)[0].seeds
    first = materialize_banks(config, seeds)
    second = materialize_banks(config, seeds)
    assert first.metadata == second.metadata
    assert len(first.train.decisions) == 8
    assert len(first.diagnostic_factorial.decisions) == 8
    assert len(first.final_factorial.decisions) == 8
    assert len(first.diagnostic_causal_base) == 8
    assert all(len(values) == 8 for values in first.diagnostic_interventions.values())
    assert first.metadata["banks"]["diagnostic_factorial"]["candidate_counts"] == {
        "AAA": 1,
        "AAB": 1,
        "ABA": 1,
        "ABB": 1,
        "BAA": 1,
        "BAB": 1,
        "BBA": 1,
        "BBB": 1,
    }
    for bank_name in ("diagnostic", "final"):
        intervention_metadata = first.metadata["interventions"][bank_name]
        edit_audit = intervention_metadata["edit_audit"]
        assert edit_audit["selection_policy"] == (
            "deterministic sample/koan-keyed minimum-Hamming v1"
        )
        by_target = edit_audit["by_target"]
        assert set(by_target) == {"distractor", "herald", "law", "sage"}
        base_count = intervention_metadata["base_count"]
        assert by_target["herald"]["feature_index_counts"] == {}
        assert by_target["herald"]["hamming_distance_counts"] == {
            "0": 2 * base_count
        }
        assert by_target["herald"]["changed_channel_counts"] == {"P": base_count}
        for target, eligible in (
            ("law", {"0"}),
            ("sage", {"1"}),
            ("distractor", {str(index) for index in range(2, 9)}),
        ):
            target_audit = by_target[target]
            assert set(target_audit["feature_index_counts"]) <= eligible
            assert sum(target_audit["feature_index_counts"].values()) == 2 * base_count
            assert target_audit["hamming_distance_counts"] == {"1": 2 * base_count}


def test_absent_per_tuple_budget_preserves_legacy_concentration_exactly() -> None:
    config = _config(
        data={
            "n_train": 32,
            "q_p": 0.75,
            "q_q": 0.75,
            "feature_count": 11,
            "distractor_features": list(range(2, 11)),
            "conflict_diversity": "concentrated",
        }
    )
    banks = materialize_banks(config, build_plan(config)[0].seeds)
    assert banks.metadata["effective_training"] == {
        "conflict_diversity": "concentrated",
        "count": 32,
        "ordered_source_digest": (
            "70803558e880f1ec488055f6edbb99d6c551bd14c71c2854a89c88aa9c282d1c"
        ),
        "unique_total": 28,
        "unique_conflict": 10,
    }


def test_per_tuple_concentration_realizes_and_records_exact_allocations() -> None:
    config = _config(
        data={
            "n_train": 64,
            "q_p": 0.75,
            "q_q": 0.75,
            "feature_count": 11,
            "distractor_features": list(range(2, 11)),
            "conflict_diversity": "concentrated",
            "concentrated_unique_conflicts_per_tuple": 2,
        }
    )
    banks = materialize_banks(config, build_plan(config)[0].seeds)
    expected = {label: 2 for label in ("AAB", "ABA", "ABB", "BAA", "BAB", "BBA")}
    effective_metadata = banks.metadata["effective_training"]
    assert effective_metadata["requested_unique_conflicts_per_tuple"] == 2
    assert effective_metadata["unique_conflict_by_candidate_tuple"] == expected
    assert effective_metadata["unique_conflict"] == sum(expected.values())

    source_counts = Counter(
        "".join(choice.label for choice in item.candidate_tuple)
        for item in banks.train.decisions
    )
    effective_counts = Counter(
        "".join(choice.label for choice in item.candidate_tuple)
        for item in banks.effective_train_decisions
    )
    assert effective_counts == source_counts
    for source, effective in zip(
        banks.train.decisions,
        banks.effective_train_decisions,
        strict=True,
    ):
        assert effective.candidate_tuple == source.candidate_tuple
        if source.choice_p == source.choice_y and source.choice_q == source.choice_y:
            assert effective is source


def test_per_tuple_concentration_fails_if_any_conflict_group_is_too_small() -> None:
    config = _config(
        data={
            "conflict_diversity": "concentrated",
            "concentrated_unique_conflicts_per_tuple": 2,
        }
    )
    with pytest.raises(ExperimentError, match=r"conflict tuple [AB]{3} has only 1 .*requested 2"):
        materialize_banks(config, build_plan(config)[0].seeds)


def test_all_materialized_semantic_banks_are_disjoint_except_registered_relations() -> None:
    config = _config()
    banks = materialize_banks(config, build_plan(config)[0].seeds)

    base_banks = {
        "train": banks.train.decisions,
        "validation": banks.validation.decisions,
        "diagnostic": banks.diagnostic_factorial.decisions,
        "final": banks.final_factorial.decisions,
        "diagnostic_causal": banks.diagnostic_causal_base,
        "final_causal": banks.final_causal_base,
    }
    semantic = {
        name: {
            koan.scene.semantic_digest
            for decision in decisions
            for koan in decision.koans
        }
        for name, decisions in base_banks.items()
    }
    for index, left in enumerate(sorted(semantic)):
        for right in sorted(semantic)[index + 1 :]:
            assert semantic[left].isdisjoint(semantic[right])

    base_support = set().union(*semantic.values())
    intervention_support: dict[str, set[str]] = {}
    for bank_name, interventions in (
        ("diagnostic", banks.diagnostic_interventions),
        ("final", banks.final_interventions),
    ):
        for target in ("sage", "law", "distractor"):
            key = f"{bank_name}_{target}"
            intervention_support[key] = {
                koan.scene.semantic_digest
                for decision in interventions[target]
                for koan in decision.koans
            }
            assert intervention_support[key].isdisjoint(base_support)
    names = sorted(intervention_support)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert intervention_support[left].isdisjoint(intervention_support[right])
    assert all(
        value == 0
        for value in banks.metadata["semantic_scene_audit"]["cross_bank_overlap"].values()
    )
    assert banks.metadata["semantic_scene_audit"]["intervention_cross_overlap"] == 0
    assert banks.metadata["semantic_scene_audit"]["intervention_base_overlap"] == 0


def test_majority_causal_panels_are_conditioned_on_one_atom_boundaries() -> None:
    config = _config(
        data={
            "rule_family": "majority",
            "law_features": [0, 1, 2],
            "sage_features": [3],
            "distractor_features": [4, 5, 6, 7, 8],
        }
    )
    banks = materialize_banks(config, build_plan(config)[0].seeds)
    for bases, interventions in (
        (banks.diagnostic_causal_base, banks.diagnostic_interventions),
        (banks.final_causal_base, banks.final_interventions),
    ):
        assert all(is_one_atom_law_boundary(decision) for decision in bases)
        for base, changed in zip(bases, interventions["law"], strict=True):
            audit = audit_intervention(base, changed)
            assert tuple(len(indices) for indices in audit.changed_feature_indices) == (1, 1)


def test_counterbalance_is_exact_by_side_cell_and_registered_mirror() -> None:
    config = _config(
        data={
            "feature_count": 11,
            "distractor_features": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "train_renderers": ["natural_1", "natural_2"],
            "heldout_renderers": ["natural_5", "natural_6"],
            "counterbalance": True,
            "training_view": "no_signal",
            "n_eval_per_cell": 2,
        },
        evaluation={
            "causal_per_cell": 2,
            "final_causal_per_cell": 2,
            "final_eval_per_cell": 2,
            "mirror_pairs": True,
        },
    )
    seeds = build_plan(config)[0].seeds
    banks = materialize_banks(config, seeds)
    rendered = render_experiment(config, banks, ByteTokenizer(), int(seeds["rendering"]))

    assert rendered.metadata["training_renderer_counts"] == {
        "natural_1": 4,
        "natural_2": 4,
    }
    assert rendered.metadata["training_renderer_by_rewarded_side"] == {
        "A": {"natural_1": 2, "natural_2": 2},
        "B": {"natural_1": 2, "natural_2": 2},
    }
    for counts in rendered.metadata["training_renderer_by_truth_cell"].values():
        values = [counts.get(renderer, 0) for renderer in ("natural_1", "natural_2")]
        assert max(values) - min(values) <= 1
    assert rendered.metadata["validation_renderer_counts"] == {
        "natural_5": 4,
        "natural_6": 4,
    }
    assert rendered.metadata["validation_renderer_by_rewarded_side"] == {
        "A": {"natural_5": 2, "natural_6": 2},
        "B": {"natural_5": 2, "natural_6": 2},
    }
    for counts in rendered.metadata["validation_renderer_by_truth_cell"].values():
        values = [counts.get(renderer, 0) for renderer in ("natural_5", "natural_6")]
        assert max(values) - min(values) <= 1
    for key in ("diagnostic_renderer_by_truth_cell", "final_renderer_by_truth_cell"):
        assert all(
            counts == {"natural_5": 1, "natural_6": 1}
            for counts in rendered.metadata[key].values()
        )

    renderer_by_sample = {
        example["sample_id"]: example["renderer_id"]
        for example in rendered.diagnostic_factorial_examples
    }
    pairs: dict[str, dict[str, str]] = {}
    for decision in banks.diagnostic_factorial.decisions:
        assert decision.mirror_pair_id is not None and decision.mirror_role is not None
        pairs.setdefault(decision.mirror_pair_id, {})[decision.mirror_role] = decision.sample_id
    assert all(
        renderer_by_sample[roles["base"]] == renderer_by_sample[roles["mirror"]]
        for roles in pairs.values()
    )


def test_counterbalance_handles_repeated_training_prototypes_by_occurrence() -> None:
    config = _config(
        data={
            "train_renderers": ["natural_1", "natural_2"],
            "heldout_renderers": ["natural_5"],
            "counterbalance": True,
        }
    )
    seeds = build_plan(config)[0].seeds
    banks = materialize_banks(config, seeds)
    # Deliberately repeat one A and one B example; occurrence-level allocation
    # must still use both renderers exactly for each rewarded side.
    left = next(item for item in banks.train.decisions if int(item.choice_y) == 0)
    right = next(item for item in banks.train.decisions if int(item.choice_y) == 1)
    repeated = (left, left, right, right)
    custom = replace(banks, effective_train_decisions=repeated)
    rendered = render_experiment(config, custom, ByteTokenizer(), int(seeds["rendering"]))
    assert rendered.metadata["training_renderer_by_rewarded_side"] == {
        "A": {"natural_1": 1, "natural_2": 1},
        "B": {"natural_1": 1, "natural_2": 1},
    }


def test_counterbalance_fails_closed_on_infeasible_exact_quotas() -> None:
    config = _config(
        data={
            "heldout_renderers": ["natural_4", "natural_5"],
            "counterbalance": True,
            "n_eval_per_cell": 2,
        },
        evaluation={
            "mirror_pairs": True,
            "causal_per_cell": 1,
            "final_causal_per_cell": 1,
            "final_eval_per_cell": 2,
        },
    )
    with pytest.raises(ExperimentError, match="causal mirror/renderer balance"):
        materialize_banks(config, build_plan(config)[0].seeds)


def test_prompt_views_remove_exactly_the_intended_channels() -> None:
    config = _config()
    banks = materialize_banks(config, build_plan(config)[0].seeds)
    decision = banks.diagnostic_factorial.decisions[0]
    names = banks.diagnostic_factorial.feature_names

    full = render_prompt_view(
        decision, names, renderer_id="natural_1", prompt_view="full"
    )
    assert "Official Law:" in full and "Sage rule:" in full and "Herald stamp:" in full

    no_herald = render_prompt_view(
        decision, names, renderer_id="natural_1", prompt_view="no_herald"
    )
    assert "Herald stamp:" not in no_herald
    assert "Official Law:" in no_herald and "Sage rule:" in no_herald

    law_only = render_prompt_view(
        decision, names, renderer_id="natural_1", prompt_view="law_only"
    )
    assert "Official Law:" in law_only
    assert "Sage rule:" not in law_only and "Herald stamp:" not in law_only
    assert "01=" in law_only and "02=" not in law_only
    assert "03=" not in law_only

    sage_only = render_prompt_view(
        decision, names, renderer_id="natural_1", prompt_view="sage_only"
    )
    assert "Official Law:" not in sage_only and "Sage rule:" in sage_only
    assert "Herald stamp:" not in sage_only
    assert "01=" not in sage_only and "02=" in sage_only
    assert "03=" not in sage_only

    herald_only = render_prompt_view(
        decision, names, renderer_id="natural_1", prompt_view="herald_only"
    )
    assert "Official Law:" not in herald_only and "Sage rule:" not in herald_only
    assert "Herald stamp:" in herald_only and "Feature states:" not in herald_only

    no_signal = render_prompt_view(
        decision, names, renderer_id="natural_1", prompt_view="no_signal"
    )
    assert "Official Law:" not in no_signal and "Sage rule:" not in no_signal
    assert "Herald stamp:" not in no_signal and "Feature states:" not in no_signal

    surface = render_prompt_view(
        decision, names, renderer_id="natural_1", prompt_view="surface_only"
    )
    assert "Official Law:" not in surface and "Sage rule:" not in surface
    assert "Herald stamp:" not in surface
    assert "01=" not in surface and "02=" not in surface
    assert "03=" in surface and "09=" in surface


@pytest.mark.parametrize("renderer_id", ["natural_2", "nonce_2"])
def test_law_elicitation_views_are_full_context_and_layout_matched(
    renderer_id: str,
) -> None:
    config = _config(
        data={
            "feature_count": 17,
            "law_features": [0],
            "sage_features": [1],
            "distractor_features": list(range(2, 17)),
        }
    )
    banks = materialize_banks(config, build_plan(config)[0].seeds)
    decision = banks.diagnostic_factorial.decisions[0]
    names = banks.diagnostic_factorial.feature_names

    full = render_prompt_view(
        decision,
        names,
        renderer_id=renderer_id,
        prompt_view="full",
    )
    audit_full = render_prompt_view(
        decision,
        names,
        renderer_id=renderer_id,
        prompt_view="audit_law_full",
    )
    matched = render_prompt_view(
        decision,
        names,
        renderer_id=renderer_id,
        prompt_view="audit_law_matched",
    )

    # The full-context audit changes exactly the terminal instruction.  In
    # particular, candidate-rule information and renderer layout are matched.
    assert full.rsplit("\n", 1)[0] == audit_full.rsplit("\n", 1)[0]
    assert full.rsplit("\n", 1)[1] != audit_full.rsplit("\n", 1)[1]
    assert "LAW" in audit_full.rsplit("\n", 1)[1].upper()
    assert len(full.splitlines()) == len(audit_full.splitlines()) == len(matched.splitlines())

    for index in range(1, 18):
        assert f"{index:02d}=" in audit_full
        assert f"{index:02d}=" in matched
    assert "UNINFORMATIVE" not in audit_full
    assert matched.count("UNINFORMATIVE") == 3
    assert render_prompt_view(
        decision,
        names,
        renderer_id=renderer_id,
        prompt_view="audit_law_matched",
    ) == matched

    if renderer_id.startswith("natural"):
        assert "Official Law:" in matched
        assert "Sage rule: [UNINFORMATIVE PLACEHOLDER]" in matched
        assert matched.count("Herald stamp: [UNINFORMATIVE PLACEHOLDER]") == 2
        assert "SUN" not in matched and "MOON" not in matched
    else:
        assert "LAW:" in matched
        assert "SAGE-RULE: [UNINFORMATIVE]" in matched
        assert matched.count("HERALD: [UNINFORMATIVE]") == 2
        assert "SOL" not in matched and "LUN" not in matched


def test_law_elicitation_views_are_bound_into_prompt_and_token_digests() -> None:
    config = _config(
        evaluation={
            "prompt_views": ["full", "audit_law_full", "audit_law_matched", "law_only"],
        }
    )
    seeds = build_plan(config)[0].seeds
    banks = materialize_banks(config, seeds)
    tokenizer = ByteTokenizer()
    first = render_experiment(config, banks, tokenizer, int(seeds["rendering"]))
    second = render_experiment(config, banks, tokenizer, int(seeds["rendering"]))

    assert first.metadata["diagnostic_factorial_prompt_digest"] == second.metadata[
        "diagnostic_factorial_prompt_digest"
    ]
    assert first.metadata["evaluation_prompt_views"] == [
        "full",
        "audit_law_full",
        "audit_law_matched",
        "law_only",
    ]
    assert "not proof" in first.metadata["law_elicitation_audits"]["interpretation"]
    assert set(first.diagnostic_factorial_examples[0]["prompt_views"]) == {
        "full",
        "audit_law_full",
        "audit_law_matched",
        "law_only",
    }

    token_metadata = tokenization_metadata(
        first,
        tokenizer,
        ("A", "B"),
        add_special_tokens=True,
        max_sequence_length=4096,
    )
    assert "diagnostic_factorial/audit_law_full" in token_metadata[
        "lengths_by_bank_and_view"
    ]
    assert "diagnostic_factorial/audit_law_matched" in token_metadata[
        "lengths_by_bank_and_view"
    ]


def test_tokenization_audit_has_quantiles_cells_and_fails_before_truncation() -> None:
    config = _config()
    seeds = build_plan(config)[0].seeds
    banks = materialize_banks(config, seeds)
    tokenizer = ByteTokenizer()
    rendered = render_experiment(config, banks, tokenizer, int(seeds["rendering"]))
    metadata = tokenization_metadata(
        rendered,
        tokenizer,
        ("A", "B"),
        add_special_tokens=True,
        max_sequence_length=4096,
    )
    assert metadata["length_quantiles"]["max"] == metadata[
        "maximum_prompt_plus_action_tokens"
    ]
    assert "Y=A|P=A|Q=A" in metadata["lengths_by_truth_cell"]
    assert "final_factorial/full" in metadata["lengths_by_bank_and_view"]
    with pytest.raises(ExperimentError, match="refusing silent truncation"):
        tokenization_metadata(
            rendered,
            tokenizer,
            ("A", "B"),
            add_special_tokens=True,
            max_sequence_length=10,
        )


def test_backend_runs_without_network_and_emits_canonical_rows_and_checkpoints(
    tmp_path: Path,
) -> None:
    config = _config()
    spec = build_plan(config)[0]
    outcome = run_one(
        spec,
        repo=_repo(tmp_path),
        output_root=tmp_path / "artifacts",
        backend=GoalZendoExperiment(model_loader=_loader),
    )
    assert outcome.state == "complete"
    path = Path(outcome.path)
    metrics = read_jsonl(path / "metrics.jsonl")
    kinds = {row["kind"] for row in metrics}
    assert {
        "behavioral_agreement",
        "factorial",
        "intervention",
        "checkpoint_summary",
        "optimization",
    } <= kinds
    wide = [row for row in metrics if row["kind"] == "checkpoint_summary"]
    assert wide and {"rho_y", "rho_p", "rho_q", "causal_y", "causal_p", "causal_q"} <= set(
        wide[0]
    )
    factorial = [row for row in metrics if row["kind"] == "factorial"]
    assert factorial and {"choice_y", "choice_p", "choice_q", "action_b_rate"} <= set(
        factorial[0]
    )
    optimization = [row for row in metrics if row["kind"] == "optimization"]
    assert optimization
    assert all(
        row["sampled_b_rate"] is None
        and row["both_actions_sampled_fraction"] is None
        and row["all_zero_loo_advantages_fraction"] is None
        for row in optimization
    )
    assert read_jsonl(path / "predictions.jsonl")
    checkpoint_directory = path / "checkpoints"
    retired = json.loads((checkpoint_directory / "retired.json").read_text(encoding="utf-8"))
    assert retired["step"] == 2
    assert retired["artifact_kind"] == "retired_resumable_checkpoint"
    assert not (checkpoint_directory / "latest.json").exists()
    assert not tuple(checkpoint_directory.glob("resume-step-*.pt"))
    for kind in ("dataset", "model", "tokenizer"):
        assert (path / "manifests" / f"{kind}.json").is_file()
    tokenizer_manifest = json.loads(
        (path / "manifests" / "tokenizer.json").read_text(encoding="utf-8")
    )
    assert tokenizer_manifest["metadata"]["experiment_tokenization"]["length_quantiles"]


def test_backend_keeps_one_resumable_checkpoint_and_selected_weights_only_snapshots(
    tmp_path: Path,
) -> None:
    config = _config(run={"snapshot_steps": [1, 2]})
    spec = build_plan(config)[0]
    outcome = run_one(
        spec,
        repo=_repo(tmp_path),
        output_root=tmp_path / "artifacts",
        backend=GoalZendoExperiment(model_loader=_loader),
    )
    path = Path(outcome.path)
    checkpoint_directory = path / "checkpoints"
    resume_files = sorted(checkpoint_directory.glob("resume-step-*.pt"))
    assert not resume_files
    assert not tuple(checkpoint_directory.glob("step-*.pt"))

    assert not (checkpoint_directory / "latest.json").exists()
    retired = json.loads((checkpoint_directory / "retired.json").read_text(encoding="utf-8"))
    assert retired["artifact_kind"] == "retired_resumable_checkpoint"
    assert retired["final_snapshot"]["file"] == "weights-step-00000002.pt"

    snapshot_directory = path / "snapshots"
    index = json.loads((snapshot_directory / "index.json").read_text(encoding="utf-8"))
    assert index["artifact_kind"] == "weights_only_snapshot_index"
    assert [entry["step"] for entry in index["snapshots"]] == [1, 2]
    for entry in index["snapshots"]:
        snapshot_path = snapshot_directory / entry["file"]
        assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == entry["sha256"]
        assert len(entry["model_state_sha256"]) == 64
        payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        assert payload["artifact_kind"] == "weights_only_snapshot"
        assert payload["model_state_sha256"] == entry["model_state_sha256"]
        assert "model_state" in payload
        assert "optimizer_state" not in payload
        assert "train_state" not in payload
        assert "hook_state" not in payload


def test_backend_records_deterministic_numerical_contract(tmp_path: Path) -> None:
    config = _config(
        run={"save_checkpoints": False, "checkpoint_steps": []},
        train={
            "deterministic_algorithms": True,
            "allow_tf32": False,
            "cublas_workspace_config": ":4096:8",
        },
    )
    outcome = run_one(
        build_plan(config)[0],
        repo=_repo(tmp_path),
        output_root=tmp_path / "artifacts",
        backend=GoalZendoExperiment(model_loader=_loader),
    )
    model_manifest = json.loads(
        (Path(outcome.path) / "manifests" / "model.json").read_text(encoding="utf-8")
    )
    execution = model_manifest["metadata"]["numerical_execution"]
    assert execution["deterministic_algorithms"] is True
    assert execution["deterministic_warn_only"] is False
    assert execution["cuda_matmul_allow_tf32"] is False
    assert execution["cudnn_allow_tf32"] is False
    assert execution["cublas_workspace_config"] == ":4096:8"


def test_outcome_rl_backend_logs_sampling_collapse_diagnostics_each_step(
    tmp_path: Path,
) -> None:
    config = _config(
        run={"save_checkpoints": False, "checkpoint_steps": []},
        train={
            "algorithm": "outcome_rl",
            "steps": 3,
            "eval_steps": [0, 3],
            "samples_per_prompt": 4,
        },
    )
    spec = build_plan(config)[0]
    outcome = run_one(
        spec,
        repo=_repo(tmp_path),
        output_root=tmp_path / "artifacts",
        backend=GoalZendoExperiment(model_loader=_loader),
    )
    optimization = [
        row
        for row in read_jsonl(Path(outcome.path) / "metrics.jsonl")
        if row["kind"] == "optimization"
    ]
    assert sorted(row["step"] for row in optimization) == [1, 2, 3]
    for row in optimization:
        assert 0.0 <= row["sampled_b_rate"] <= 1.0
        assert 0.0 <= row["both_actions_sampled_fraction"] <= 1.0
        assert 0.0 <= row["all_zero_loo_advantages_fraction"] <= 1.0
        assert (
            row["both_actions_sampled_fraction"]
            + row["all_zero_loo_advantages_fraction"]
        ) == pytest.approx(1.0)


def test_expected_outcome_rl_backend_logs_exact_reward_without_sampling_fields(
    tmp_path: Path,
) -> None:
    config = _config(
        run={"save_checkpoints": False, "checkpoint_steps": []},
        train={
            "algorithm": "expected_outcome_rl",
            "steps": 3,
            "eval_steps": [0, 3],
            "entropy_coefficient": 0.01,
        },
    )
    outcome = run_one(
        build_plan(config)[0],
        repo=_repo(tmp_path),
        output_root=tmp_path / "artifacts",
        backend=GoalZendoExperiment(model_loader=_loader),
    )
    optimization = [
        row
        for row in read_jsonl(Path(outcome.path) / "metrics.jsonl")
        if row["kind"] == "optimization"
    ]
    assert sorted(row["step"] for row in optimization) == [1, 2, 3]
    for row in optimization:
        assert 0.0 <= row["mean_reward"] <= 1.0
        assert row["entropy"] is not None
        assert row["sampled_b_rate"] is None
        assert row["both_actions_sampled_fraction"] is None
        assert row["all_zero_loo_advantages_fraction"] is None
        geometry = row["loss_geometry"]
        assert geometry["gradient_exponent"] == 1.0
        assert geometry["strata"]["all"]["n"] == 4
        assert "sage_agreement" in geometry["strata"]


@pytest.mark.parametrize(
    ("algorithm", "exponent"),
    [
        ("tempered_outcome_control", 0.5),
        ("logprob_outcome_control", 0.0),
    ],
)
def test_power_gradient_backend_logs_algorithm_identity_and_loss_geometry(
    tmp_path: Path,
    algorithm: str,
    exponent: float,
) -> None:
    config = _config(
        run={"save_checkpoints": False, "checkpoint_steps": []},
        train={
            "algorithm": algorithm,
            "reward_gradient_exponent": exponent,
            "steps": 2,
            "eval_steps": [0, 2],
            "entropy_coefficient": 0.01,
        },
    )
    outcome = run_one(
        build_plan(config)[0],
        repo=_repo(tmp_path),
        output_root=tmp_path / "artifacts",
        backend=GoalZendoExperiment(model_loader=_loader),
    )
    optimization = [
        row
        for row in read_jsonl(Path(outcome.path) / "metrics.jsonl")
        if row["kind"] == "optimization"
    ]
    assert sorted(row["step"] for row in optimization) == [1, 2]
    for row in optimization:
        assert row["algorithm"] == algorithm
        assert row["loss_geometry"]["gradient_exponent"] == exponent
        strata = row["loss_geometry"]["strata"]
        assert strata["all"]["n"] == 4
        assert 0.0 <= strata["all"]["mean_correct_probability"] <= 1.0
        assert 0.0 <= strata["all"]["mean_active_policy_gradient_magnitude"] <= 1.0
        diagnostics = row["objective_gradient_diagnostics"]
        assert diagnostics["schema_version"] == 1
        assert diagnostics["gradient_exponent"] == exponent
        assert diagnostics["mean_correct_probability"] == pytest.approx(
            strata["all"]["mean_correct_probability"],
            abs=1e-6,
        )
        assert diagnostics["mean_expected_reward_gradient_magnitude"] == pytest.approx(
            strata["all"]["mean_expected_reward_gradient_magnitude"],
            abs=1e-6,
        )
        assert diagnostics["mean_logprob_gradient_magnitude"] == pytest.approx(
            strata["all"]["mean_logprob_gradient_magnitude"],
            abs=1e-6,
        )
        assert diagnostics["mean_policy_gradient_magnitude"] == pytest.approx(
            strata["all"]["mean_active_policy_gradient_magnitude"],
            abs=1e-6,
        )
        assert row["sampled_b_rate"] is None


def test_interrupted_backend_resumes_from_atomic_boundary_without_duplicate_rows(
    tmp_path: Path,
) -> None:
    config = _config()
    spec = build_plan(config)[0]
    repo = _repo(tmp_path)
    output = tmp_path / "artifacts"

    def interrupt_after_step_two(step: int) -> None:
        if step == 2:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_one(
            spec,
            repo=repo,
            output_root=output,
            backend=GoalZendoExperiment(
                model_loader=_loader,
                evaluation_observer=interrupt_after_step_two,
            ),
        )
    run_path = next(output.glob("**/identity.json")).parent
    latest = json.loads((run_path / "checkpoints" / "latest.json").read_text())
    assert latest["step"] == 1
    resume_payload = torch.load(
        run_path / "checkpoints" / latest["file"],
        map_location="cpu",
        weights_only=False,
    )
    assert resume_payload["artifact_kind"] == "resumable_checkpoint"
    assert {"optimizer_state", "train_state", "hook_state"} <= set(resume_payload)
    orphan = run_path / "checkpoints" / "resume-step-99999999.pt"
    shutil.copyfile(run_path / "checkpoints" / latest["file"], orphan)

    completed = run_one(
        spec,
        repo=repo,
        output_root=output,
        backend=GoalZendoExperiment(model_loader=_loader),
    )
    assert completed.state == "complete"
    metrics = read_jsonl(Path(completed.path) / "metrics.jsonl")
    predictions = read_jsonl(Path(completed.path) / "predictions.jsonl")
    assert len({row["record_id"] for row in metrics}) == len(metrics)
    assert len({row["record_id"] for row in predictions}) == len(predictions)
    assert json.loads(
        (Path(completed.path) / "checkpoints" / "retired.json").read_text()
    )["step"] == 2
    assert not (Path(completed.path) / "checkpoints" / "latest.json").exists()
    assert not orphan.exists()
    assert not tuple((Path(completed.path) / "checkpoints").glob("resume-step-*.pt"))
