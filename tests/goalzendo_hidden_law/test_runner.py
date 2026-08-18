from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import goalzendo_hidden_law.runner as runner_module
from goalzendo_hidden_law.artifacts import HiddenLawRunStore, verify_completed_run
from goalzendo_hidden_law.config import build_hidden_law_plan, load_hidden_law_config
from goalzendo_hidden_law.game import build_small_bank
from goalzendo_hidden_law.runner import (
    HiddenLawRunnerError,
    execute_condition,
    shard_plan,
    smoke_condition,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/goalzendo/qwen35_hidden_law_finite_choice.yaml"


class FakeBank:
    def __init__(self) -> None:
        fixture = build_small_bank()
        self.training_families = fixture.families
        self.evaluation_families = fixture.families
        self.interim_evaluation_families = fixture.families
        self.digest = "1" * 64
        self.pairing_key = "2" * 64
        self.manifest = {"fake": True, "training": 2, "evaluation": 2}


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 256
    name_or_path = "fake-tokenizer"
    chat_template = "fake-chat-template"

    def __init__(self, revision: str) -> None:
        self.init_kwargs = {"_commit_hash": revision}

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        values = [ord(character) for character in text]
        return [1, *values] if add_special_tokens else values

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return "\n".join(item["content"] for item in messages) + "\nANSWER:"


class FakeModel(nn.Module):
    def __init__(self, revision: str) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-0.2, 0.2, 9))
        self.config = SimpleNamespace(_commit_hash=revision, use_cache=True)
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True


class FakePolicy:
    def __init__(
        self,
        model: FakeModel,
        _tokenizer: FakeTokenizer,
        *,
        max_prompt_tokens: int,
        max_batch_size: int,
    ) -> None:
        assert max_prompt_tokens == 1536 and max_batch_size == 4
        self.model = model
        self.forward_calls = 0
        self.scored_prompt_count = 0
        self.scored_prompt_tokens_unpadded = 0
        self.maximum_prompt_tokens = 100

    def score(self, prompts: tuple[str, ...], action_labels: tuple[str, ...]) -> torch.Tensor:
        self.forward_calls += 1
        self.scored_prompt_count += len(prompts)
        self.scored_prompt_tokens_unpadded += sum(len(prompt) for prompt in prompts)
        indices = torch.tensor([ord(label) - ord("A") for label in action_labels])
        return self.model.bias[indices][None, :].expand(len(prompts), -1)


def _loader(model_config: dict[str, object], _update: dict[str, object]) -> tuple[FakeModel, FakeTokenizer]:
    revision = str(model_config["revision"])
    return FakeModel(revision), FakeTokenizer(revision)


def test_native_hash_shards_are_disjoint_and_complete() -> None:
    plan = build_hidden_law_plan(load_hidden_law_config(CONFIG))
    shards = [shard_plan(plan, shard_index=index, num_shards=7) for index in range(7)]
    assert sum(len(shard) for shard in shards) == 24
    assert {item.run_id for item in plan} == {item.run_id for shard in shards for item in shard}
    assert all(
        not ({item.run_id for item in shards[left]} & {item.run_id for item in shards[right]})
        for left in range(7)
        for right in range(left + 1, 7)
    )


def test_registered_qwen_runtime_identity_is_exact() -> None:
    condition = build_hidden_law_plan(load_hidden_law_config(CONFIG))[0]
    manifest: dict[str, object] = {
        "requested_model": condition.model_name,
        "requested_revision": condition.model_revision,
        "resolved_revision": None,
        "tokenizer_resolved_revision": None,
        "requested_dtype": "bfloat16",
        "action_labels": ["A", "B"],
        "finite_action_alphabets": {
            "binary": ["A", "B"],
            "candidate": ["A", "B", "C", "D"],
            "query": list("ABCDEFGHI"),
        },
        "gradient_checkpointing": True,
        "use_cache": False,
        "full_model_update": True,
        "parameter_count": 752_393_024,
        "trainable_parameter_count": 752_393_024,
        "trainable_parameter_dtype_counts": {"torch.bfloat16": 752_393_024},
        "model_class": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM",
        "tokenizer_class": "transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast",
        "tokenizer_name_or_path": condition.model_name,
        "dependency_versions": {
            "torch": "2.8.0+cu128",
            "torch_cuda": "12.8",
            "transformers": "5.15.0",
            "tokenizers": "0.22.2",
            "peft": "0.20.0",
            "accelerate": "1.14.0",
            "huggingface_hub": "1.5.0",
            "safetensors": "0.8.0",
        },
    }
    runner_module._validate_loaded_model_provenance(
        manifest,
        condition,
        strict_backend_identity=True,
    )
    manifest["parameter_count"] = 1
    manifest["trainable_parameter_count"] = 1
    with pytest.raises(HiddenLawRunnerError, match="pinned execution stack"):
        runner_module._validate_loaded_model_provenance(
            manifest,
            condition,
            strict_backend_identity=True,
        )


def test_scientific_execution_requires_frozen_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_hidden_law_config(CONFIG)
    config["experiment"]["status"] = "prospective_draft"
    monkeypatch.setattr(runner_module, "load_hidden_law_config", lambda _path: config)
    with pytest.raises(HiddenLawRunnerError, match="prospectively frozen"):
        runner_module.execute_plan(
            CONFIG,
            repo_root=ROOT,
            shard_index=0,
            num_shards=1,
        )


def test_jsonl_repairs_only_a_torn_final_suffix(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    complete = b'{"record_id":"kept","step":0}\n'
    path.write_bytes(complete + b'{"record_id":"torn"')
    assert runner_module._jsonl_rows(path) == [{"record_id": "kept", "step": 0}]
    assert path.read_bytes() == complete

    path.write_bytes(complete + b'{"record_id":}\n')
    with pytest.raises(HiddenLawRunnerError, match="invalid JSONL"):
        runner_module._jsonl_rows(path)


def test_checkpoint_resume_then_complete_runs_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_hidden_law_config(CONFIG)
    source = build_hidden_law_plan(config)[0]
    condition = smoke_condition(source, int(config["run"]["smoke_seed"]))
    output = tmp_path / "runs"
    original_finish = HiddenLawRunStore.finish

    def interrupt_before_seal(self: HiddenLawRunStore, summary: object) -> object:
        del self, summary
        raise RuntimeError("simulated interruption after checkpoint")

    monkeypatch.setattr(HiddenLawRunStore, "finish", interrupt_before_seal)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        execute_condition(
            condition,
            config,
            repo_root=ROOT,
            output_root=output,
            smoke=True,
            bank_builder=lambda _seed: FakeBank(),  # type: ignore[arg-type]
            model_loader=_loader,  # type: ignore[arg-type]
            policy_factory=FakePolicy,
        )
    run = output / "execution-smoke" / condition.run_id
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["last_step"] == 1
    assert (run / "checkpoints/latest.json").is_file()

    monkeypatch.setattr(HiddenLawRunStore, "finish", original_finish)
    resumed = execute_condition(
        condition,
        config,
        repo_root=ROOT,
        output_root=output,
        smoke=True,
        bank_builder=lambda _seed: FakeBank(),  # type: ignore[arg-type]
        model_loader=_loader,  # type: ignore[arg-type]
        policy_factory=FakePolicy,
    )
    assert resumed["status"] == "complete"
    verify_completed_run(run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    completed_status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert completed_status["state"] == completed_status["phase"] == "complete"
    assert summary["operational"]["forward_calls"] > 0
    assert summary["operational"]["scored_prompt_tokens_unpadded"] > 0
    assert not list(run.glob("checkpoints/*.pt"))
    assert not (run / "checkpoints/latest.json").exists()
    skipped = execute_condition(
        condition,
        config,
        repo_root=ROOT,
        output_root=output,
        smoke=True,
        bank_builder=lambda _seed: FakeBank(),  # type: ignore[arg-type]
        model_loader=_loader,  # type: ignore[arg-type]
        policy_factory=FakePolicy,
    )
    assert skipped["status"] == "skipped_complete"


def test_uncheckpointed_record_suffix_is_removed_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_hidden_law_config(CONFIG)
    source = build_hidden_law_plan(config)[0]
    condition = smoke_condition(source, int(config["run"]["smoke_seed"]))
    output = tmp_path / "suffix"
    original_save = runner_module.save_checkpoint

    def interrupt_before_checkpoint(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated pre-checkpoint interruption")

    monkeypatch.setattr(runner_module, "save_checkpoint", interrupt_before_checkpoint)
    with pytest.raises(RuntimeError, match="pre-checkpoint"):
        execute_condition(
            condition,
            config,
            repo_root=ROOT,
            output_root=output,
            smoke=True,
            bank_builder=lambda _seed: FakeBank(),  # type: ignore[arg-type]
            model_loader=_loader,  # type: ignore[arg-type]
            policy_factory=FakePolicy,
        )
    run = output / "execution-smoke" / condition.run_id
    metric = json.loads((run / "metrics.jsonl").read_text(encoding="utf-8"))
    metric["loss"] = 999.0
    (run / "metrics.jsonl").write_text(json.dumps(metric) + "\n", encoding="utf-8")
    (run / "transcripts.jsonl").write_text(
        json.dumps({"record_id": "stale:transcript", "step": 1}) + "\n",
        encoding="utf-8",
    )
    (run / "predictions.jsonl").write_text(
        json.dumps({"record_id": "stale:prediction", "step": 1}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(runner_module, "save_checkpoint", original_save)
    result = execute_condition(
        condition,
        config,
        repo_root=ROOT,
        output_root=output,
        smoke=True,
        bank_builder=lambda _seed: FakeBank(),  # type: ignore[arg-type]
        model_loader=_loader,  # type: ignore[arg-type]
        policy_factory=FakePolicy,
    )
    assert result["status"] == "complete"
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["loss"] != 999.0
    assert (run / "transcripts.jsonl").read_text(encoding="utf-8") == ""
    assert (run / "predictions.jsonl").read_text(encoding="utf-8") == ""
    assert not list(run.glob("checkpoints/*.pt"))


def test_execution_failure_is_written_to_status(tmp_path: Path) -> None:
    config = load_hidden_law_config(CONFIG)
    source = build_hidden_law_plan(config)[0]
    condition = smoke_condition(source, int(config["run"]["smoke_seed"]))

    def fail_loader(_model: object, _update: object) -> object:
        raise RuntimeError("model load failed")

    with pytest.raises(RuntimeError, match="model load failed"):
        execute_condition(
            condition,
            config,
            repo_root=ROOT,
            output_root=tmp_path / "failed",
            smoke=True,
            bank_builder=lambda _seed: FakeBank(),  # type: ignore[arg-type]
            model_loader=fail_loader,  # type: ignore[arg-type]
        )
    run = tmp_path / "failed" / "execution-smoke" / condition.run_id
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["phase"] == "exception"
    assert "model load failed" in status["error"]


def test_smoke_refuses_a_different_resolved_model_revision(tmp_path: Path) -> None:
    config = load_hidden_law_config(CONFIG)
    source = build_hidden_law_plan(config)[0]
    condition = smoke_condition(source, int(config["run"]["smoke_seed"]))

    def wrong_revision_loader(
        model_config: dict[str, object], _update: dict[str, object]
    ) -> tuple[FakeModel, FakeTokenizer]:
        del model_config
        wrong = "0" * 40
        return FakeModel(wrong), FakeTokenizer(wrong)

    with pytest.raises(HiddenLawRunnerError, match="registered identity"):
        execute_condition(
            condition,
            config,
            repo_root=ROOT,
            output_root=tmp_path / "wrong-revision",
            smoke=True,
            bank_builder=lambda _seed: FakeBank(),  # type: ignore[arg-type]
            model_loader=wrong_revision_loader,  # type: ignore[arg-type]
            policy_factory=FakePolicy,
        )
