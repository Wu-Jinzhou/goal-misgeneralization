from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from goalzendo_interactive import cli
from goalzendo_interactive.episodes import HiddenEpisode


def test_bank_plan_cli_is_canonical_and_non_authorizing(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(("bank-plan", "--with-qa")) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["plan"]["planned_request_count"] == 2_816
    assert value["plan"]["materialization_authorized"] is False
    assert value["qa"]["structural_checks_passed"] is True
    assert value["qa"]["materialization_authorized"] is False


def test_example_cli_renders_reference_and_hides_evaluator_state_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    hidden_episode: HiddenEpisode,
) -> None:
    path = tmp_path / "bank.json"
    path.write_text("{}\n", encoding="ascii")
    fake_bank = SimpleNamespace(
        episodes=(hidden_episode,),
        spec=SimpleNamespace(bank_id="fake-bank"),
        digest="a" * 64,
    )
    monkeypatch.setattr(cli, "parse_episode_bank", lambda _: fake_bank)
    assert cli.main(("example", str(path))) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["scope"] == "deterministic_reference_example_not_model_result"
    assert value["episode_id"] == hidden_episode.episode_id
    assert value["dialogue"][-1]["role"] == "assistant"
    assert "evaluator_only" not in value

    assert cli.main(("example", str(path), "--include-hidden")) == 0
    revealed = json.loads(capsys.readouterr().out)
    assert revealed["evaluator_only"]["target_rule_id"] == hidden_episode.target.rule_id


def test_cli_rejects_missing_canonical_newline(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="ascii")
    with pytest.raises(SystemExit) as exc:
        cli.main(("example", str(path)))
    assert exc.value.code == 2
