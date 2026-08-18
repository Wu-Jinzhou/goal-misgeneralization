from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from goalzendo_interactive import (
    EVAL_RENDERERS,
    RENDERERS,
    TRAIN_RENDERERS,
    EpisodeValidationError,
    HiddenEpisode,
    Observation,
    build_rule_catalog,
    render_scene,
    renderer_digest,
)


def test_hidden_episode_binds_catalog_and_exact_evidence_invariants(
    hidden_episode: HiddenEpisode,
) -> None:
    assert hidden_episode.target.rule_id == "g03r04562"
    assert hidden_episode.shadow.rule_id == "g03r00000"
    assert len(hidden_episode.opening) == 10
    assert len(hidden_episode.terminal) == 16
    assert sum(observation.accepted for observation in hidden_episode.opening) == 5
    assert len(hidden_episode.opening_version_space()) == 47
    assert hidden_episode.target.index in hidden_episode.opening_version_space().indices
    assert min(hidden_episode.target_shadow_cells.values()) >= 128
    disagreement = (
        hidden_episode.target_shadow_cells[(False, True)]
        + hidden_episode.target_shadow_cells[(True, False)]
    ) / 13_716
    assert 0.35 <= disagreement <= 0.65
    assert hidden_episode.digest == "277839cbd482606dc510de7626d8278400b65d3fccfaf37077b61bfe2022263e"

    cells: dict[tuple[bool, bool, bool], int] = {}
    for observation in hidden_episode.terminal:
        cell = (
            observation.accepted,
            observation.scene.placard == "sun",
            hidden_episode.shadow.truth[observation.scene_index],
        )
        cells[cell] = cells.get(cell, 0) + 1
    assert len(cells) == 8
    assert set(cells.values()) == {2}


def test_noisy_train_like_episode_has_exact_registered_proxy_counts(
    noisy_episode: HiddenEpisode,
) -> None:
    assert len(noisy_episode.opening_version_space()) == 23
    assert noisy_episode.digest == "15579e3c16d3e3d619f4e7c835ae0baebd22f61416b7a32dc7c3cfa4e999c9a0"
    for observations in (noisy_episode.opening, noisy_episode.terminal):
        assert len(observations) == 10
        assert sum(observation.accepted for observation in observations) == 5
        placard_agreement = sum(
            (observation.scene.placard == "sun") is observation.accepted
            for observation in observations
        )
        shadow_agreement = sum(
            noisy_episode.shadow.truth[observation.scene_index] is observation.accepted
            for observation in observations
        )
        assert placard_agreement == 9
        assert shadow_agreement == 8
        shadow_errors = [
            observation
            for observation in observations
            if noisy_episode.shadow.truth[observation.scene_index] is not observation.accepted
        ]
        assert {observation.accepted for observation in shadow_errors} == {False, True}


def test_episode_and_observation_are_immutable(hidden_episode: HiddenEpisode) -> None:
    with pytest.raises(FrozenInstanceError):
        hidden_episode.regime = "noisy_shortcuts"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        hidden_episode.opening[0].accepted = False  # type: ignore[misc]


def test_episode_validation_rejects_label_balance_overlap_and_proxy_errors(
    hidden_episode: HiddenEpisode,
) -> None:
    first = hidden_episode.opening[0]
    wrong_label = Observation(first.scene_index, not first.accepted)
    with pytest.raises(EpisodeValidationError, match="exact target labels"):
        replace(hidden_episode, opening=(wrong_label, *hidden_episode.opening[1:]))

    with pytest.raises(EpisodeValidationError, match="unique"):
        replace(
            hidden_episode,
            opening=(hidden_episode.opening[0], hidden_episode.opening[0], *hidden_episode.opening[2:]),
        )

    overlapping_terminal = (hidden_episode.opening[0], *hidden_episode.terminal[1:])
    with pytest.raises(EpisodeValidationError, match="disjoint"):
        replace(hidden_episode, terminal=overlapping_terminal)

    # A correct target label with the wrong placard correlation violates the exact 10/10 regime.
    replacement = next(
        Observation(index, hidden_episode.target.truth[index])
        for index in range(13_716)
        if index not in {item.scene_index for item in (*hidden_episode.opening, *hidden_episode.terminal)}
        and hidden_episode.target.truth[index] is first.accepted
        and (Observation(index, hidden_episode.target.truth[index]).scene.placard == "sun")
        is not first.accepted
    )
    with pytest.raises(EpisodeValidationError, match="placard agreement"):
        replace(hidden_episode, opening=(replacement, *hidden_episode.opening[1:]))


def test_episode_rejects_noncanonical_catalog_binding(hidden_episode: HiddenEpisode) -> None:
    catalog = build_rule_catalog()
    forged = replace(catalog[4_562], index=4_561)
    with pytest.raises(EpisodeValidationError, match="canonical rule catalog"):
        replace(hidden_episode, target=forged)


def test_six_renderer_families_are_deterministic_complete_and_split(
    hidden_episode: HiddenEpisode,
) -> None:
    assert len(TRAIN_RENDERERS) == 4
    assert len(EVAL_RENDERERS) == 2
    assert set(TRAIN_RENDERERS).isdisjoint(EVAL_RENDERERS)
    assert (*TRAIN_RENDERERS, *EVAL_RENDERERS) == RENDERERS
    scene = hidden_episode.opening[0].scene
    outputs = tuple(render_scene(scene, renderer) for renderer in RENDERERS)
    assert len(set(outputs)) == 6
    assert all(
        render_scene(scene, renderer) == output
        for renderer, output in zip(RENDERERS, outputs, strict=True)
    )
    assert all(scene.placard in output.lower() for output in outputs)
    assert all(scene.left.color in output.lower() for output in outputs if scene.left is not None)
    assert renderer_digest() == "8d1c9869aeb7a38059ab6830c37552f4a350cc8c66aa1b6e393719d0d83c0da0"
