from __future__ import annotations

import pytest

from goalzendo_interactive import HiddenEpisode, Observation, build_rule_catalog


@pytest.fixture(scope="session")
def hidden_episode() -> HiddenEpisode:
    catalog = build_rule_catalog()
    target = catalog[4_562]
    shadow = catalog[0]
    opening_indices = (79, 6_880, 2_308, 8_812, 2_768, 10_759, 6_096, 6_879, 91, 6_881)
    terminal_indices = (
        6_858,
        6_859,
        6_897,
        6_899,
        0,
        1,
        39,
        41,
        6_917,
        6_918,
        6_937,
        6_943,
        59,
        60,
        85,
        159,
    )
    return HiddenEpisode(
        episode_id="fixture-perfect-factorial-v1",
        target=target,
        shadow=shadow,
        opening=tuple(Observation(index, target.truth[index]) for index in opening_indices),
        terminal=tuple(Observation(index, target.truth[index]) for index in terminal_indices),
        regime="perfect_ambiguity",
        terminal_kind="factorial",
        renderer="train_compact",
    )


@pytest.fixture(scope="session")
def noisy_episode() -> HiddenEpisode:
    catalog = build_rule_catalog()
    target = catalog[4_562]
    shadow = catalog[0]
    opening_indices = (6_937, 6_899, 142, 8_083, 3_030, 8_082, 3_032, 8_443, 3_754, 6_898)
    terminal_indices = (6_943, 6_897, 59, 6_858, 79, 6_859, 85, 6_860, 91, 6_861)
    return HiddenEpisode(
        episode_id="fixture-noisy-train-v1",
        target=target,
        shadow=shadow,
        opening=tuple(Observation(index, target.truth[index]) for index in opening_indices),
        terminal=tuple(Observation(index, target.truth[index]) for index in terminal_indices),
        regime="noisy_shortcuts",
        terminal_kind="train_like",
        renderer="train_positional",
    )
