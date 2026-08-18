from __future__ import annotations

from dataclasses import replace

import pytest

from goalzendo_interactive import (
    AbortEvent,
    AnswerAction,
    AnswerEvent,
    Atom,
    BinaryRule,
    HiddenEpisode,
    HiddenLawEnvironment,
    InvalidEvent,
    Literal,
    Observation,
    ReadyAction,
    ReadyEvent,
    TestAction,
    TestEvent,
    Transcript,
    TranscriptReplayError,
    entropy_reduction,
    optimal_legal_query,
    parse_transcript,
    play_reference_episode,
    query_metrics,
    replay_transcript,
    run_reference_inquiry,
    serialize_action,
    serialize_transcript,
    terminal_classifications,
    truth_vector,
)


def test_globally_optimal_query_and_information_metrics_are_exact(
    hidden_episode: HiddenEpisode,
) -> None:
    space = hidden_episode.opening_version_space()
    choice = optimal_legal_query(space)
    assert choice.scene_index == 10_944
    assert (choice.rejected_count, choice.accepted_count) == (24, 23)
    assert choice.tie_break_digest == (
        "00142aaa951e7bea25957a57435a296e42934faef32c4cce1578f48cefaac556"
    )
    assert choice.expected_entropy_reduction == entropy_reduction(47, 23)
    # A 23/24 split reaches the theoretical best balance for 47 hypotheses.
    assert min(choice.rejected_count, choice.accepted_count) == len(space) // 2

    metrics, after, observation = query_metrics(
        space,
        choice.scene_index,
        target=hidden_episode.target,
        shadow=hidden_episode.shadow,
    )
    assert observation == Observation(10_944, False)
    assert len(after) == metrics.after_count == 24
    assert metrics.realized_version_space_reduction == 23
    assert metrics.expected_entropy_reduction == metrics.best_expected_entropy_reduction
    assert metrics.regret == 0.0
    assert metrics.fraction_of_maximum_gain == 1.0
    assert metrics.separates_target_from_placard is False
    assert metrics.separates_target_from_shadow is True

    repeated, repeated_space, _ = query_metrics(
        after,
        choice.scene_index,
        target=hidden_episode.target,
        shadow=hidden_episode.shadow,
        observed_scene_indices=frozenset({choice.scene_index}),
    )
    assert repeated.duplicate is True
    assert repeated.expected_entropy_reduction == 0.0
    assert repeated.regret > 0.0
    assert repeated.realized_version_space_reduction == 0
    assert repeated_space == after


def test_reference_expert_identifies_constructed_episode_within_budget(
    hidden_episode: HiddenEpisode,
) -> None:
    inquiry = run_reference_inquiry(hidden_episode)
    assert inquiry.target_identified is True
    assert len(inquiry.queries) == 5
    assert [query.choice.scene_index for query in inquiry.queries] == [
        10_944,
        3_970,
        1_120,
        441,
        10_810,
    ]
    assert [query.before_count for query in inquiry.queries] == [47, 24, 10, 5, 2]
    assert [query.after_count for query in inquiry.queries] == [24, 10, 5, 2, 1]
    assert inquiry.final_space.indices == (hidden_episode.target.index,)
    observed = {observation.scene_index for observation in hidden_episode.opening}
    for query in inquiry.queries:
        assert query.choice.scene_index not in observed
        observed.add(query.choice.scene_index)


def test_complete_reference_game_scores_and_replays_exactly(hidden_episode: HiddenEpisode) -> None:
    transcript = play_reference_episode(hidden_episode)
    assert transcript.state == "complete"
    assert len(transcript.events) == 7
    assert sum(isinstance(event, TestEvent) for event in transcript.events) == 5
    assert isinstance(transcript.events[-2], ReadyEvent)
    assert transcript.events[-2].premature is False
    assert isinstance(transcript.events[-1], AnswerEvent)
    score = transcript.events[-1].score
    assert score.classification_accuracy == 1.0
    assert score.rule_equivalent is True
    assert score.query_count == 5
    assert score.reward == pytest.approx(0.9583333333333333)
    assert transcript.digest == "8e46ce342d78af437515a17468f4cdb09d41aa012f64f495fa2b52fab06f179c"

    encoded = serialize_transcript(transcript)
    parsed = parse_transcript(encoded)
    assert parsed == transcript
    assert parsed.digest == transcript.digest
    replayed = replay_transcript(hidden_episode, parsed)
    assert replayed.transcript == transcript
    assert replayed.final_reward == score.reward

    canonical = transcript.as_obj()

    def assert_no_floats(value: object) -> None:
        assert not isinstance(value, float)
        if isinstance(value, dict):
            for item in value.values():
                assert_no_floats(item)
        elif isinstance(value, list):
            for item in value:
                assert_no_floats(item)

    assert_no_floats(canonical)


def test_replay_rejects_a_tampered_or_wrong_episode_transcript(hidden_episode: HiddenEpisode) -> None:
    transcript = play_reference_episode(hidden_episode)
    first = transcript.events[0]
    assert isinstance(first, TestEvent)
    tampered_event = replace(
        first,
        observation=Observation(first.observation.scene_index, not first.observation.accepted),
    )
    tampered = Transcript(
        transcript.episode_digest,
        transcript.opening,
        (tampered_event, *transcript.events[1:]),
        transcript.state,
    )
    with pytest.raises(TranscriptReplayError, match="differs"):
        replay_transcript(hidden_episode, tampered)

    forged_invalid_type = Transcript(
        hidden_episode.digest,
        hidden_episode.opening,
        (InvalidEvent('{"move":"ready"}', "invalid_action_type", "forged"),),
        "invalid",
    )
    with pytest.raises(TranscriptReplayError, match="differs"):
        replay_transcript(hidden_episode, forged_invalid_type)


def test_premature_ready_and_extensional_rule_equivalence_scoring(
    hidden_episode: HiddenEpisode,
) -> None:
    environment = HiddenLawEnvironment(hidden_episode)
    with pytest.raises(RuntimeError, match="withheld"):
        environment.terminal_scenes()
    ready = environment.consume(ReadyAction())
    assert ready.outcome == "premature_ready"
    assert environment.state == "awaiting_answer"
    assert len(environment.terminal_scenes()) == 16

    first, second = hidden_episode.target.rule.args
    equivalent_but_distinct = BinaryRule(
        "exactly_one",
        (
            Literal(first.atom, negated=not first.negated),
            Literal(second.atom, negated=not second.negated),
        ),
    )
    assert equivalent_but_distinct != hidden_episode.target.rule
    assert truth_vector(equivalent_but_distinct) == hidden_episode.target.truth
    answer = AnswerAction(
        equivalent_but_distinct,
        terminal_classifications(hidden_episode),  # type: ignore[arg-type]
    )
    result = environment.consume(answer)
    assert result.outcome == "complete"
    assert result.reward == 1.0
    assert environment.terminal_score is not None
    assert environment.terminal_score.rule_equivalent is True


def test_duplicate_queries_are_legal_recorded_and_still_consume_budget(
    hidden_episode: HiddenEpisode,
) -> None:
    environment = HiddenLawEnvironment(hidden_episode)
    initial_space = environment.version_space
    duplicate_action = TestAction(hidden_episode.opening[0].scene)
    first = environment.consume(duplicate_action)
    assert first.outcome == "observation"
    assert isinstance(first.event, TestEvent)
    assert first.event.metrics.duplicate is True
    assert first.event.metrics.realized_version_space_reduction == 0
    assert environment.version_space == initial_space

    for expected_count in range(2, 7):
        result = environment.consume(duplicate_action)
        assert environment.query_count == expected_count
    assert result.outcome == "budget_exhausted"
    assert environment.state == "awaiting_answer"

    seventh = environment.consume(duplicate_action)
    assert seventh.outcome == "invalid"
    assert seventh.error_code == "expected_answer"
    assert environment.state == "invalid"
    assert environment.final_reward == 0.0


def test_accidental_withheld_terminal_scene_collision_remains_a_legal_query(
    hidden_episode: HiddenEpisode,
) -> None:
    environment = HiddenLawEnvironment(hidden_episode)
    terminal_scene = hidden_episode.terminal[0].scene
    result = environment.consume(TestAction(terminal_scene))
    assert result.outcome == "observation"
    assert result.duplicate is False
    with pytest.raises(RuntimeError, match="withheld"):
        environment.terminal_scenes()

    environment.consume(ReadyAction())
    assert terminal_scene in environment.terminal_scenes()


@pytest.mark.parametrize(
    ("action", "code"),
    [
        ('{"move":"nonsense"}', "invalid_move"),
        ('{ "move": "ready" }', "noncanonical_json"),
        (
            serialize_action(
                AnswerAction(
                    Literal(Atom("exists", attribute="color", value="red")),
                    ("fits",),
                )
            ),
            "unexpected_answer",
        ),
    ],
)
def test_invalid_actions_end_the_trajectory_with_zero_reward(
    hidden_episode: HiddenEpisode,
    action: str,
    code: str,
) -> None:
    environment = HiddenLawEnvironment(hidden_episode)
    result = environment.consume(action)
    assert result.outcome == "invalid"
    assert result.error_code == code
    assert environment.state == "invalid"
    assert environment.final_reward == 0.0
    again = environment.consume('{"move":"ready"}')
    assert again.outcome == "already_terminal"
    assert len(environment.transcript.events) == 1


def test_unsupported_python_objects_raise_without_creating_unreplayable_events(
    hidden_episode: HiddenEpisode,
) -> None:
    environment = HiddenLawEnvironment(hidden_episode)
    with pytest.raises(TypeError, match="canonical Action or JSON string"):
        environment.consume(object())  # type: ignore[arg-type]
    assert environment.state == "inquiry"
    assert environment.transcript.events == ()


@pytest.mark.parametrize("reason", ["incomplete", "overlength", "timed_out"])
def test_external_abort_is_canonical_replayable_and_scores_zero(
    hidden_episode: HiddenEpisode,
    reason: str,
) -> None:
    environment = HiddenLawEnvironment(hidden_episode)
    environment.consume(TestAction(hidden_episode.opening[0].scene))
    result = environment.abort(reason)  # type: ignore[arg-type]
    assert result.outcome == "aborted"
    assert result.reward == 0.0
    assert result.error_code == reason
    assert isinstance(result.event, AbortEvent)
    assert result.event.reason == reason
    assert environment.state == "aborted"
    assert environment.final_reward == 0.0
    with pytest.raises(RuntimeError, match="zero-reward"):
        environment.terminal_scenes()

    encoded = serialize_transcript(environment.transcript)
    parsed = parse_transcript(encoded)
    replayed = replay_transcript(hidden_episode, parsed)
    assert replayed.transcript == environment.transcript
    assert replayed.final_reward == 0.0


def test_wrong_terminal_length_is_not_repaired(hidden_episode: HiddenEpisode) -> None:
    environment = HiddenLawEnvironment(hidden_episode)
    environment.consume(ReadyAction())
    short = AnswerAction(hidden_episode.target.rule, ("fits",))
    result = environment.consume(short)
    assert result.outcome == "invalid"
    assert result.error_code == "wrong_classification_count"
    assert environment.final_reward == 0.0
