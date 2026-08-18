"""Tests for sweep execution metadata and final evaluation bookkeeping."""

from __future__ import annotations

from forkworld.runner import _terminal_training_counters


def test_terminal_training_counters_use_protocol_realized_progress() -> None:
    metrics = [
        {"global_step": 2, "examples_seen": 64},
        {"global_step": 11, "examples_seen": 704},
        {"global_step": 7, "examples_seen": 448},
    ]

    assert _terminal_training_counters(metrics, fallback_step=99) == (11, 704)


def test_terminal_training_counters_fall_back_when_protocol_emits_none() -> None:
    assert _terminal_training_counters([], fallback_step=37) == (37, None)

