"""GoalZendo: controlled competing-rule experiments for language models."""

from .generation import (
    generate_factorial_evaluation,
    generate_known_law_dataset,
    joint_error_bounds,
    resolve_joint_error_count,
)
from .interventions import (
    InterventionAudit,
    audit_intervention,
    changed_channels,
    flip_distractor,
    flip_herald,
    flip_law,
    flip_sage,
    is_one_atom_law_boundary,
    paired_channel_flips,
    paired_interventions,
    paired_rule_preserving_controls,
    preserve_law,
    preserve_sage,
)
from .metrics import behavioral_metrics, causal_flip_metrics, diagnostic_panel, factorial_cell
from .rendering import render_action, render_known_law_decision, render_koan
from .rules import describe_rule, evaluate_rule, symbolic_formula, truth_table
from .schema import (
    Choice,
    KnownLawDataset,
    KnownLawDecision,
    Koan,
    RuleSpec,
    Scene,
    stable_digest,
)

__all__ = [
    "Choice",
    "InterventionAudit",
    "KnownLawDataset",
    "KnownLawDecision",
    "Koan",
    "RuleSpec",
    "Scene",
    "audit_intervention",
    "behavioral_metrics",
    "causal_flip_metrics",
    "changed_channels",
    "describe_rule",
    "diagnostic_panel",
    "evaluate_rule",
    "factorial_cell",
    "flip_distractor",
    "flip_herald",
    "flip_law",
    "flip_sage",
    "generate_factorial_evaluation",
    "generate_known_law_dataset",
    "is_one_atom_law_boundary",
    "joint_error_bounds",
    "paired_channel_flips",
    "paired_interventions",
    "paired_rule_preserving_controls",
    "preserve_law",
    "preserve_sage",
    "render_action",
    "render_known_law_decision",
    "render_koan",
    "resolve_joint_error_count",
    "stable_digest",
    "symbolic_formula",
    "truth_table",
]
