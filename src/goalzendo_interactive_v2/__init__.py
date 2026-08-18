"""Prospective, nonauthorizing G03-v2 evaluation primitives.

This package is deliberately separate from :mod:`goalzendo_interactive` so
that additions made while designing G03-v2 cannot change the frozen source
fingerprint used by the G03-G v1 pipeline smoke.
"""

from .challenge_query import (
    CHALLENGE_RESERVOIR_PANEL_COUNT,
    CHALLENGE_RESERVOIR_PANEL_SIZE,
    CHALLENGE_SET_SCHEMA_VERSION,
    GREEDY_REFERENCE_RECOVERY_BUDGET_CEILING,
    MAX_EXACT_VERSION_SPACE_SIZE,
    QUERY_POLICY_CEILING_SCHEMA_VERSION,
    ChallengeCandidateClassV2,
    ChallengeDPLayerV2,
    ChallengeQueryV2Error,
    MinimumChallengeSetV2,
    PolicyBudgetComparisonV2,
    PolicyPathStepV2,
    PolicySummaryV2,
    QueryChoiceV2,
    QueryPolicyCeilingReportV2,
    build_minimum_challenge_set_v2,
    build_query_policy_ceiling_report_v2,
    minimum_challenge_set_v2_from_obj,
    parse_minimum_challenge_set_v2,
    parse_query_policy_ceiling_report_v2,
    query_policy_ceiling_report_v2_from_obj,
    select_common_untouched_panel_v2,
    serialize_minimum_challenge_set_v2,
    serialize_query_policy_ceiling_report_v2,
    verify_minimum_challenge_set_v2,
    verify_query_policy_ceiling_report_v2,
)

__all__ = [
    "CHALLENGE_RESERVOIR_PANEL_COUNT",
    "CHALLENGE_RESERVOIR_PANEL_SIZE",
    "CHALLENGE_SET_SCHEMA_VERSION",
    "GREEDY_REFERENCE_RECOVERY_BUDGET_CEILING",
    "MAX_EXACT_VERSION_SPACE_SIZE",
    "QUERY_POLICY_CEILING_SCHEMA_VERSION",
    "ChallengeCandidateClassV2",
    "ChallengeDPLayerV2",
    "ChallengeQueryV2Error",
    "MinimumChallengeSetV2",
    "PolicyBudgetComparisonV2",
    "PolicyPathStepV2",
    "PolicySummaryV2",
    "QueryChoiceV2",
    "QueryPolicyCeilingReportV2",
    "build_minimum_challenge_set_v2",
    "build_query_policy_ceiling_report_v2",
    "minimum_challenge_set_v2_from_obj",
    "parse_minimum_challenge_set_v2",
    "parse_query_policy_ceiling_report_v2",
    "query_policy_ceiling_report_v2_from_obj",
    "select_common_untouched_panel_v2",
    "serialize_minimum_challenge_set_v2",
    "serialize_query_policy_ceiling_report_v2",
    "verify_minimum_challenge_set_v2",
    "verify_query_policy_ceiling_report_v2",
]
