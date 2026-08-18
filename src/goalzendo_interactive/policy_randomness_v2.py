"""Stateless, replayable random draws for G03 on-policy action sampling.

Sampling must not depend on the order in which episode rollouts happen to be
scheduled across workers.  This module derives one immutable turn seed from
the registered run seed, hidden-episode digest, rollout and turn indices, and
the frozen policy-state digest.  Each action-token draw is then addressed by
its token index rather than consumed from a mutable RNG.

The resulting 53-bit words map exactly into the open interval ``(0, 1)`` and
can drive an inverse-CDF choice over sorted legal token IDs.  This module does
not inspect logits, load a model, or authorize a weight update.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ._json import json_digest

POLICY_RANDOMNESS_SCHEMA_VERSION = 1
POLICY_RANDOMNESS_CONTRACT_ID = "goalzendo-stateless-policy-randomness-v1"
ROLLOUTS_PER_EPISODE = 8
MAXIMUM_TURNS_PER_ROLLOUT = 8
_U53_DENOMINATOR = 1 << 53

_TURN_SEED_DOMAIN = "goalzendo-interactive-policy-turn-seed-v1"
_TOKEN_DRAW_WORD_DOMAIN = "goalzendo-interactive-policy-token-draw-word-v1"
_TOKEN_DRAW_RECORD_DOMAIN = "goalzendo-interactive-policy-token-draw-record-v1"


class PolicyRandomnessError(ValueError):
    """Raised when a policy draw cannot be bound or reconstructed exactly."""


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bounded_index(value: object, *, name: str, upper: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyRandomnessError(f"{name} must be a non-negative integer")
    if upper is not None and value >= upper:
        raise PolicyRandomnessError(f"{name} must be smaller than {upper}")
    return value


@dataclass(frozen=True, slots=True)
class PolicyTurnSeed:
    """All immutable coordinates defining one policy decision's draw stream."""

    run_seed: int
    episode_digest: str
    rollout_index: int
    turn_index: int
    policy_state_digest: str
    schema_version: int = POLICY_RANDOMNESS_SCHEMA_VERSION
    contract_id: str = POLICY_RANDOMNESS_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_RANDOMNESS_SCHEMA_VERSION:
            raise PolicyRandomnessError("unexpected policy-randomness schema version")
        if self.contract_id != POLICY_RANDOMNESS_CONTRACT_ID:
            raise PolicyRandomnessError("unexpected policy-randomness contract id")
        if (
            isinstance(self.run_seed, bool)
            or not isinstance(self.run_seed, int)
            or not 0 <= self.run_seed < 2**63
        ):
            raise PolicyRandomnessError("run_seed must be an integer in [0, 2^63)")
        if not _is_sha256(self.episode_digest):
            raise PolicyRandomnessError("episode_digest must be a lowercase SHA-256")
        if not _is_sha256(self.policy_state_digest):
            raise PolicyRandomnessError("policy_state_digest must be a lowercase SHA-256")
        _bounded_index(
            self.rollout_index,
            name="rollout_index",
            upper=ROLLOUTS_PER_EPISODE,
        )
        _bounded_index(
            self.turn_index,
            name="turn_index",
            upper=MAXIMUM_TURNS_PER_ROLLOUT,
        )

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "run_seed": self.run_seed,
            "episode_digest": self.episode_digest,
            "rollout_index": self.rollout_index,
            "turn_index": self.turn_index,
            "policy_state_digest": self.policy_state_digest,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_TURN_SEED_DOMAIN)

    def draw(self, action_token_index: int) -> PolicyTokenDraw:
        """Return the independently addressed draw for one action token."""

        return PolicyTokenDraw.from_turn_seed(self, action_token_index)


def policy_turn_seed_from_obj(value: object) -> PolicyTurnSeed:
    expected = {
        "schema_version",
        "contract_id",
        "run_seed",
        "episode_digest",
        "rollout_index",
        "turn_index",
        "policy_state_digest",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise PolicyRandomnessError("policy-turn seed has noncanonical fields")
    result = PolicyTurnSeed(
        schema_version=value["schema_version"],
        contract_id=value["contract_id"],
        run_seed=value["run_seed"],
        episode_digest=value["episode_digest"],
        rollout_index=value["rollout_index"],
        turn_index=value["turn_index"],
        policy_state_digest=value["policy_state_digest"],
    )
    if result.as_obj() != value:
        raise PolicyRandomnessError("policy-turn seed is valid but not canonical")
    return result


@dataclass(frozen=True, slots=True)
class PolicyTokenDraw:
    """One exact 53-bit draw bound to a turn and action-token index."""

    turn_seed_digest: str
    action_token_index: int
    word_u53: int
    schema_version: int = POLICY_RANDOMNESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_RANDOMNESS_SCHEMA_VERSION:
            raise PolicyRandomnessError("unexpected token-draw schema version")
        if not _is_sha256(self.turn_seed_digest):
            raise PolicyRandomnessError("turn_seed_digest must be a lowercase SHA-256")
        _bounded_index(self.action_token_index, name="action_token_index")
        if (
            isinstance(self.word_u53, bool)
            or not isinstance(self.word_u53, int)
            or not 0 <= self.word_u53 < _U53_DENOMINATOR
        ):
            raise PolicyRandomnessError("word_u53 must be an integer in [0, 2^53)")

    @classmethod
    def from_turn_seed(
        cls,
        turn_seed: PolicyTurnSeed,
        action_token_index: int,
    ) -> PolicyTokenDraw:
        if type(turn_seed) is not PolicyTurnSeed:
            raise TypeError("turn_seed must be a PolicyTurnSeed")
        token_index = _bounded_index(action_token_index, name="action_token_index")
        digest = json_digest(
            {
                "turn_seed_digest": turn_seed.digest,
                "action_token_index": token_index,
            },
            domain=_TOKEN_DRAW_WORD_DOMAIN,
        )
        # Fourteen hexadecimal digits contain 56 bits.  Dropping the final
        # three bits leaves a uniform 53-bit integer without platform RNGs.
        word = int(digest[:14], 16) >> 3
        return cls(
            turn_seed_digest=turn_seed.digest,
            action_token_index=token_index,
            word_u53=word,
        )

    @property
    def open_unit_interval(self) -> Fraction:
        """Map the word to an exact rational strictly between zero and one."""

        return Fraction(2 * self.word_u53 + 1, 2 * _U53_DENOMINATOR)

    def as_obj(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "turn_seed_digest": self.turn_seed_digest,
            "action_token_index": self.action_token_index,
            "word_u53": self.word_u53,
        }

    @property
    def digest(self) -> str:
        return json_digest(self.as_obj(), domain=_TOKEN_DRAW_RECORD_DOMAIN)


def policy_token_draw_from_obj(value: object) -> PolicyTokenDraw:
    expected = {
        "schema_version",
        "turn_seed_digest",
        "action_token_index",
        "word_u53",
    }
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise PolicyRandomnessError("policy-token draw has noncanonical fields")
    result = PolicyTokenDraw(
        schema_version=value["schema_version"],
        turn_seed_digest=value["turn_seed_digest"],
        action_token_index=value["action_token_index"],
        word_u53=value["word_u53"],
    )
    if result.as_obj() != value:
        raise PolicyRandomnessError("policy-token draw is valid but not canonical")
    return result


def verify_policy_token_draw(
    draw: PolicyTokenDraw,
    turn_seed: PolicyTurnSeed,
) -> PolicyTokenDraw:
    if type(draw) is not PolicyTokenDraw or type(turn_seed) is not PolicyTurnSeed:
        raise TypeError("draw and turn_seed must be typed policy-randomness records")
    expected = turn_seed.draw(draw.action_token_index)
    if expected != draw:
        raise PolicyRandomnessError("policy-token draw does not rederive from its turn seed")
    return draw


def policy_randomness_manifest() -> dict[str, object]:
    return {
        "schema_version": POLICY_RANDOMNESS_SCHEMA_VERSION,
        "contract_id": POLICY_RANDOMNESS_CONTRACT_ID,
        "rollouts_per_episode": ROLLOUTS_PER_EPISODE,
        "maximum_turns_per_rollout": MAXIMUM_TURNS_PER_ROLLOUT,
        "seed_coordinates": [
            "run_seed",
            "episode_digest",
            "rollout_index",
            "turn_index",
            "policy_state_digest",
        ],
        "draw_address": "turn_seed_digest plus action_token_index",
        "draw_width_bits": 53,
        "unit_interval_mapping": "(word_u53 + 0.5) / 2^53",
        "stateful_rng_present": False,
        "order_dependent": False,
        "live_model_authorization": False,
        "weight_update_authorization": False,
    }
