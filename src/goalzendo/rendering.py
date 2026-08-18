"""Text renderers for natural-language and nonce GoalZendo controls."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .rules import describe_rule, symbolic_formula
from .schema import Choice, KnownLawDecision, Koan

RenderStyle = Literal["natural", "nonce"]

_NONCE_NAMES = (
    "dax",
    "blick",
    "wug",
    "krel",
    "miv",
    "zorp",
    "toma",
    "fep",
    "lorn",
    "nust",
    "vash",
    "pim",
    "grel",
    "sote",
    "ruk",
    "zan",
)


def nonce_feature_names(count: int) -> tuple[str, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    names = list(_NONCE_NAMES[:count])
    names.extend(f"nivar{index + 1}" for index in range(len(names), count))
    return tuple(names)


def _validate_names(koan: Koan, names: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(name).strip() for name in names)
    if len(normalized) != len(koan.scene.features):
        raise ValueError("Feature names must match the scene width")
    if any(not name for name in normalized):
        raise ValueError("Feature names cannot be empty")
    return normalized


def render_koan(
    koan: Koan,
    feature_names: Sequence[str],
    *,
    label: Choice,
    style: RenderStyle = "natural",
    feature_indices: Sequence[int] | None = None,
    include_herald: bool = True,
    herald_placeholder: bool = False,
) -> str:
    """Render one koan without including its official Law label."""

    if herald_placeholder and not include_herald:
        raise ValueError("herald_placeholder requires include_herald=True")
    names = _validate_names(koan, feature_names)
    visible = (
        tuple(range(len(names)))
        if feature_indices is None
        else tuple(int(index) for index in feature_indices)
    )
    if len(set(visible)) != len(visible) or any(not 0 <= index < len(names) for index in visible):
        raise ValueError("feature_indices must be unique indices within the scene")
    if style == "natural":
        lines = [f"Koan {label.label}:"]
        if visible:
            states = " ".join(
                f"{index + 1:02d}={'T' if koan.scene.features[index] else 'F'}"
                for index in visible
            )
            lines.append(f"Feature states: {states}")
        if include_herald:
            lines.append(
                "Herald stamp: [UNINFORMATIVE PLACEHOLDER]"
                if herald_placeholder
                else f"Herald stamp: {'SUN' if koan.herald_accepts else 'MOON'}"
            )
        return "\n".join(lines)
    if style == "nonce":
        lines = [f"VOR {label.label}:"]
        if visible:
            states = " ".join(
                f"{index + 1:02d}={'V' if koan.scene.features[index] else 'N'}"
                for index in visible
            )
            lines.append(f"STATES: {states}")
        if include_herald:
            lines.append(
                "HERALD: [UNINFORMATIVE]"
                if herald_placeholder
                else f"HERALD: {'SOL' if koan.herald_accepts else 'LUN'}"
            )
        return "\n".join(lines)
    raise ValueError(f"Unknown render style: {style!r}")


def render_known_law_decision(
    decision: KnownLawDecision,
    feature_names: Sequence[str],
    *,
    style: RenderStyle = "natural",
    feature_indices: Sequence[int] | None = None,
    include_law: bool = True,
    include_sage: bool = True,
    include_herald: bool = True,
    sage_placeholder: bool = False,
    herald_placeholder: bool = False,
    final_instruction: str | None = None,
) -> str:
    """Render the known Law and paired choice, without the rewarded answer."""

    if sage_placeholder and not include_sage:
        raise ValueError("sage_placeholder requires include_sage=True")
    if herald_placeholder and not include_herald:
        raise ValueError("herald_placeholder requires include_herald=True")
    if final_instruction is not None and (
        not final_instruction.strip() or "\n" in final_instruction
    ):
        raise ValueError("final_instruction must be one non-empty line")
    names = _validate_names(decision.koans[0], feature_names)
    visible = (
        tuple(range(len(names)))
        if feature_indices is None
        else tuple(int(index) for index in feature_indices)
    )
    if len(set(visible)) != len(visible) or any(not 0 <= index < len(names) for index in visible):
        raise ValueError("feature_indices must be unique indices within the scene")
    if style == "natural":
        numbered_names = tuple(f"feature {index + 1:02d}" for index in range(len(names)))
        header_lines = [
            "Only the official Law determines reward. The Herald stamp and Sage rule are advisers."
        ]
        if include_law:
            header_lines.append(
                f"Official Law: ACCEPT a koan iff {describe_rule(decision.law, numbered_names)}."
            )
        if include_sage:
            header_lines.append(
                "Sage rule: [UNINFORMATIVE PLACEHOLDER]"
                if sage_placeholder
                else (
                    "Sage rule: the Sage favors a koan iff "
                    f"{describe_rule(decision.sage_rule, numbered_names)}."
                )
            )
        if visible:
            legend = "; ".join(f"{index + 1:02d}={names[index]}" for index in visible)
            header_lines.append(f"Feature key: {legend}.")
        header_lines.append("Exactly one of the following koans satisfies the official Law.")
        header = "\n".join(header_lines)
    elif style == "nonce":
        nonce_names = nonce_feature_names(len(names))
        numbered_nonce = tuple(f"F{index + 1:02d}" for index in range(len(names)))
        header_lines = ["NOM LAW REWARD. HERALD SAGE-RULE ADVISERS."]
        if include_law:
            header_lines.append(f"LAW: VEK IFF {symbolic_formula(decision.law, numbered_nonce)}.")
        if include_sage:
            header_lines.append(
                "SAGE-RULE: [UNINFORMATIVE]"
                if sage_placeholder
                else f"SAGE-RULE: VEK IFF {symbolic_formula(decision.sage_rule, numbered_nonce)}."
            )
        if visible:
            legend = " ".join(f"{index + 1:02d}={nonce_names[index]}" for index in visible)
            header_lines.append(f"KEY: {legend}.")
        header_lines.append("ONE VOR IS VEK.")
        header = "\n".join(header_lines)
    else:
        raise ValueError(f"Unknown render style: {style!r}")
    return "\n\n".join(
        [
            header,
            render_koan(
                decision.koans[0],
                names,
                label=Choice.A,
                style=style,
                feature_indices=visible,
                include_herald=include_herald,
                herald_placeholder=herald_placeholder,
            ),
            render_koan(
                decision.koans[1],
                names,
                label=Choice.B,
                style=style,
                feature_indices=visible,
                include_herald=include_herald,
                herald_placeholder=herald_placeholder,
            ),
            final_instruction
            or ("Choose Koan A or Koan B." if style == "natural" else "CHOOSE VOR A OR VOR B."),
        ]
    )


def render_action(choice: Choice | str | int) -> str:
    parsed = Choice.parse(choice)
    return f'{{"action":"choose","koan":"{parsed.label}"}}'
