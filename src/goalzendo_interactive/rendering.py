"""Deterministic natural-language renderers for canonical G03 scenes."""

from __future__ import annotations

from typing import Literal, cast

from ._json import json_digest
from .schema import POSITIONS, Piece, Scene

RendererName = Literal[
    "train_compact",
    "train_positional",
    "train_tabletop",
    "train_inventory",
    "eval_reverse",
    "eval_ledger",
]

TRAIN_RENDERERS: tuple[RendererName, ...] = (
    "train_compact",
    "train_positional",
    "train_tabletop",
    "train_inventory",
)
EVAL_RENDERERS: tuple[RendererName, ...] = ("eval_reverse", "eval_ledger")
RENDERERS: tuple[RendererName, ...] = (*TRAIN_RENDERERS, *EVAL_RENDERERS)
RENDERER_SCHEMA_VERSION = 1


def _piece(piece: Piece) -> str:
    return f"a {piece.size} {piece.color} {piece.shape}"


def _slot(scene: Scene, position: str) -> str:
    piece = getattr(scene, position)
    return "empty" if piece is None else _piece(piece)


def render_scene(scene: Scene, renderer: RendererName) -> str:
    """Render every semantic field exactly once using one frozen family."""

    if type(scene) is not Scene:
        raise TypeError("render_scene requires a canonical Scene")
    if renderer not in RENDERERS:
        raise ValueError(f"unknown renderer: {renderer!r}")

    if renderer == "train_compact":
        clauses = [f"{position}: {_slot(scene, position)}" for position in POSITIONS]
        return "; ".join(clauses) + f"; placard: {scene.placard}."
    if renderer == "train_positional":
        clauses = [
            f"The {position} position is {_slot(scene, position)}" for position in POSITIONS
        ]
        return ". ".join(clauses) + f". The placard shows {scene.placard}."
    if renderer == "train_tabletop":
        occupied = [
            f"{_piece(piece)} on the {position}"
            for position in POSITIONS
            if (piece := getattr(scene, position)) is not None
        ]
        empty = [position for position in POSITIONS if getattr(scene, position) is None]
        occupied_text = ", ".join(occupied)
        empty_text = ", ".join(empty) if empty else "none"
        return (
            f"On the table: {occupied_text}. Empty positions: {empty_text}. "
            f"A {scene.placard} placard is displayed."
        )
    if renderer == "train_inventory":
        entries = [f"{position}={_slot(scene, position)}" for position in POSITIONS]
        return f"Display [{'; '.join(entries)}]. Its visible sign is {scene.placard}."
    if renderer == "eval_reverse":
        clauses = [
            f"on the {position}, {_slot(scene, position)}" for position in reversed(POSITIONS)
        ]
        return f"The sign reads {scene.placard}; " + "; ".join(clauses) + "."

    slot_words = [f"{position.upper()}({_slot(scene, position)})" for position in POSITIONS]
    return f"Placard={scene.placard.upper()} | " + " | ".join(slot_words)


def renderer_digest() -> str:
    """Digest the frozen renderer registry and representative outputs."""

    from .schema import scene_at

    probes = (0, 1, 6_857, 6_858, 13_715)
    return json_digest(
        {
            "schema_version": RENDERER_SCHEMA_VERSION,
            "renderers": [
                {
                    "name": renderer,
                    "probes": [render_scene(scene_at(index), renderer) for index in probes],
                }
                for renderer in RENDERERS
            ],
        },
        domain="goalzendo-interactive-renderers-v1",
    )


def parse_renderer(value: str) -> RendererName:
    if type(value) is not str or value not in RENDERERS:
        raise ValueError(f"unknown renderer: {value!r}")
    return cast(RendererName, value)
