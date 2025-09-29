"""Dash application factory for the volume inspector."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from dash import Dash

from .callbacks import register_callbacks
from .data import VolumeRepository
from .layout import build_layout


def create_dash_app(
    data_roots: Iterable[Path] | None = None,
    title: str = "BraTS Volume Inspector",
) -> Dash:
    roots: Sequence[Path]
    if data_roots is None:
        roots = (
            Path("training_data"),
            Path("training_data_additional"),
            Path("validation_data"),
        )
    else:
        roots = tuple(Path(path) for path in data_roots)

    repo = VolumeRepository(roots)
    app = Dash(__name__, title=title, suppress_callback_exceptions=False)
    app.layout = build_layout(repo)
    register_callbacks(app, repo)
    return app


__all__ = ["create_dash_app"]
