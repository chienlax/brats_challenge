"""Dash layout for the volume inspector."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from dash import dcc, html

from apps.common.volume_utils import AXIS_TO_INDEX, SEGMENTATION_LABELS

from .config import DEFAULT_DISPLAY_MODE, DEFAULT_VIEW_MODE, DISPLAY_MODES, VIEW_MODES
from .data import VolumeRepository


DEFAULT_AXIS = "axial"


def _build_segmentation_legend() -> html.Div:
    entries: List[html.Div] = []
    for label, (name, hex_color) in SEGMENTATION_LABELS.items():
        if label == 0:
            continue
        entries.append(
            html.Div(
                className="legend-entry",
                children=[
                    html.Span(className="legend-swatch", style={"backgroundColor": hex_color}),
                    html.Span(f"{label}: {name}", className="legend-label"),
                ],
            )
        )
    return html.Div(className="legend-container", children=[html.H4("Segmentation legend"), *entries])


def _build_axis_marks(axis_size: int) -> Mapping[int, str]:
    if axis_size <= 1:
        return {0: "0"}
    mid = axis_size // 2
    return {0: "0", mid: str(mid), axis_size - 1: str(axis_size - 1)}


def build_layout(repo: VolumeRepository) -> html.Div:
    dataset_options = repo.dataset_options()
    default_root_value = repo.default_root()
    case_options = repo.case_options(default_root_value) if default_root_value else []
    default_case_value = repo.default_case_key(default_root_value)

    default_metadata: Dict[str, Any] | None = None
    default_modality_options: List[Dict[str, str]] = []
    default_modality_value: str | None = None
    slider_max = 0
    slider_value = 0
    slider_marks: Mapping[int, str] = {0: "0"}

    if default_case_value:
        raw_metadata = repo.case_metadata(default_case_value)
        default_metadata = {
            **raw_metadata,
            "shape": list(raw_metadata.get("shape", (1, 1, 1))),
            "modalities": list(raw_metadata.get("modalities", [])),
        }
        modalities = default_metadata.get("modalities", [])
        default_modality_options = [{"label": m.upper(), "value": m} for m in modalities]
        default_modality_value = modalities[0] if modalities else None
        shape = default_metadata.get("shape", (1, 1, 1))
        axis_index = AXIS_TO_INDEX[DEFAULT_AXIS]
        slider_max = max(int(shape[axis_index]) - 1, 0)
        slider_value = slider_max // 2 if slider_max > 0 else 0
        slider_marks = _build_axis_marks(slider_max + 1)

    quality_options = [{"label": cfg["label"], "value": key} for key, cfg in DISPLAY_MODES.items()]
    view_mode_options = [{"label": cfg["label"], "value": key} for key, cfg in VIEW_MODES.items()]

    controls = html.Div(
        className="sidebar",
        children=[
            html.H2("Volume Inspector"),
            dcc.Dropdown(
                id="dataset-select",
                options=dataset_options,
                value=default_root_value,
                placeholder="Select dataset",
                clearable=False,
            ),
            dcc.Dropdown(
                id="case-select",
                options=case_options,
                value=default_case_value,
                placeholder="Select case",
                clearable=False,
            ),
            dcc.Dropdown(
                id="modality-select",
                options=default_modality_options,
                value=default_modality_value,
                placeholder="Select modality",
                clearable=False,
            ),
            dcc.RadioItems(
                id="axis-radio",
                options=[{"label": label.capitalize(), "value": label} for label in AXIS_TO_INDEX],
                value=DEFAULT_AXIS,
                labelStyle={"display": "inline-block", "margin-right": "12px"},
            ),
            html.Div(
                className="control-group",
                children=[
                    html.Label("View mode"),
                    dcc.RadioItems(
                        id="view-mode",
                        options=view_mode_options,
                        value=DEFAULT_VIEW_MODE,
                        labelStyle={"display": "inline-block", "margin-right": "12px"},
                    ),
                ],
            ),
            html.Div(
                className="control-group",
                children=[
                    html.Label("Slice index"),
                    dcc.Slider(
                        id="slice-slider",
                        min=0,
                        max=slider_max,
                        value=slider_value,
                        marks=slider_marks,
                        tooltip={"always_visible": False, "placement": "bottom"},
                    ),
                ],
            ),
            dcc.Checklist(
                id="overlay-toggle",
                options=[{"label": "Show segmentation overlay", "value": "overlay"}],
                value=["overlay"],
                inputClassName="overlay-checkbox",
            ),
            html.Div(
                className="control-group",
                children=[
                    html.Label("Display quality"),
                    dcc.RadioItems(
                        id="display-quality",
                        options=[{"label": opt["label"], "value": opt["value"]} for opt in quality_options],
                        value=DEFAULT_DISPLAY_MODE,
                        labelStyle={"display": "inline-block", "margin-right": "12px"},
                    ),
                ],
            ),
            html.Button("Export current slice", id="export-button", n_clicks=0, className="export-button"),
            dcc.Download(id="export-download"),
        ],
    )

    main_view = html.Div(
        className="main-view",
        children=[
            dcc.Store(id="case-metadata", data=default_metadata),
            dcc.Loading(
                id="loading-main",
                type="default",
                children=dcc.Graph(id="main-figure", config={"displayModeBar": False}),
            ),
            dcc.Loading(
                id="loading-orth",
                type="cube",
                children=dcc.Graph(id="orth-figure", config={"displayModeBar": False}),
            ),
            html.Div(id="slice-info", className="slice-info"),
            _build_segmentation_legend(),
        ],
    )

    return html.Div(className="app-container", children=[controls, main_view])
