"""Dash callbacks wiring for the volume inspector."""

from __future__ import annotations

import datetime as dt
import io
from typing import List, Mapping

import imageio.v2 as iio
from dash import Dash, Input, Output, State, no_update, dcc

from apps.common.volume_utils import AXIS_TO_INDEX, prepare_slice

from .data import VolumeRepository
from .rendering import blank_figure, compose_rgba_slice, create_orthogonal_figure, create_slice_figure


def _ensure_jsonable(metadata: Mapping[str, object]) -> Mapping[str, object]:
    shape = metadata.get("shape", (1, 1, 1))
    modalities = metadata.get("modalities", [])
    return {
        **metadata,
        "shape": list(shape),
        "modalities": list(modalities),
    }


def register_callbacks(app: Dash, repo: VolumeRepository) -> None:
    @app.callback(
        Output("case-select", "options"),
        Output("case-select", "value"),
        Input("dataset-select", "value"),
        State("case-select", "value"),
    )
    def update_case_dropdown(dataset_value: str | None, current_case: str | None):
        if not dataset_value:
            return [], None
        options = repo.case_options(dataset_value)
        if not options:
            return [], None
        option_values = {opt["value"] for opt in options}
        if current_case in option_values:
            return options, current_case
        return options, options[0]["value"]

    @app.callback(
        Output("case-metadata", "data"),
        Output("modality-select", "options"),
        Output("modality-select", "value"),
        Input("case-select", "value"),
        State("modality-select", "value"),
    )
    def update_case_metadata(case_key: str | None, current_modality: str | None):
        if not case_key:
            return None, [], None
        metadata = _ensure_jsonable(repo.case_metadata(case_key))
        modalities: List[str] = metadata.get("modalities", [])  # type: ignore[assignment]
        options = [{"label": m.upper(), "value": m} for m in modalities]
        selected = current_modality if current_modality in modalities else (modalities[0] if modalities else None)
        return metadata, options, selected

    @app.callback(
        Output("slice-slider", "max"),
        Output("slice-slider", "value"),
        Output("slice-slider", "marks"),
        Input("axis-radio", "value"),
        Input("case-metadata", "data"),
    )
    def update_slider(axis_name: str, metadata: Mapping[str, object] | None):
        if not metadata:
            return 0, 0, {0: "0"}
        shape = metadata.get("shape", [1, 1, 1])
        axis_index = AXIS_TO_INDEX[axis_name]
        axis_length = int(shape[axis_index])
        max_value = max(axis_length - 1, 0)
        midpoint = max_value // 2 if max_value > 0 else 0
        marks = {0: "0"}
        if axis_length > 1:
            marks = {0: "0", midpoint: str(midpoint), max_value: str(max_value)}
        return max_value, midpoint, marks

    @app.callback(
        Output("main-figure", "figure"),
        Output("orth-figure", "figure"),
        Output("slice-info", "children"),
        Input("case-select", "value"),
        Input("case-metadata", "data"),
        Input("axis-radio", "value"),
        Input("slice-slider", "value"),
        Input("modality-select", "value"),
        Input("overlay-toggle", "value"),
    )
    def update_figures(
        case_key: str | None,
        metadata: Mapping[str, object] | None,
        axis_name: str,
        slice_index: int,
        modality: str | None,
        overlay_toggle: List[str] | None,
    ):
        if not case_key or not metadata or not modality:
            return blank_figure(), blank_figure(), ""

        overlay_enabled = bool(overlay_toggle and "overlay" in overlay_toggle)
        volumes, segmentation = repo.volumes(case_key)
        if modality not in volumes:
            modality = next(iter(volumes))
        axis_index = AXIS_TO_INDEX[axis_name]
        axis_length = int(metadata.get("shape", [1, 1, 1])[axis_index])
        clamped_index = max(0, min(slice_index or 0, axis_length - 1))

        main_fig = create_slice_figure(volumes, segmentation, modality, axis_name, clamped_index, overlay_enabled)

        volume = volumes[modality]
        indices = {
            "sagittal": int(metadata.get("shape", [1, 1, 1])[0]) // 2,
            "coronal": int(metadata.get("shape", [1, 1, 1])[1]) // 2,
            "axial": int(metadata.get("shape", [1, 1, 1])[2]) // 2,
        }
        indices[axis_name] = clamped_index
        orth_fig = create_orthogonal_figure(volume, segmentation, indices, overlay_enabled)

        case_id = metadata.get("case_id", "Unknown")
        info = f"Case {case_id} · Modality {modality.upper()} · {axis_name.capitalize()} slice {clamped_index}/{axis_length - 1}"
        if segmentation is None:
            info += " · Segmentation unavailable"
        elif overlay_enabled:
            info += " · Overlay on"
        else:
            info += " · Overlay off"

        return main_fig, orth_fig, info

    @app.callback(
        Output("export-download", "data"),
        Input("export-button", "n_clicks"),
        State("case-select", "value"),
        State("case-metadata", "data"),
        State("axis-radio", "value"),
        State("slice-slider", "value"),
        State("modality-select", "value"),
        State("overlay-toggle", "value"),
        prevent_initial_call=True,
    )
    def export_slice(
        n_clicks: int,
        case_key: str | None,
        metadata: Mapping[str, object] | None,
        axis_name: str,
        slice_index: int,
        modality: str | None,
        overlay_toggle: List[str] | None,
    ):
        if not case_key or not metadata or not modality:
            return no_update

        overlay_enabled = bool(overlay_toggle and "overlay" in overlay_toggle)
        volumes, segmentation = repo.volumes(case_key)
        if modality not in volumes:
            modality = next(iter(volumes))

        slice_array = prepare_slice(volumes[modality], axis_name, slice_index)
        seg_slice = prepare_slice(segmentation, axis_name, slice_index) if overlay_enabled and segmentation is not None else None
        rgba = compose_rgba_slice(slice_array, seg_slice, overlay_enabled and segmentation is not None)

        buffer = io.BytesIO()
        iio.imwrite(buffer, rgba, format="png")
        buffer.seek(0)

        case_id = metadata.get("case_id", "case")
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{case_id}_{modality}_{axis_name}_{slice_index}_{timestamp}.png"
        return dcc.send_bytes(buffer.getvalue, filename)


__all__ = ["register_callbacks"]
