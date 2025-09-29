"""Rendering helpers for Dash figures."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.common.volume_utils import AXIS_TO_INDEX, SEGMENTATION_LABELS, prepare_slice


def _hex_to_rgba(color: str) -> np.ndarray:
    color = color.lstrip("#")
    if len(color) == 8:
        r, g, b, a = (int(color[i : i + 2], 16) for i in range(0, 8, 2))
    elif len(color) == 6:
        r, g, b = (int(color[i : i + 2], 16) for i in range(0, 6, 2))
        a = 255
    else:
        raise ValueError(f"Unexpected color format: {color}")
    return np.array([r, g, b, a], dtype=np.uint8)


def _segmentation_to_rgba(labels: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
    for label, (_, color) in SEGMENTATION_LABELS.items():
        mask = labels == label
        if not np.any(mask):
            continue
        rgba[mask] = _hex_to_rgba(color)
    return rgba


def compose_rgba_slice(volume: np.ndarray, segmentation: np.ndarray | None, overlay: bool) -> np.ndarray:
    scaled = np.clip(volume * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(scaled[..., None], 3, axis=-1).astype(np.float32)
    alpha = np.full(scaled.shape, 255, dtype=np.uint8)

    if overlay and segmentation is not None:
        seg_rgba = _segmentation_to_rgba(segmentation)
        mask = seg_rgba[..., 3] > 0
        if np.any(mask):
            blend_alpha = 0.45
            rgb[mask] = (1 - blend_alpha) * rgb[mask] + blend_alpha * seg_rgba[mask, :3].astype(np.float32)

    rgba = np.concatenate([rgb.astype(np.uint8), alpha[..., None]], axis=-1)
    return rgba


def create_slice_figure(
    volumes: Mapping[str, np.ndarray],
    segmentation: np.ndarray | None,
    modality: str,
    axis_name: str,
    index: int,
    overlay: bool,
) -> go.Figure:
    volume = volumes[modality]
    slice_2d = prepare_slice(volume, axis_name, index)
    seg_slice = prepare_slice(segmentation, axis_name, index) if overlay and segmentation is not None else None
    rgba = compose_rgba_slice(slice_2d, seg_slice, overlay and segmentation is not None)

    fig = go.Figure(go.Image(z=rgba))
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    fig.update_layout(title=f"{axis_name.capitalize()} slice {index}")
    return fig


def create_orthogonal_figure(
    volume: np.ndarray,
    segmentation: np.ndarray | None,
    indices: Mapping[str, int],
    overlay: bool,
) -> go.Figure:
    axes_order = ("sagittal", "coronal", "axial")
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{ax.capitalize()} {indices[ax]}" for ax in axes_order])

    for col, axis_name in enumerate(axes_order, start=1):
        idx = indices[axis_name]
        slice_2d = prepare_slice(volume, axis_name, idx)
        seg_slice = prepare_slice(segmentation, axis_name, idx) if overlay and segmentation is not None else None
        rgba = compose_rgba_slice(slice_2d, seg_slice, overlay and segmentation is not None)
        fig.add_trace(go.Image(z=rgba), row=1, col=col)

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        height=420,
    )
    for axis in fig.layout:
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            fig.layout[axis].update(showticklabels=False)
        if axis.startswith("yaxis"):
            fig.layout[axis].update(autorange="reversed")
    return fig


def blank_figure(message: str = "Select a case to view slices.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig
