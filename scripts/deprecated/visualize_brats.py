"""Utility for visualizing BraTS post-treatment glioma cases.

Example usage:
    python scripts/visualize_brats.py --case-dir training_data_additional/BraTS-GLI-02405-100 \
        --layout modalities --axis axial --fraction 0.55 --output outputs/BraTS-GLI-02405-100_axial.png

    python scripts/visualize_brats.py --case-dir training_data_additional/BraTS-GLI-02405-100 \
        --layout orthogonal --modality t1c --indices 90 110 70 --output outputs/BraTS-GLI-02405-100_orth.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from apps.common.volume_utils import (
    AXIS_TO_INDEX,
    MODALITY_LABELS,
    SEGMENTATION_LABELS,
    build_segmentation_colormap,
    infer_case_prefix,
    load_segmentation,
    pick_slice_index,
    prepare_slice,
    prepare_volumes,
    segmentation_legend_handles,
)


def create_modalities_panel(
    volumes: Mapping[str, np.ndarray],
    segmentation: np.ndarray | None,
    axis_name: str,
    index: int,
    case_id: str,
    overlay: bool,
) -> plt.Figure:
    cmap_seg, norm_seg = build_segmentation_colormap()
    modalities = [m for m in MODALITY_LABELS if m in volumes]

    fig, axes = plt.subplots(1, len(modalities), figsize=(4.2 * len(modalities), 4.2), constrained_layout=True)
    if len(modalities) == 1:
        axes = [axes]

    for ax, modality in zip(axes, modalities):
        volume = volumes[modality]
        slice_2d = prepare_slice(volume, axis_name, index)
        ax.imshow(slice_2d, cmap="gray", interpolation="none")
        ax.set_title(f"{MODALITY_LABELS[modality]}\n{axis_name.capitalize()} slice {index}")
        ax.axis("off")

        if overlay and segmentation is not None:
            seg_slice = prepare_slice(segmentation, axis_name, index)
            mask = np.ma.masked_where(seg_slice < 0.5, seg_slice)
            ax.imshow(mask, cmap=cmap_seg, norm=norm_seg, alpha=0.45, interpolation="none")

    fig.suptitle(f"BraTS case {case_id}")
    add_segmentation_legend(fig)
    return fig


def create_orthogonal_panel(
    volume: np.ndarray,
    segmentation: np.ndarray | None,
    indices: Mapping[str, int],
    modality: str,
    case_id: str,
    overlay: bool,
) -> plt.Figure:
    cmap_seg, norm_seg = build_segmentation_colormap()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)

    for ax, axis_name in zip(axes, ("sagittal", "coronal", "axial")):
        idx = indices[axis_name]
        slice_2d = prepare_slice(volume, axis_name, idx)
        ax.imshow(slice_2d, cmap="gray", interpolation="none")
        ax.set_title(f"{axis_name.capitalize()} slice {idx}")
        ax.axis("off")
        if overlay and segmentation is not None:
            seg_slice = prepare_slice(segmentation, axis_name, idx)
            mask = np.ma.masked_where(seg_slice < 0.5, seg_slice)
            ax.imshow(mask, cmap=cmap_seg, norm=norm_seg, alpha=0.45, interpolation="none")

    fig.suptitle(f"BraTS case {case_id} — {MODALITY_LABELS.get(modality, modality)}")
    add_segmentation_legend(fig)
    return fig


def add_segmentation_legend(fig: plt.Figure) -> None:
    handles = segmentation_legend_handles()
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render BraTS modalities with segmentation overlays.")
    parser.add_argument("--case-dir", type=Path, required=True, help="Path to a BraTS case directory.")
    parser.add_argument(
        "--layout",
        choices=("modalities", "orthogonal"),
        default="modalities",
        help="Visualization layout. 'modalities' shows all modalities on a single plane; 'orthogonal' shows three planes for a single modality.",
    )
    parser.add_argument("--axis", choices=tuple(AXIS_TO_INDEX), default="axial", help="Axis for the modalities layout.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Relative slice location [0, 1] when --index is not provided (modalities layout).",
    )
    parser.add_argument("--index", type=int, help="Explicit slice index for the selected axis (modalities layout).")
    parser.add_argument(
        "--indices",
        type=int,
        nargs=3,
        metavar=("SAGITTAL", "CORONAL", "AXIAL"),
        help="Explicit slice indices for the orthogonal layout (sagittal, coronal, axial).",
    )
    parser.add_argument("--modality", choices=tuple(MODALITY_LABELS), default="t1c", help="Modality for the orthogonal layout.")
    parser.add_argument("--no-overlay", action="store_true", help="Disable segmentation overlay.")
    parser.add_argument("--output", type=Path, help="If provided, save the figure to this path.")
    parser.add_argument("--dpi", type=int, default=160, help="Resolution for saved figures.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively (ignored when running headless).")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    case_dir = args.case_dir.expanduser().resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory {case_dir} does not exist.")

    case_prefix = infer_case_prefix(case_dir)
    case_id = case_prefix
    volumes = prepare_volumes(case_dir, case_prefix)
    segmentation = load_segmentation(case_dir, case_prefix)

    if args.layout == "modalities":
        reference_volume = next(iter(volumes.values()))
        slice_index = pick_slice_index(reference_volume, args.axis, args.index, args.fraction)
        fig = create_modalities_panel(volumes, segmentation, args.axis, slice_index, case_id, overlay=not args.no_overlay)
    else:
        if args.indices:
            indices = {axis: idx for axis, idx in zip(("sagittal", "coronal", "axial"), args.indices)}
        else:
            vol = next(iter(volumes.values()))
            indices = {
                axis: pick_slice_index(vol, axis, None, args.fraction) for axis in ("sagittal", "coronal", "axial")
            }
        if args.modality not in volumes:
            raise ValueError(f"Requested modality '{args.modality}' was not found for case {case_id}.")
        fig = create_orthogonal_panel(volumes[args.modality], segmentation, indices, args.modality, case_id, overlay=not args.no_overlay)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {args.output}")
    if args.show or not args.output:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
