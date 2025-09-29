"""Utility for visualizing BraTS post-treatment glioma cases.

Example usage:
    python scripts/visualize_brats.py --case-dir training_data_additional/BraTS-GLI-02405-100 \
        --layout modalities --axis axial --fraction 0.55 --output outputs/BraTS-GLI-02405-100_axial.png

    python scripts/visualize_brats.py --case-dir training_data_additional/BraTS-GLI-02405-100 \
        --layout orthogonal --modality t1c --indices 90 110 70 --output outputs/BraTS-GLI-02405-100_orth.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import colors
from matplotlib import patches

MODALITY_LABELS: Mapping[str, str] = {
    "t1c": "T1-weighted (post-contrast)",
    "t1n": "T1-weighted (pre-contrast)",
    "t2f": "T2 FLAIR",
    "t2w": "T2-weighted",
}

# Post-treatment BraTS 2024 semantic labels (per challenge guidelines).
SEGMENTATION_LABELS: Mapping[int, Tuple[str, str]] = {
    0: ("Background", "#00000000"),
    1: ("Enhancing Tumor", "#1f77b4"),  # Blue
    2: ("Non-Enhancing Tumor", "#d62728"),  # Red
    3: ("Peritumoral Edema / SNFH", "#2ca02c"),  # Green
    4: ("Resection Cavity", "#f1c40f"),  # Yellow
}

AXIS_TO_INDEX = {"sagittal": 0, "coronal": 1, "axial": 2}


def infer_case_prefix(case_dir: Path) -> str:
    seg_files = list(case_dir.glob("*-seg.nii.gz"))
    if not seg_files:
        raise FileNotFoundError(f"Could not find a segmentation file (*-seg.nii.gz) in {case_dir}.")
    return seg_files[0].name.replace("-seg.nii.gz", "")


def load_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(volume)
    if not np.any(finite_mask):
        return np.zeros_like(volume, dtype=np.float32)

    cleaned = volume[finite_mask]
    lo, hi = np.percentile(cleaned, (1, 99))
    if math.isclose(lo, hi):
        lo, hi = cleaned.min(), cleaned.max()
    if math.isclose(lo, hi):
        return np.clip(volume, 0.0, 1.0)

    normalized = (volume - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)


def prepare_slice(volume: np.ndarray, axis_name: str, index: int) -> np.ndarray:
    axis = AXIS_TO_INDEX[axis_name]
    slicer: List[slice | int] = [slice(None)] * 3
    slicer[axis] = index
    data = volume[tuple(slicer)]

    if axis_name == "axial":
        rotated = np.rot90(data)
    elif axis_name == "coronal":
        rotated = np.rot90(np.flipud(data))
    else:  # sagittal
        rotated = np.rot90(np.flipud(data))
    return rotated


def pick_slice_index(volume: np.ndarray, axis_name: str, index: int | None, fraction: float) -> int:
    axis = AXIS_TO_INDEX[axis_name]
    max_index = volume.shape[axis]
    if index is not None:
        if not (0 <= index < max_index):
            raise ValueError(f"Slice index {index} is out of bounds for axis {axis_name} with size {max_index}.")
        return index
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return int(round(fraction * (max_index - 1)))


def build_segmentation_colormap() -> Tuple[colors.ListedColormap, colors.BoundaryNorm]:
    rgba = []
    for label in range(max(SEGMENTATION_LABELS) + 1):
        _, hex_color = SEGMENTATION_LABELS.get(label, (f"Label {label}", "#ffffff"))
        rgba.append(colors.to_rgba(hex_color))
    cmap = colors.ListedColormap(rgba)
    norm = colors.BoundaryNorm(np.arange(len(rgba) + 1) - 0.5, len(rgba))
    return cmap, norm


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
    handles = []
    for label, (name, hex_color) in SEGMENTATION_LABELS.items():
        if label == 0:
            continue
        handles.append(patches.Patch(color=hex_color, label=f"{label}: {name}"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)))


def prepare_volumes(case_dir: Path, case_prefix: str) -> Dict[str, np.ndarray]:
    volumes: Dict[str, np.ndarray] = {}
    for modality in MODALITY_LABELS:
        path = case_dir / f"{case_prefix}-{modality}.nii.gz"
        if path.exists():
            volumes[modality] = normalize_intensity(load_volume(path))
    if not volumes:
        raise FileNotFoundError(f"No imaging modalities were found in {case_dir}.")
    return volumes


def load_segmentation(case_dir: Path, case_prefix: str) -> np.ndarray | None:
    seg_path = case_dir / f"{case_prefix}-seg.nii.gz"
    if not seg_path.exists():
        return None
    data = load_volume(seg_path)
    return np.rint(data).astype(np.int16)


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
