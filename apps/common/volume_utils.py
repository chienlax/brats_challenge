"""Shared utilities for working with BraTS volumes."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import nibabel as nib
import numpy as np
from matplotlib import colors, patches

MODALITY_LABELS: Mapping[str, str] = {
    "t1c": "T1-weighted (post-contrast)",
    "t1n": "T1-weighted (pre-contrast)",
    "t2f": "T2 FLAIR",
    "t2w": "T2-weighted",
}

SEGMENTATION_LABELS: Mapping[int, Tuple[str, str]] = {
    0: ("Background", "#00000000"),
    1: ("Enhancing Tumor", "#1f77b4"),
    2: ("Non-Enhancing Tumor", "#d62728"),
    3: ("Peritumoral Edema / SNFH", "#2ca02c"),
    4: ("Resection Cavity", "#f1c40f"),
}

AXIS_TO_INDEX: Mapping[str, int] = {"sagittal": 0, "coronal": 1, "axial": 2}


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


def segmentation_legend_handles() -> List[patches.Patch]:
    handles: List[patches.Patch] = []
    for label, (name, hex_color) in SEGMENTATION_LABELS.items():
        if label == 0:
            continue
        handles.append(patches.Patch(color=hex_color, label=f"{label}: {name}"))
    return handles


def prepare_volumes(case_dir: Path, case_prefix: str | None = None) -> Dict[str, np.ndarray]:
    if case_prefix is None:
        case_prefix = infer_case_prefix(case_dir)
    volumes: Dict[str, np.ndarray] = {}
    for modality in MODALITY_LABELS:
        path = case_dir / f"{case_prefix}-{modality}.nii.gz"
        if path.exists():
            volumes[modality] = normalize_intensity(load_volume(path))
    if not volumes:
        raise FileNotFoundError(f"No imaging modalities were found in {case_dir}.")
    return volumes


def load_segmentation(case_dir: Path, case_prefix: str | None = None) -> np.ndarray | None:
    if case_prefix is None:
        case_prefix = infer_case_prefix(case_dir)
    seg_path = case_dir / f"{case_prefix}-seg.nii.gz"
    if not seg_path.exists():
        return None
    data = load_volume(seg_path)
    return np.rint(data).astype(np.int16)


def case_metadata(case_dir: Path) -> MutableMapping[str, object]:
    case_prefix = infer_case_prefix(case_dir)
    volumes = prepare_volumes(case_dir, case_prefix)
    example = next(iter(volumes.values()))
    meta: MutableMapping[str, object] = {
        "case_id": case_prefix,
        "shape": example.shape,
        "modalities": tuple(volumes.keys()),
    }
    seg = load_segmentation(case_dir, case_prefix)
    meta["has_segmentation"] = seg is not None
    return meta


def discover_cases(data_roots: Sequence[Path]) -> List[Path]:
    cases: List[Path] = []
    for root in data_roots:
        for path in sorted(root.glob("BraTS-GLI-*")):
            if any(path.glob("*-seg.nii.gz")):
                cases.append(path)
    return cases


@lru_cache(maxsize=16)
def cached_volumes(case_dir: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray | None]:
    case_prefix = infer_case_prefix(case_dir)
    volumes = prepare_volumes(case_dir, case_prefix)
    segmentation = load_segmentation(case_dir, case_prefix)
    return volumes, segmentation
