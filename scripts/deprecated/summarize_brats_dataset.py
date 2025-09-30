"""Summarize structural properties of a BraTS post-treatment dataset.

This script scans a dataset directory (e.g. `training_data_additional`) and reports
aggregate statistics such as shapes, voxel spacing, modality availability, and
segmentation label counts.

Example usage:
    python scripts/summarize_brats_dataset.py --root training_data_additional --output stats.json
    python scripts/summarize_brats_dataset.py --root training_data_additional --sample-cases 3
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes

MODALITIES = ("t1c", "t1n", "t2f", "t2w")


@dataclass
class CaseSummary:
    case_id: str
    modalities_present: List[str]
    shape: Tuple[int, int, int]
    voxel_spacing: Tuple[float, float, float]
    orientation: Tuple[str, str, str]
    intensity_stats: Mapping[str, Mapping[str, float]]
    modality_dtypes: Mapping[str, str]
    segmentation_labels: Mapping[int, int]


def infer_case_prefix(case_dir: Path) -> str:
    seg_files = list(case_dir.glob("*-seg.nii.gz"))
    if seg_files:
        return seg_files[0].name.replace("-seg.nii.gz", "")
    # Fall back to first modality
    modality_file = next(case_dir.glob("*.nii.gz"), None)
    if modality_file is None:
        raise FileNotFoundError(f"No NIfTI files found in {case_dir}.")
    return "-".join(modality_file.name.split("-")[:-1])


def load_image(path: Path) -> nib.Nifti1Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return nib.load(str(path))


def compute_intensity_stats(volume: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(volume)
    if not np.any(finite):
        return {"p01": float("nan"), "p50": float("nan"), "p99": float("nan"), "mean": float("nan")}
    data = volume[finite]
    p01, p50, p99 = np.percentile(data, [1, 50, 99])
    return {"p01": float(p01), "p50": float(p50), "p99": float(p99), "mean": float(np.mean(data))}


def collect_case_summary(case_dir: Path) -> CaseSummary:
    case_prefix = infer_case_prefix(case_dir)

    modality_stats: Dict[str, Mapping[str, float]] = {}
    modality_dtypes: Dict[str, str] = {}
    modalities_present: List[str] = []
    shape = None
    spacing = None
    orientation = None

    for modality in MODALITIES:
        path = case_dir / f"{case_prefix}-{modality}.nii.gz"
        if path.exists():
            img = load_image(path)
            data = img.get_fdata(dtype=np.float32)
            modality_stats[modality] = compute_intensity_stats(data)
            modality_dtypes[modality] = str(img.get_data_dtype())
            modalities_present.append(modality)
            if shape is None:
                shape = tuple(int(v) for v in data.shape)
                pixdim = img.header.get_zooms()[:3]
                spacing = tuple(float(round(v, 3)) for v in pixdim)
                orientation = aff2axcodes(img.affine)
        else:
            modality_stats[modality] = {}
            modality_dtypes[modality] = "missing"

    if shape is None:
        raise FileNotFoundError(f"No modalities found in case directory {case_dir}.")

    seg_labels: Dict[int, int] = {}
    seg_path = case_dir / f"{case_prefix}-seg.nii.gz"
    if seg_path.exists():
        seg = load_image(seg_path).get_fdata(dtype=np.float32)
        labels, counts = np.unique(np.rint(seg).astype(np.int16), return_counts=True)
        seg_labels = {int(label): int(count) for label, count in zip(labels, counts)}

    return CaseSummary(
        case_id=case_prefix,
        modalities_present=sorted(modalities_present),
        shape=shape,
        voxel_spacing=spacing if spacing is not None else (float("nan"),) * 3,
        intensity_stats=modality_stats,
        segmentation_labels=seg_labels,
        modality_dtypes=modality_dtypes,
        orientation=orientation if orientation is not None else ("?", "?", "?"),
    )


def aggregate_summaries(summaries: Iterable[CaseSummary]) -> Dict[str, object]:
    summaries = list(summaries)
    shape_counts = Counter()
    spacing_counts = Counter()
    modality_presence = Counter()
    dtype_counts = defaultdict(Counter)
    orientation_counts = Counter()
    label_counts = Counter()
    intensity_agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for summary in summaries:
        shape_counts[str(summary.shape)] += 1
        spacing_counts[str(summary.voxel_spacing)] += 1
        for modality in summary.modalities_present:
            modality_presence[modality] += 1
        for modality, stats in summary.intensity_stats.items():
            if not stats:
                continue
            for key, value in stats.items():
                intensity_agg[modality][key].append(value)
        for modality, dtype in summary.modality_dtypes.items():
            dtype_counts[modality][dtype] += 1
        for label, count in summary.segmentation_labels.items():
            label_counts[label] += count
        orientation_counts[str(summary.orientation)] += 1

    intensity_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for modality, stats in intensity_agg.items():
        intensity_summary[modality] = {}
        for key, values in stats.items():
            array = np.asarray(values)
            intensity_summary[modality][key] = {
                "mean": float(np.nanmean(array)),
                "min": float(np.nanmin(array)),
                "max": float(np.nanmax(array)),
            }

    return {
        "num_cases": len(summaries),
        "shape_counts": shape_counts,
        "voxel_spacing_counts": spacing_counts,
        "modality_presence": modality_presence,
        "intensity_summary": intensity_summary,
    "label_counts": {int(label): int(count) for label, count in label_counts.items()},
        "dtype_counts": {modality: convert_counters(counter) for modality, counter in dtype_counts.items()},
        "orientation_counts": orientation_counts,
    }


def convert_counters(obj: object) -> object:
    if isinstance(obj, Counter):
        return {str(k): int(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): convert_counters(v) for k, v in obj.items()}
    return obj


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a BraTS dataset directory.")
    parser.add_argument("--root", type=Path, required=True, help="Path to a dataset folder (e.g. training_data_additional).")
    parser.add_argument("--max-cases", type=int, help="Optionally limit the number of cases to process.")
    parser.add_argument("--sample-cases", type=int, default=0, help="Include detailed summaries for the first N cases.")
    parser.add_argument("--output", type=Path, help="Write summary JSON to this path instead of stdout.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root {root} does not exist.")

    case_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if args.max_cases is not None:
        case_dirs = case_dirs[: args.max_cases]

    summaries: List[CaseSummary] = []
    for idx, case_dir in enumerate(case_dirs, start=1):
        summary = collect_case_summary(case_dir)
        summaries.append(summary)
        print(f"Processed {summary.case_id} ({idx}/{len(case_dirs)})")

    aggregate = aggregate_summaries(summaries)
    result = {
        "root": str(root),
        "aggregate": convert_counters(aggregate),
    }

    if args.sample_cases:
        result["sample_cases"] = [
            {
                "case_id": summary.case_id,
                "modalities_present": summary.modalities_present,
                "shape": summary.shape,
                "voxel_spacing": summary.voxel_spacing,
                "orientation": summary.orientation,
                "intensity_stats": summary.intensity_stats,
                "modality_dtypes": summary.modality_dtypes,
                "segmentation_labels": summary.segmentation_labels,
            }
            for summary in summaries[: args.sample_cases]
        ]

    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote summary to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
