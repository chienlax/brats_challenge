"""Generate per-case histograms and statistics for BraTS post-treatment dataset.

Outputs intensity histograms (PNG) for each modality and a JSON summary of label volumes
per case. Designed for heavier exploratory analytics beyond the global summary script.

Usage examples:
    python scripts/generate_case_statistics.py --root training_data_additional --output-dir outputs/stats --max-cases 5
    python scripts/generate_case_statistics.py --case BraTS-GLI-02405-100 --output-dir outputs/stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

MODALITIES = ("t1c", "t1n", "t2f", "t2w")
LABEL_MAP = {
    0: "Background",
    1: "Enhancing Tumor",
    2: "Non-Enhancing Tumor",
    3: "Peritumoral Edema",
    4: "Resection Cavity",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-case histogram figures and statistics.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("training_data_additional"),
        help="Dataset root (contains case directories). Defaults to training_data_additional.",
    )
    parser.add_argument("--case", type=str, help="Process a single case id (overrides scanning the full root).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for figures and stats JSON.")
    parser.add_argument("--max-cases", type=int, help="Limit number of cases when scanning a root directory.")
    parser.add_argument("--bins", type=int, default=100, help="Histogram bin count.")
    parser.add_argument("--lower-percentile", type=float, default=1.0, help="Lower percentile for intensity normalization.")
    parser.add_argument("--upper-percentile", type=float, default=99.0, help="Upper percentile for intensity normalization.")
    return parser.parse_args(argv)


def infer_case_prefix(case_dir: Path) -> str:
    return case_dir.name


def list_case_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def load_image(path: Path) -> nib.Nifti1Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def get_volume(path: Path) -> np.ndarray:
    return load_image(path).get_fdata(dtype=np.float32)


def normalize(volume: np.ndarray, lo: float, hi: float) -> np.ndarray:
    finite = np.isfinite(volume)
    if not finite.any():
        return np.zeros_like(volume)
    data = volume[finite]
    v_lo, v_hi = np.percentile(data, (lo, hi))
    if np.isclose(v_lo, v_hi):
        v_lo, v_hi = data.min(), data.max()
    if np.isclose(v_lo, v_hi):
        return np.clip(volume, 0.0, 1.0)
    return np.clip((volume - v_lo) / (v_hi - v_lo), 0.0, 1.0)


def plot_histograms(case_id: str, volumes: Dict[str, np.ndarray], bins: int, output_dir: Path) -> None:
    cols = len(volumes)
    fig, axes = plt.subplots(1, cols, figsize=(4.5 * cols, 4))
    if cols == 1:
        axes = [axes]
    for ax, (modality, data) in zip(axes, volumes.items()):
        ax.hist(data.flatten(), bins=bins, range=(0.0, 1.0), color="steelblue")
        ax.set_title(f"{case_id} — {modality.upper()}")
        ax.set_xlabel("Normalized intensity")
        ax.set_ylabel("Voxel count")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / f"{case_id}_hist.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compute_label_stats(seg: np.ndarray) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    total = seg.size
    labels, counts = np.unique(seg, return_counts=True)
    label_map = dict(zip(labels.astype(int), counts.astype(int)))
    for label, name in LABEL_MAP.items():
        voxels = label_map.get(label, 0)
        stats[f"voxels_label_{label}"] = int(voxels)
        stats[f"pct_label_{label}"] = (float(voxels) / total) * 100.0
    return stats


def process_case(
    case_dir: Path,
    output_dir: Path,
    bins: int,
    lo: float,
    hi: float,
) -> Dict[str, float]:
    case_id = infer_case_prefix(case_dir)
    volumes: Dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        path = case_dir / f"{case_id}-{modality}.nii.gz"
        if not path.exists():
            continue
        volumes[modality] = normalize(get_volume(path), lo, hi)
    if not volumes:
        raise FileNotFoundError(f"No modalities found for {case_id}")
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segmentation for {case_id}")
    seg = np.rint(get_volume(seg_path)).astype(np.int16)

    case_output_dir = output_dir / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)
    plot_histograms(case_id, volumes, bins, case_output_dir)

    stats = {"case_id": case_id}
    stats.update(compute_label_stats(seg))
    stats["hist_path"] = str(case_output_dir / f"{case_id}_hist.png")
    return stats


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.case:
        case_dirs = [root / args.case]
    else:
        if not root.is_dir():
            raise FileNotFoundError(root)
        case_dirs = list_case_dirs(root)
        if args.max_cases is not None:
            case_dirs = case_dirs[: args.max_cases]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats: List[Dict[str, float]] = []

    for idx, case_dir in enumerate(case_dirs, start=1):
        if not case_dir.is_dir():
            raise FileNotFoundError(case_dir)
        print(f"Processing {case_dir.name} ({idx}/{len(case_dirs)})")
        stats.append(process_case(case_dir, args.output_dir, args.bins, args.lower_percentile, args.upper_percentile))

    output_json = args.output_dir / "case_statistics.json"
    output_json.write_text(json.dumps(stats, indent=2))
    print(f"Wrote stats to {output_json}")


if __name__ == "__main__":
    main()
