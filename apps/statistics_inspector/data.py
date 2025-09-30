"""Data management for statistics inspector.

Handles dataset discovery, statistics computation, and caching.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes

MODALITIES = ("t1c", "t1n", "t2f", "t2w")
LABEL_MAP = {
    0: "Background",
    1: "Enhancing Tumor",
    2: "Non-Enhancing Tumor",
    3: "Peritumoral Edema / SNFH",
    4: "Resection Cavity",
}


@dataclass
class CaseSummary:
    """Summary statistics for a single case."""
    case_id: str
    modalities_present: List[str]
    shape: Tuple[int, int, int]
    voxel_spacing: Tuple[float, float, float]
    orientation: Tuple[str, str, str]
    intensity_stats: Mapping[str, Mapping[str, float]]
    modality_dtypes: Mapping[str, str]
    segmentation_labels: Mapping[int, int]


class StatisticsRepository:
    """Manages dataset roots and statistics computation."""

    def __init__(self, data_roots: List[Path] | None = None):
        """Initialize with optional custom data roots."""
        if data_roots is None:
            # Default roots relative to repository root
            repo_root = Path(__file__).resolve().parents[2]
            data_roots = [
                repo_root / "training_data",
                repo_root / "training_data_additional",
                repo_root / "validation_data",
            ]
        self.data_roots = [p.expanduser().resolve() for p in data_roots if p.exists()]
        if not self.data_roots:
            raise FileNotFoundError("No valid dataset roots found.")
        
        # Cache directory
        repo_root = Path(__file__).resolve().parents[2]
        self.cache_dir = repo_root / "outputs" / "statistics_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_datasets(self) -> List[str]:
        """Return list of available dataset names."""
        return [root.name for root in self.data_roots]

    def get_cases_for_dataset(self, dataset_name: str) -> List[str]:
        """Return list of case IDs for a given dataset."""
        dataset_root = next((root for root in self.data_roots if root.name == dataset_name), None)
        if dataset_root is None:
            return []
        case_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        return [p.name for p in case_dirs]

    def get_case_directory(self, dataset_name: str, case_id: str) -> Path:
        """Get the full path to a case directory."""
        dataset_root = next((root for root in self.data_roots if root.name == dataset_name), None)
        if dataset_root is None:
            raise FileNotFoundError(f"Dataset {dataset_name} not found.")
        case_dir = dataset_root / case_id
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory {case_dir} does not exist.")
        return case_dir

    def get_dataset_cache_path(self, dataset_name: str) -> Path:
        """Get the cache file path for dataset statistics."""
        return self.cache_dir / f"{dataset_name}_stats.json"

    def has_cached_dataset_stats(self, dataset_name: str) -> bool:
        """Check if cached statistics exist for a dataset."""
        return self.get_dataset_cache_path(dataset_name).exists()

    def load_cached_dataset_stats(self, dataset_name: str) -> Dict:
        """Load cached dataset statistics."""
        cache_path = self.get_dataset_cache_path(dataset_name)
        if not cache_path.exists():
            raise FileNotFoundError(f"No cached statistics for {dataset_name}")
        return json.loads(cache_path.read_text())

    def save_dataset_stats_cache(self, dataset_name: str, stats: Dict) -> None:
        """Save dataset statistics to cache."""
        cache_path = self.get_dataset_cache_path(dataset_name)
        cache_path.write_text(json.dumps(stats, indent=2))

    def compute_dataset_statistics(self, dataset_name: str, max_cases: int | None = None) -> Dict:
        """Compute aggregate statistics for an entire dataset."""
        dataset_root = next((root for root in self.data_roots if root.name == dataset_name), None)
        if dataset_root is None:
            raise FileNotFoundError(f"Dataset {dataset_name} not found.")

        case_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        if max_cases is not None:
            case_dirs = case_dirs[:max_cases]

        summaries: List[CaseSummary] = []
        for case_dir in case_dirs:
            try:
                summary = self._collect_case_summary(case_dir)
                summaries.append(summary)
            except Exception as e:
                print(f"Warning: Skipping {case_dir.name}: {e}")
                continue

        aggregate = self._aggregate_summaries(summaries)
        result = {
            "root": str(dataset_root),
            "dataset_name": dataset_name,
            "aggregate": self._convert_counters(aggregate),
            "num_cases_processed": len(summaries),
        }
        
        # Save to cache
        self.save_dataset_stats_cache(dataset_name, result)
        return result

    def compute_case_statistics(self, dataset_name: str, case_id: str, bins: int = 100, 
                                lo_percentile: float = 1.0, hi_percentile: float = 99.0) -> Dict:
        """Compute detailed statistics for a single case."""
        case_dir = self.get_case_directory(dataset_name, case_id)
        case_prefix = self._infer_case_prefix(case_dir)

        # Load and normalize modality volumes
        volumes: Dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            path = case_dir / f"{case_prefix}-{modality}.nii.gz"
            if path.exists():
                volume = self._get_volume(path)
                volumes[modality] = self._normalize(volume, lo_percentile, hi_percentile)

        if not volumes:
            raise FileNotFoundError(f"No modalities found for {case_id}")

        # Load segmentation
        seg_path = case_dir / f"{case_prefix}-seg.nii.gz"
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segmentation for {case_id}")
        seg = np.rint(self._get_volume(seg_path)).astype(np.int16)

        # Compute histogram data
        histogram_data = {}
        for modality, data in volumes.items():
            hist, bin_edges = np.histogram(data.flatten(), bins=bins, range=(0.0, 1.0))
            histogram_data[modality] = {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
            }

        # Compute label statistics
        label_stats = self._compute_label_stats(seg)

        return {
            "case_id": case_id,
            "dataset": dataset_name,
            "histogram_data": histogram_data,
            "label_stats": label_stats,
            "modalities": list(volumes.keys()),
        }

    # Private helper methods

    def _infer_case_prefix(self, case_dir: Path) -> str:
        """Infer case prefix from directory name."""
        return case_dir.name

    def _get_volume(self, path: Path) -> np.ndarray:
        """Load NIfTI volume as float32 array."""
        if not path.exists():
            raise FileNotFoundError(path)
        return nib.load(str(path)).get_fdata(dtype=np.float32)

    def _normalize(self, volume: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Normalize volume using percentile range."""
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

    def _compute_label_stats(self, seg: np.ndarray) -> Dict[str, float]:
        """Compute segmentation label statistics."""
        stats: Dict[str, float] = {}
        total = seg.size
        labels, counts = np.unique(seg, return_counts=True)
        label_map = dict(zip(labels.astype(int), counts.astype(int)))
        for label, name in LABEL_MAP.items():
            voxels = label_map.get(label, 0)
            stats[f"voxels_label_{label}"] = int(voxels)
            stats[f"pct_label_{label}"] = (float(voxels) / total) * 100.0
            stats[f"label_{label}_name"] = name
        return stats

    def _collect_case_summary(self, case_dir: Path) -> CaseSummary:
        """Collect summary statistics for a single case."""
        case_prefix = self._infer_case_prefix(case_dir)

        modality_stats: Dict[str, Mapping[str, float]] = {}
        modality_dtypes: Dict[str, str] = {}
        modalities_present: List[str] = []
        shape = None
        spacing = None
        orientation = None

        for modality in MODALITIES:
            path = case_dir / f"{case_prefix}-{modality}.nii.gz"
            if path.exists():
                img = nib.load(str(path))
                data = img.get_fdata(dtype=np.float32)
                modality_stats[modality] = self._compute_intensity_stats(data)
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
            raise FileNotFoundError(f"No modalities found in {case_dir}")

        seg_labels: Dict[int, int] = {}
        seg_path = case_dir / f"{case_prefix}-seg.nii.gz"
        if seg_path.exists():
            seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
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

    def _compute_intensity_stats(self, volume: np.ndarray) -> Dict[str, float]:
        """Compute intensity statistics for a volume."""
        finite = np.isfinite(volume)
        if not np.any(finite):
            return {"p01": float("nan"), "p50": float("nan"), "p99": float("nan"), "mean": float("nan")}
        data = volume[finite]
        p01, p50, p99 = np.percentile(data, [1, 50, 99])
        return {"p01": float(p01), "p50": float(p50), "p99": float(p99), "mean": float(np.mean(data))}

    def _aggregate_summaries(self, summaries: List[CaseSummary]) -> Dict[str, object]:
        """Aggregate statistics from multiple case summaries."""
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

    def _convert_counters(self, obj: object) -> object:
        """Recursively convert Counter objects to dicts for JSON serialization."""
        return convert_counters(obj)


def convert_counters(obj: object) -> object:
    """Recursively convert Counter objects to dicts for JSON serialization."""
    if isinstance(obj, Counter):
        return {str(k): int(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): convert_counters(v) for k, v in obj.items()}
    return obj
