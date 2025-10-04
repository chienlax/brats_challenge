"""Compute BraTS 2024 lesion-wise metrics for a set of segmentations.

This script compares a folder of predicted segmentations against ground-truth
segmentations and computes the BraTS lesion-wise Dice and 95% Hausdorff
distance for the canonical tumor sub-regions:

* Enhancing Tissue (ET / label 1)
* Non-enhancing Tumor Core (NETC / label 2)
* Surrounding Non-enhancing FLAIR Hyperintensity (SNFH / label 3)
* Resection Cavity (RC / label 4)
* Tumor Core (TC = ET ∪ NETC)
* Whole Tumor (WT = ET ∪ NETC ∪ SNFH)

The implementation mirrors the official BraTS 2024 “lesion-wise” definition by
operating on connected components (26-connectivity) of each sub-region. Matched
lesions contribute their individual Dice and HD95 values, while missed or false
positive lesions are penalised with Dice = 0 and HD95 equal to the maximum
possible spatial distance within the volume.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


LABEL_GROUPS: Mapping[str, Sequence[int]] = {
	"ET": (1,),
	"NETC": (2,),
	"SNFH": (3,),
	"RC": (4,),
	"TC": (1, 2),
	"WT": (1, 2, 3),
}


@dataclass(slots=True)
class LesionMatch:
	"""Stores per-lesion metrics for reporting."""

	status: str
	dice: float
	hd95: float
	gt_voxels: int
	pred_voxels: int


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def case_id_from_filename(path: Path) -> str:
	name = path.name
	if name.endswith(".nii.gz"):
		name = name[: -len(".nii.gz")]
	for suffix in ("-seg", "_seg"):
		if name.endswith(suffix):
			name = name[: -len(suffix)]
			break
	return name


def find_prediction_file(pred_dir: Path, case_id: str) -> Path:
	candidates = [
		pred_dir / f"{case_id}.nii.gz",
		pred_dir / f"{case_id}-seg.nii.gz",
		pred_dir / f"{case_id}_seg.nii.gz",
		pred_dir / f"{case_id}_pred.nii.gz",
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(f"Could not locate prediction for case '{case_id}' in {pred_dir}.")


def load_segmentation(path: Path) -> tuple[np.ndarray, Tuple[float, float, float]]:
	img = nib.load(str(path))
	data = np.rint(img.get_fdata(dtype=np.float32)).astype(np.int16)
	if data.ndim != 3:
		raise ValueError(f"Expected 3D segmentation, got shape {data.shape} for {path}.")
	spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
	return data, spacing


def make_label_mask(volume: np.ndarray, labels: Sequence[int]) -> np.ndarray:
	return np.isin(volume, labels)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
	structure = np.ones((3, 3, 3), dtype=np.uint8)
	labeled, num = ndimage.label(mask, structure=structure)
	return labeled.astype(np.int32, copy=False), int(num)


def compute_overlap_matrix(
	gt_labeled: np.ndarray, pred_labeled: np.ndarray, gt_count: int, pred_count: int
) -> np.ndarray:
	flat_gt = gt_labeled.reshape(-1)
	flat_pred = pred_labeled.reshape(-1)
	valid = (flat_gt > 0) | (flat_pred > 0)
	flat_gt = flat_gt[valid]
	flat_pred = flat_pred[valid]
	overlap = np.zeros((gt_count + 1, pred_count + 1), dtype=np.int64)
	np.add.at(overlap, (flat_gt, flat_pred), 1)
	return overlap


def greedy_match_pairs(overlap: np.ndarray) -> List[Tuple[int, int, int]]:
	pairs: List[Tuple[int, int, int]] = []
	gt_count, pred_count = overlap.shape[0] - 1, overlap.shape[1] - 1
	# Collect candidate pairs with positive overlap
	candidates: List[Tuple[int, int, int]] = []
	for gt_id in range(1, gt_count + 1):
		pred_ids = np.nonzero(overlap[gt_id, 1:])[0] + 1
		for pred_id in pred_ids:
			candidates.append((int(overlap[gt_id, pred_id]), gt_id, pred_id))

	# Sort descending by intersection size to prioritise best matches.
	candidates.sort(key=lambda item: item[0], reverse=True)

	used_gt: set[int] = set()
	used_pred: set[int] = set()
	for intersection, gt_id, pred_id in candidates:
		if gt_id in used_gt or pred_id in used_pred:
			continue
		used_gt.add(gt_id)
		used_pred.add(pred_id)
		pairs.append((intersection, gt_id, pred_id))

	return pairs


def dice_coefficient(intersection: int, gt_size: int, pred_size: int) -> float:
	if gt_size + pred_size == 0:
		return 1.0
	return float(2.0 * intersection / (gt_size + pred_size))


def compute_surface_points(mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
	if not np.any(mask):
		return np.empty((0, 3), dtype=np.float32)
	structure = np.ones((3, 3, 3), dtype=bool)
	eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
	surface = mask & ~eroded
	coords = np.argwhere(surface)
	return coords.astype(np.float32) * np.asarray(spacing, dtype=np.float32)


def hausdorff95(mask_a: np.ndarray, mask_b: np.ndarray, spacing: Sequence[float]) -> float:
	surface_a = compute_surface_points(mask_a, spacing)
	surface_b = compute_surface_points(mask_b, spacing)
	if surface_a.size == 0 and surface_b.size == 0:
		return 0.0
	if surface_a.size == 0 or surface_b.size == 0:
		return math.inf

	tree_a = cKDTree(surface_a)
	tree_b = cKDTree(surface_b)

	dists_ab = tree_b.query(surface_a, k=1, workers=-1)[0]
	dists_ba = tree_a.query(surface_b, k=1, workers=-1)[0]
	all_dists = np.concatenate([dists_ab, dists_ba])
	return float(np.percentile(all_dists, 95))


def penalised_hd95(value: float, max_distance: float) -> float:
	if math.isinf(value) or np.isnan(value):
		return max_distance
	return value


def compute_case_metrics(
	case_id: str,
	gt_volume: np.ndarray,
	pred_volume: np.ndarray,
	spacing: Sequence[float],
	label_groups: Mapping[str, Sequence[int]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, List[LesionMatch]]]:
	if gt_volume.shape != pred_volume.shape:
		raise ValueError(f"Shape mismatch for case {case_id}: gt {gt_volume.shape} vs pred {pred_volume.shape}.")

	diag_mm = float(np.linalg.norm((np.array(gt_volume.shape) * np.asarray(spacing, dtype=np.float32))))

	dice_scores: Dict[str, float] = {}
	hd95_scores: Dict[str, float] = {}
	lesion_tables: Dict[str, List[LesionMatch]] = {}

	for label_name, labels in label_groups.items():
		gt_mask = make_label_mask(gt_volume, labels)
		pred_mask = make_label_mask(pred_volume, labels)

		gt_labeled, gt_count = label_components(gt_mask)
		pred_labeled, pred_count = label_components(pred_mask)

		gt_sizes = np.bincount(gt_labeled.reshape(-1), minlength=gt_count + 1)
		pred_sizes = np.bincount(pred_labeled.reshape(-1), minlength=pred_count + 1)

		overlap = compute_overlap_matrix(gt_labeled, pred_labeled, gt_count, pred_count)
		matches = greedy_match_pairs(overlap)

		lesions: List[LesionMatch] = []

		# Matched lesions
		for intersection, gt_idx, pred_idx in matches:
			gt_mask_comp = gt_labeled == gt_idx
			pred_mask_comp = pred_labeled == pred_idx
			dice_value = dice_coefficient(intersection, int(gt_sizes[gt_idx]), int(pred_sizes[pred_idx]))
			hd_value = penalised_hd95(hausdorff95(gt_mask_comp, pred_mask_comp, spacing), diag_mm)
			lesions.append(
				LesionMatch(
					status="matched",
					dice=dice_value,
					hd95=hd_value,
					gt_voxels=int(gt_sizes[gt_idx]),
					pred_voxels=int(pred_sizes[pred_idx]),
				)
			)

		matched_gt = {gt_idx for _, gt_idx, _ in matches}
		matched_pred = {pred_idx for _, _, pred_idx in matches}

		# False negatives: ground-truth lesions without a match
		for gt_idx in range(1, gt_count + 1):
			if gt_idx in matched_gt:
				continue
			lesions.append(
				LesionMatch(
					status="missed",
					dice=0.0,
					hd95=diag_mm,
					gt_voxels=int(gt_sizes[gt_idx]),
					pred_voxels=0,
				)
			)

		# False positives: predicted lesions without a match
		for pred_idx in range(1, pred_count + 1):
			if pred_idx in matched_pred:
				continue
			lesions.append(
				LesionMatch(
					status="spurious",
					dice=0.0,
					hd95=diag_mm,
					gt_voxels=0,
					pred_voxels=int(pred_sizes[pred_idx]),
				)
			)

		lesion_tables[label_name] = lesions

		if lesions:
			dice_scores[label_name] = float(np.mean([lesion.dice for lesion in lesions]))
			hd95_scores[label_name] = float(np.mean([lesion.hd95 for lesion in lesions]))
		else:
			# No lesions in either mask — treat as perfect detection.
			dice_scores[label_name] = 1.0
			hd95_scores[label_name] = 0.0

	return dice_scores, hd95_scores, lesion_tables


def summarise_metrics(per_case: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
	summary: Dict[str, Dict[str, float]] = {}
	if not per_case:
		return summary
	for label_name in next(iter(per_case.values())).keys():
		values = np.asarray([metrics[label_name] for metrics in per_case.values()], dtype=np.float32)
		summary[label_name] = {
			"mean": float(np.mean(values)),
			"median": float(np.median(values)),
			"min": float(np.min(values)),
			"max": float(np.max(values)),
			"std": float(np.std(values, ddof=0)),
		}
	return summary


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compute BraTS lesion-wise metrics.")
	parser.add_argument("ground_truth", type=Path, help="Directory containing ground-truth segmentations (NIfTI).")
	parser.add_argument("predictions", type=Path, help="Directory containing predicted segmentations (NIfTI).")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("outputs/brats_lesion_metrics.json"),
		help="Path to the JSON file to write (default: outputs/brats_lesion_metrics.json).",
	)
	parser.add_argument(
		"--labels",
		nargs="*",
		type=int,
		default=[1, 2, 3, 4],
		help="Foreground labels expected in the dataset (default: 1 2 3 4).",
	)
	parser.add_argument(
		"--omit-aggregates",
		action="store_true",
		help="Only report per-label metrics (skip TC/WT aggregates).",
	)
	parser.add_argument(
		"--pretty",
		action="store_true",
		help="Pretty-print the JSON output for readability.",
	)
	return parser.parse_args(argv)


def build_label_groups(labels: Sequence[int], omit_aggregates: bool) -> Mapping[str, Sequence[int]]:
	groups = {name: tuple(LABEL_GROUPS[name]) for name in ("ET", "NETC", "SNFH", "RC") if LABEL_GROUPS[name][0] in labels}
	if not omit_aggregates:
		groups["TC"] = tuple(LABEL_GROUPS["TC"])
		groups["WT"] = tuple(LABEL_GROUPS["WT"])
	return groups


def collect_cases(directory: Path) -> List[Path]:
	if not directory.is_dir():
		raise NotADirectoryError(directory)
	files = sorted([p for p in directory.glob("*.nii.gz") if not p.name.startswith(".")])
	if not files:
		raise FileNotFoundError(f"No NIfTI files found in {directory}.")
	return files


def main(argv: Iterable[str] | None = None) -> None:
	args = parse_args(argv)

	gt_files = collect_cases(args.ground_truth)
	label_groups = build_label_groups(args.labels, args.omit_aggregates)

	per_case_dice: Dict[str, Dict[str, float]] = {}
	per_case_hd95: Dict[str, Dict[str, float]] = {}
	lesion_details: Dict[str, Dict[str, List[MutableMapping[str, float | str | int]]]] = {}

	for gt_path in gt_files:
		case_id = case_id_from_filename(gt_path)
		pred_path = find_prediction_file(args.predictions, case_id)

		gt_volume, spacing = load_segmentation(gt_path)
		pred_volume, _ = load_segmentation(pred_path)

		dice_scores, hd95_scores, lesions = compute_case_metrics(case_id, gt_volume, pred_volume, spacing, label_groups)
		per_case_dice[case_id] = dice_scores
		per_case_hd95[case_id] = hd95_scores
		lesion_details[case_id] = {
			label: [
				{
					"status": lesion.status,
					"dice": lesion.dice,
					"hd95": lesion.hd95,
					"gt_voxels": lesion.gt_voxels,
					"pred_voxels": lesion.pred_voxels,
				}
				for lesion in lesion_list
			]
			for label, lesion_list in lesions.items()
		}

	output = {
		"cases": {
			case_id: {
				"LesionWise_Dice": per_case_dice[case_id],
				"LesionWise_HD95": per_case_hd95[case_id],
				"lesions": lesion_details[case_id],
			}
			for case_id in per_case_dice
		},
		"summary": {
			"LesionWise_Dice": summarise_metrics(per_case_dice),
			"LesionWise_HD95": summarise_metrics(per_case_hd95),
		},
	}

	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open("w", encoding="utf-8") as handle:
		if args.pretty:
			json.dump(output, handle, indent=2)
		else:
			json.dump(output, handle)

	print(f"Wrote lesion-wise metrics for {len(per_case_dice)} cases to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
	main()
