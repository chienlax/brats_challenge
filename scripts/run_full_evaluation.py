"""Orchestrate nnU-Net evaluation and BraTS lesion-wise metrics in one step.

This helper script is designed to be run from the nnU-Net inference workspace.
It accepts a ground-truth folder and a predictions folder, invokes
``nnUNetv2_evaluate_simple`` (unless disabled) and then computes the BraTS
post-treatment lesion-wise metrics using the repository's
``compute_brats_lesion_metrics`` module. All reports are written to the same
output directory so a single command yields the standard nnU-Net summary,
lesion-wise details, and a combined manifest tying everything together.

CLI:
python scripts/run_full_evaluation.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/monai_ft_nnunet_aligned/predictions/fold0 `
    --output-dir outputs/monai_ft_nnunet_aligned/reports/fold0 `
    --pretty

"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import nibabel as nib
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - allow import when used as module
	from compute_brats_lesion_metrics import (
		build_label_groups,
		case_id_from_filename,
		collect_cases,
		compute_case_metrics,
		find_prediction_file,
		load_segmentation,
		summarise_metrics,
	)
except ImportError:  # pragma: no cover
	from scripts.compute_brats_lesion_metrics import (  # type: ignore
		build_label_groups,
		case_id_from_filename,
		collect_cases,
		compute_case_metrics,
		find_prediction_file,
		load_segmentation,
		summarise_metrics,
	)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run nnU-Net and lesion-wise evaluation together.")
	parser.add_argument(
		"ground_truth",
		type=Path,
		help="Directory containing ground-truth segmentations (NIfTI).",
	)
	parser.add_argument("predictions", type=Path, help="Directory containing predicted segmentations (NIfTI).")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("outputs/nnunet/reports/metrics/full"),
		help="Directory to write all reports into (default: outputs/nnunet/reports/metrics/full).",
	)
	parser.add_argument(
		"--labels",
		nargs="*",
		type=int,
		default=[1, 2, 3, 4],
		help="Foreground labels to evaluate (default: 1 2 3 4).",
	)
	parser.add_argument(
		"--omit-aggregates",
		action="store_true",
		help="Skip Tumor Core (TC) and Whole Tumor (WT) aggregates in the lesion report.",
	)
	parser.add_argument(
		"--nnunet-cmd",
		default="nnUNetv2_evaluate_simple",
		help="Command used to run the nnU-Net evaluator (default: nnUNetv2_evaluate_simple).",
	)
	parser.add_argument(
		"--nnunet-output",
		type=Path,
		help="Optional explicit path for the nnU-Net metrics JSON. Overrides the default inside --output-dir.",
	)
	parser.add_argument(
		"--lesion-output",
		type=Path,
		help="Optional explicit path for the lesion metrics JSON. Overrides the default inside --output-dir.",
	)
	parser.add_argument(
		"--combined-output",
		type=Path,
		help="Optional explicit path for the merged JSON manifest. Overrides the default inside --output-dir.",
	)
	parser.add_argument(
		"--skip-nnunet",
		action="store_true",
		help="Do not invoke nnUNetv2_evaluate_simple (useful when the CLI is unavailable).",
	)
	parser.add_argument(
		"--pretty",
		action="store_true",
		help="Pretty-print JSON files with indentation for easier inspection.",
	)
	return parser.parse_args(argv)


def ensure_directory(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def run_nnunet_cli(
	command_name: str,
	ground_truth: Path,
	predictions: Path,
	labels: Sequence[int],
	output_path: Path,
) -> Mapping[str, object]:
	executable = shutil.which(command_name)
	if executable is None:
		raise RuntimeError(
			f"Could not locate '{command_name}' on PATH. Activate the nnU-Net environment or pass --skip-nnunet."
		)

	cmdline = [
		executable,
		str(ground_truth),
		str(predictions),
		"-o",
		str(output_path),
		"-l",
		*[str(label) for label in labels],
	]

	completed = subprocess.run(cmdline, check=True, capture_output=True, text=True)
	if completed.stdout:
		print(completed.stdout.strip())
	if completed.stderr:
		print(completed.stderr.strip())

	if not output_path.exists():
		raise FileNotFoundError(
			f"nnU-Net command completed but did not produce expected metrics file: {output_path}."
		)

	with output_path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def compute_lesion_metrics_report(
	ground_truth: Path,
	predictions: Path,
	labels: Sequence[int],
	omit_aggregates: bool,
	output_path: Path,
	pretty: bool,
) -> Mapping[str, object]:
	gt_files = collect_cases(ground_truth)
	label_groups = build_label_groups(labels, omit_aggregates)

	per_case_dice: MutableMapping[str, MutableMapping[str, float]] = {}
	per_case_hd95: MutableMapping[str, MutableMapping[str, float]] = {}
	lesion_details: MutableMapping[str, Mapping[str, list[Mapping[str, float | int | str]]]] = {}

	for gt_path in gt_files:
		case_id = case_id_from_filename(gt_path)
		pred_path = find_prediction_file(predictions, case_id)
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

	output: Mapping[str, object] = {
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

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		if pretty:
			json.dump(output, handle, indent=2)
		else:
			json.dump(output, handle)

	return output


def write_json(output_path: Path, payload: Mapping[str, object], pretty: bool) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		if pretty:
			json.dump(payload, handle, indent=2)
		else:
			json.dump(payload, handle)


def main(argv: Iterable[str] | None = None) -> None:
	args = parse_args(argv)

	ground_truth = args.ground_truth.resolve()
	predictions = args.predictions.resolve()
	if not ground_truth.exists():
		raise FileNotFoundError(f"Ground-truth directory not found: {ground_truth}")
	if not predictions.exists():
		raise FileNotFoundError(f"Predictions directory not found: {predictions}")

	output_dir = ensure_directory(args.output_dir.resolve())
	nnunet_output = (args.nnunet_output or (output_dir / "nnunet_metrics.json")).resolve()
	lesion_output = (args.lesion_output or (output_dir / "lesion_metrics.json")).resolve()
	combined_output = (args.combined_output or (output_dir / "full_metrics.json")).resolve()

	standard_metrics: Mapping[str, object] | None = None
	if not args.skip_nnunet:
		standard_metrics = run_nnunet_cli(
			args.nnunet_cmd,
			ground_truth,
			predictions,
			args.labels,
			nnunet_output,
		)
		if args.pretty and nnunet_output.exists():
			# Reformat in pretty mode for consistency.
			write_json(nnunet_output, standard_metrics, pretty=True)

	lesion_metrics = compute_lesion_metrics_report(
		ground_truth,
		predictions,
		args.labels,
		args.omit_aggregates,
		lesion_output,
		args.pretty,
	)

	combined_payload: Mapping[str, object] = {
		"ground_truth": str(ground_truth),
		"predictions": str(predictions),
		"labels": list(args.labels),
		"omit_aggregates": bool(args.omit_aggregates),
		"nnunet_command": None if args.skip_nnunet else args.nnunet_cmd,
		"nnunet_metrics": standard_metrics,
		"lesion_metrics": lesion_metrics,
	}

	write_json(combined_output, combined_payload, args.pretty)

	print("\nEvaluation complete.")
	if not args.skip_nnunet:
		print(f"  • nnU-Net metrics: {nnunet_output}")
	print(f"  • Lesion-wise metrics: {lesion_output}")
	print(f"  • Combined report: {combined_output}")


if __name__ == "__main__":  # pragma: no cover
	main()
