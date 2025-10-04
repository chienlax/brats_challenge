"""Unit coverage for BraTS lesion-wise metrics script."""

from __future__ import annotations

import math
import json

import numpy as np
import pytest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib

from scripts.compute_brats_lesion_metrics import (
	LesionMatch,
	build_label_groups,
	compute_case_metrics,
)
from scripts.run_full_evaluation import (
	compute_lesion_metrics_report,
	run_nnunet_cli,
)


def _default_groups():
	return build_label_groups([1, 2, 3, 4], omit_aggregates=False)


def test_perfect_match_single_lesion():
	gt = np.zeros((6, 6, 6), dtype=np.int16)
	gt[1:3, 1:3, 1:3] = 1
	pred = gt.copy()
	spacing = (1.0, 1.0, 1.0)

	dice_scores, hd95_scores, lesions = compute_case_metrics("case", gt, pred, spacing, _default_groups())

	assert math.isclose(dice_scores["ET"], 1.0, rel_tol=1e-6)
	assert math.isclose(hd95_scores["ET"], 0.0, abs_tol=1e-6)

	assert len(lesions["ET"]) == 1
	lesion = lesions["ET"][0]
	assert isinstance(lesion, LesionMatch)
	assert lesion.status == "matched"
	assert math.isclose(lesion.dice, 1.0, rel_tol=1e-6)
	assert math.isclose(lesion.hd95, 0.0, abs_tol=1e-6)


def test_missed_lesion_penalty_uses_volume_diagonal():
	gt = np.zeros((5, 8, 9), dtype=np.int16)
	gt[2:4, 3:5, 4:6] = 2
	pred = np.zeros_like(gt)
	spacing = (1.0, 0.8, 0.5)

	dice_scores, hd95_scores, lesions = compute_case_metrics("case", gt, pred, spacing, _default_groups())

	diag_mm = math.sqrt(
		(gt.shape[0] * spacing[0]) ** 2
		+ (gt.shape[1] * spacing[1]) ** 2
		+ (gt.shape[2] * spacing[2]) ** 2
	)

	assert math.isclose(dice_scores["NETC"], 0.0, abs_tol=1e-6)
	assert math.isclose(hd95_scores["NETC"], diag_mm, rel_tol=1e-6)

	missed = [lesion for lesion in lesions["NETC"] if lesion.status == "missed"]
	assert len(missed) == 1
	assert math.isclose(missed[0].dice, 0.0, abs_tol=1e-6)
	assert math.isclose(missed[0].hd95, diag_mm, rel_tol=1e-6)


def test_false_positive_lesion_penalty_matches_spec():
	gt = np.zeros((4, 4, 4), dtype=np.int16)
	pred = np.zeros_like(gt)
	pred[1:3, 1:3, 1:3] = 1
	spacing = (1.5, 1.5, 1.5)

	dice_scores, hd95_scores, lesions = compute_case_metrics("case", gt, pred, spacing, _default_groups())

	diag_mm = math.sqrt(
		(gt.shape[0] * spacing[0]) ** 2
		+ (gt.shape[1] * spacing[1]) ** 2
		+ (gt.shape[2] * spacing[2]) ** 2
	)

	assert math.isclose(dice_scores["ET"], 0.0, abs_tol=1e-6)
	assert math.isclose(hd95_scores["ET"], diag_mm, rel_tol=1e-6)

	spurious = [lesion for lesion in lesions["ET"] if lesion.status == "spurious"]
	assert len(spurious) == 1
	assert math.isclose(spurious[0].dice, 0.0, abs_tol=1e-6)
	assert math.isclose(spurious[0].hd95, diag_mm, rel_tol=1e-6)


def test_empty_masks_return_perfect_scores():
	gt = np.zeros((4, 4, 4), dtype=np.int16)
	pred = np.zeros_like(gt)
	spacing = (1.0, 1.0, 1.0)

	dice_scores, hd95_scores, lesions = compute_case_metrics("case", gt, pred, spacing, _default_groups())

	assert math.isclose(dice_scores["RC"], 1.0, abs_tol=1e-6)
	assert math.isclose(hd95_scores["RC"], 0.0, abs_tol=1e-6)
	assert lesions["RC"] == []


def _write_segmentation(path: Path, data: np.ndarray) -> None:
	affine = np.eye(4, dtype=np.float32)
	nib.save(nib.Nifti1Image(data.astype(np.int16), affine), str(path))


def test_compute_lesion_metrics_report_on_disk(tmp_path):
	gt_dir = tmp_path / "gt"
	pred_dir = tmp_path / "pred"
	gt_dir.mkdir()
	pred_dir.mkdir()

	case = "BraTS-GLI-00001-000"
	gt_volume = np.zeros((4, 4, 4), dtype=np.int16)
	gt_volume[1:3, 1:3, 1:3] = 1
	pred_volume = gt_volume.copy()

	gt_path = gt_dir / f"{case}-seg.nii.gz"
	pred_path = pred_dir / f"{case}.nii.gz"
	_write_segmentation(gt_path, gt_volume)
	_write_segmentation(pred_path, pred_volume)

	output_path = tmp_path / "lesion_metrics.json"
	report = compute_lesion_metrics_report(
		gt_dir,
		pred_dir,
		labels=[1, 2, 3, 4],
		omit_aggregates=False,
		output_path=output_path,
		pretty=True,
	)

	assert output_path.exists()
	summary = report["summary"]["LesionWise_Dice"]
	assert math.isclose(summary["ET"]["mean"], 1.0, abs_tol=1e-6)
	assert "WT" in summary  # Aggregates included


def test_run_nnunet_cli_missing_command_raises(tmp_path):
	with patch("scripts.run_full_evaluation.shutil.which", return_value=None):
		with pytest.raises(RuntimeError):
			run_nnunet_cli("nnUNetv2_evaluate_simple", tmp_path, tmp_path, [1], tmp_path / "metrics.json")


def test_run_nnunet_cli_success(tmp_path):
	metrics_path = tmp_path / "metrics.json"

	def _fake_run(cmdline, check, capture_output, text):  # noqa: ANN001
		metrics_path.write_text(json.dumps({"dice": {"foreground": 0.9}}), encoding="utf-8")
		return SimpleNamespace(stdout="done", stderr="")

	with patch("scripts.run_full_evaluation.shutil.which", return_value="nnUNetv2_evaluate_simple"), patch(
		"scripts.run_full_evaluation.subprocess.run", side_effect=_fake_run
	):
		metrics = run_nnunet_cli(
			"nnUNetv2_evaluate_simple",
			Path("/tmp/gt"),
			Path("/tmp/pred"),
			[1, 2],
			metrics_path,
		)

	assert metrics_path.exists()
	assert metrics["dice"]["foreground"] == 0.9
