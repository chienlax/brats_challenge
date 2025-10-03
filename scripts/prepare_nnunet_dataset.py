"""Utility to reorganize BraTS cases into nnU-Net v2 dataset structure."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

MODALITY_ORDER = [
    ("t2f", "T2 FLAIR"),
    ("t1n", "T1-weighted (pre-contrast)"),
    ("t1c", "T1-weighted (post-contrast)"),
    ("t2w", "T2-weighted"),
]

LABELS = {
    "background": 0,
    "enhancing_tumor": 1,
    "non_enhancing_tumor": 2,
    "peritumoral_edema_snf": 3,
    "resection_cavity": 4,
}


def discover_case_dirs(roots: Sequence[Path], *, require_segmentation: bool) -> Sequence[Path]:
    cases: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for case_dir in sorted(root.glob("BraTS-GLI-*")):
            if not case_dir.is_dir():
                continue
            has_seg = any(case_dir.glob("*-seg.nii.gz"))
            if require_segmentation and not has_seg:
                continue
            cases.append(case_dir)
    return cases


@dataclass
class LinkStrategy:
    destination: Path

    def place(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def write_dataset_json(dataset_root: Path, *, num_train: int, num_test: int) -> None:
    dataset_description = {
        "name": "BraTSPostTx",
        "description": "BraTS post-treatment cohort converted for nnU-Net v2",
        "reference": "https://www.med.upenn.edu/cbica/brats2023/data.html",
        "licence": "BraTS 2023 Challenge Terms of Use",
        "release": "2025-10-03",
        "tensorImageSize": "3D",
        "channel_names": {str(idx): name for idx, (_, name) in enumerate(MODALITY_ORDER)},
        "labels": LABELS,
        "numTraining": num_train,
        "file_ending": ".nii.gz",
    }

    if num_test:
        dataset_description["numTest"] = num_test

    dataset_json = dataset_root / "dataset.json"
    dataset_json.write_text(json.dumps(dataset_description, indent=2) + "\n")


def convert_cases(train_dirs: Sequence[Path], test_dirs: Sequence[Path], dataset_root: Path) -> None:
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    images_ts = dataset_root / "imagesTs"
    if test_dirs:
        images_ts.mkdir(parents=True, exist_ok=True)

    linker = LinkStrategy(destination=dataset_root)

    for case_dir in train_dirs:
        case_prefix = case_dir.name
        seg_src = case_dir / f"{case_prefix}-seg.nii.gz"
        if not seg_src.exists():
            raise FileNotFoundError(f"Segmentation file missing for {case_prefix}")

        # Copy / hardlink segmentation
        seg_dst = labels_tr / f"{case_prefix}.nii.gz"
        linker.place(seg_src, seg_dst)

        # Copy / hardlink modalities in canonical nnU-Net order
        for idx, (suffix, _human_name) in enumerate(MODALITY_ORDER):
            src = case_dir / f"{case_prefix}-{suffix}.nii.gz"
            if not src.exists():
                raise FileNotFoundError(f"Missing modality '{suffix}' for case {case_prefix}")
            dst = images_tr / f"{case_prefix}_{idx:04d}.nii.gz"
            linker.place(src, dst)

    for case_dir in test_dirs:
        case_prefix = case_dir.name
        for idx, (suffix, _human_name) in enumerate(MODALITY_ORDER):
            src = case_dir / f"{case_prefix}-{suffix}.nii.gz"
            if not src.exists():
                raise FileNotFoundError(f"Missing modality '{suffix}' for test case {case_prefix}")
            dst = images_ts / f"{case_prefix}_{idx:04d}.nii.gz"
            linker.place(src, dst)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare BraTS data for nnU-Net v2")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="DEPRECATED alias for --train-sources",
    )
    parser.add_argument(
        "--train-sources",
        nargs="*",
        default=None,
        help="Folders containing labeled training cases (default: training_data and training_data_additional)",
    )
    parser.add_argument(
        "--test-sources",
        nargs="*",
        default=None,
        help="Folders containing unlabeled test cases to populate imagesTs",
    )
    parser.add_argument(
        "--dataset-id",
        default="Dataset501_BraTSPostTx",
        help="Folder name for the nnU-Net dataset inside nnUNet_raw (default: Dataset501_BraTSPostTx)",
    )
    parser.add_argument(
        "--nnunet-raw",
        default="outputs/nnunet/nnUNet_raw",
        help="Path to nnUNet_raw directory (default: outputs/nnunet/nnUNet_raw)",
    )
    args = parser.parse_args(argv)

    if args.train_sources is not None:
        train_sources = args.train_sources
    elif args.sources is not None:
        train_sources = args.sources
    else:
        train_sources = ["training_data", "training_data_additional"]

    train_roots = [Path(path).resolve() for path in train_sources]
    train_dirs = discover_case_dirs(train_roots, require_segmentation=True)
    if not train_dirs:
        raise SystemExit("No labeled BraTS training cases were found. Check the --train-sources directories.")

    test_sources = args.test_sources or []
    test_roots = [Path(path).resolve() for path in test_sources]
    test_dirs = discover_case_dirs(test_roots, require_segmentation=False)

    dataset_root = Path(args.nnunet_raw).resolve() / args.dataset_id
    dataset_root.mkdir(parents=True, exist_ok=True)

    convert_cases(train_dirs, test_dirs, dataset_root)
    write_dataset_json(dataset_root, num_train=len(train_dirs), num_test=len(test_dirs))

    print(
        "Prepared",
        len(train_dirs),
        "train cases",
        f"and {len(test_dirs)} test cases" if test_dirs else "",
        f"in {dataset_root}",
    )


if __name__ == "__main__":
    main()
