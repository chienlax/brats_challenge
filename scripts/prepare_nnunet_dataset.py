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


def discover_case_dirs(roots: Sequence[Path]) -> Sequence[Path]:
    cases: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for case_dir in sorted(root.glob("BraTS-GLI-*")):
            if case_dir.is_dir() and any(case_dir.glob("*-seg.nii.gz")):
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


def write_dataset_json(dataset_root: Path, num_cases: int) -> None:
    dataset_description = {
        "name": "BraTSPostTx",
        "description": "BraTS post-treatment cohort converted for nnU-Net v2",
        "reference": "https://www.med.upenn.edu/cbica/brats2023/data.html",
        "licence": "BraTS 2023 Challenge Terms of Use",
        "release": "2025-10-03",
        "tensorImageSize": "3D",
        "channel_names": {str(idx): name for idx, (_, name) in enumerate(MODALITY_ORDER)},
        "labels": LABELS,
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
    }

    dataset_json = dataset_root / "dataset.json"
    dataset_json.write_text(json.dumps(dataset_description, indent=2) + "\n")


def convert_cases(case_dirs: Sequence[Path], dataset_root: Path) -> None:
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    linker = LinkStrategy(destination=dataset_root)

    for case_dir in case_dirs:
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


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare BraTS data for nnU-Net v2")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["training_data", "training_data_additional"],
        help="Root folders containing BraTS cases (default: training_data and training_data_additional)",
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

    source_roots = [Path(path).resolve() for path in args.sources]
    case_dirs = discover_case_dirs(source_roots)
    if not case_dirs:
        raise SystemExit("No BraTS cases were found. Check the source directories." )

    dataset_root = Path(args.nnunet_raw).resolve() / args.dataset_id
    dataset_root.mkdir(parents=True, exist_ok=True)

    convert_cases(case_dirs, dataset_root)
    write_dataset_json(dataset_root, num_cases=len(case_dirs))

    print(f"Prepared {len(case_dirs)} cases in {dataset_root}")


if __name__ == "__main__":
    main()
