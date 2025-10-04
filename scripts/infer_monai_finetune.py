"""Run validation inference for the fine-tuned MONAI model and export predictions.

This script mirrors the preprocessing used during training to prepare the
validation set, loads a saved checkpoint, performs sliding-window inference,
optionally applies light post-processing, and writes the resulting segmentations
as NIfTI files. The filenames follow nnU-Net conventions so they can be fed
straight into ``scripts/run_full_evaluation.py`` for metric computation.

Example (PowerShell):

python scripts/infer_monai_finetune.py `
  --data-root training_data training_data_additional `
  --split-json nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json `
  --fold 0 `
  --checkpoint outputs/monai_ft/checkpoints/stage2_best.pt `
  --output-dir outputs/monai_ft/predictions/fold0 `
  --bundle-name brats_mri_segmentation

To run evaluation immediately after inference, supply the ground-truth directory
(for example ``training_data``) and the script will invoke
``scripts/run_full_evaluation.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

try:
    from huggingface_hub import snapshot_download
    from monai.bundle import load as bundle_load
    from monai.data import Dataset, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        AsDiscrete,
        Compose,
        ConcatItemsd,
        CropForegroundd,
        DeleteItemsd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        Orientationd,
        ScaleIntensityRangePercentilesd,
        Spacingd,
        SpatialPadd,
    )
except ImportError as exc:
    raise ImportError(
        "MONAI and huggingface-hub are required. Install them with `pip install monai huggingface-hub`."
    ) from exc

try:
    from monai.transforms import SaveImaged  # type: ignore
except ImportError:  # pragma: no cover - SaveImaged expected to exist
    SaveImaged = None

MODALITIES: Sequence[str] = ("t1c", "t1n", "t2w", "t2f")
NUM_OUTPUT_CHANNELS = 5


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference helper for MONAI fine-tuned model.")
    parser.add_argument(
        "--data-root",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing BraTS case folders.",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        required=True,
        help="Path to nnU-Net splits_final.json (to reuse fold membership).",
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold index to run inference on (default: 0).")
    parser.add_argument(
        "--bundle-name",
        type=str,
        default="brats_mri_segmentation",
        help="Name of the MONAI model bundle on Hugging Face Hub (default: brats_mri_segmentation).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where model bundles are stored (default: ./artifacts).",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        help="Optional path to a local bundle directory, overriding the default in --artifacts-dir.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint produced by train_monai_finetune.py (stage1 or stage2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where NIfTI predictions will be written.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        help="Target voxel spacing used during training (default: 1.0 1.0 1.0).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=(224, 224, 144),
        help="Spatial size used for SpatialPadd (default: 224 224 144).",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=(224, 224, 144),
        help="Sliding window ROI size (default: 224 224 144).",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1,
        help="Sliding window batch size (default: 1).",
    )
    parser.add_argument(
        "--sw-overlap",
        type=float,
        default=0.5,
        help="Sliding window overlap fraction (default: 0.5).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision during inference.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Optional directory containing ground-truth segmentations for evaluation.",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        help="Directory where evaluation reports should be written (requires --ground-truth).",
    )
    parser.add_argument(
        "--run-evaluation",
        action="store_true",
        help="Invoke scripts/run_full_evaluation.py after inference (requires --ground-truth).",
    )
    parser.add_argument(
        "--encoder-prefix",
        type=str,
        nargs="*",
        default=("encoder", "backbone", "swinViT", "downsample"),
        help="Parameter name prefixes that identify encoder blocks (used when loading checkpoints).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for determinism (default: 2024).",
    )
    return parser.parse_args(argv)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_splits(split_json: Path, fold: int) -> Sequence[str]:
    with split_json.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    if not isinstance(splits, list) or len(splits) <= fold:
        raise ValueError(f"Split index {fold} not found in {split_json}.")
    return [str(case) for case in splits[fold]["val"]]


def build_case_index(data_roots: Sequence[Path]) -> Mapping[str, Path]:
    index: dict[str, Path] = {}
    for root in data_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Data root {root} does not exist or is not a directory.")
        for case_dir in sorted(root.glob("BraTS-GLI-*")):
            if case_dir.is_dir():
                index[case_dir.name] = case_dir
    if not index:
        raise RuntimeError("No BraTS cases found in the provided roots.")
    return index


def case_record(case_dir: Path) -> Mapping[str, str]:
    record = {"case_id": case_dir.name, "label": str(case_dir / f"{case_dir.name}-seg.nii.gz")}
    if not Path(record["label"]).exists():
        raise FileNotFoundError(f"Missing segmentation for case {case_dir.name}")
    for modality in MODALITIES:
        modality_path = case_dir / f"{case_dir.name}-{modality}.nii.gz"
        if not modality_path.exists():
            raise FileNotFoundError(f"Missing modality {modality} for case {case_dir.name}")
        record[modality] = str(modality_path)
    return record


def build_dataset(case_ids: Sequence[str], index: Mapping[str, Path], spacing: Sequence[float], patch_size: Sequence[int]) -> Dataset:
    records = [case_record(index[case_id]) for case_id in case_ids]
    transforms = Compose(
        [
            LoadImaged(keys=list(MODALITIES) + ["label"], ensure_channel_first=False),
            EnsureChannelFirstd(keys=list(MODALITIES)),
            ConcatItemsd(keys=list(MODALITIES), name="image", dim=0),
            DeleteItemsd(keys=list(MODALITIES)),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=0.5,
                upper=99.5,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="label"),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant"),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
        ]
    )
    return Dataset(data=records, transform=transforms)


def collate_fn(batch: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    meta = {key: [item[key] for item in batch] for key in batch[0] if key not in {"image", "label"}}
    return {"image": images, "label": labels, **meta}


def initialise_model(
    bundle_name: str, artifacts_dir: Path, device: torch.device, bundle_dir: Path | None = None
) -> nn.Module:
    """Initialises model, downloading from Hugging Face if not present."""
    local_bundle_path = bundle_dir
    if local_bundle_path is None:
        local_bundle_path = artifacts_dir / bundle_name
        if not local_bundle_path.exists():
            repo_id = f"MONAI/{bundle_name}"
            print(f"Bundle not found at {local_bundle_path}. Downloading from Hugging Face: {repo_id}")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=repo_id, local_dir=local_bundle_path, repo_type="model")

    print(f"Loading MONAI bundle from: {local_bundle_path}")
    components = bundle_load(
        name=bundle_name,
        bundle_dir=local_bundle_path,
        map_location=device,
    )

    network = components.get("network")
    if network is None:
        raise RuntimeError(f"Failed to load network from MONAI bundle '{bundle_name}'.")

    state_dict = components.get("state_dict") or components.get("model")
    if state_dict:
        # Load pre-trained weights, but not for the fine-tuning run
        network.load_state_dict(state_dict)

    network.to(device)
    return network


def replace_output_head(model: nn.Module, new_out_channels: int) -> nn.Module:
    def _replace(module: nn.Module) -> bool:
        for name, child in reversed(list(module.named_children())):
            if isinstance(child, nn.Conv3d):
                new_conv = nn.Conv3d(
                    in_channels=child.in_channels,
                    out_channels=new_out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                )
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
                setattr(module, name, new_conv)
                return True
            if _replace(child):
                return True
        return False

    if not _replace(model):
        raise RuntimeError("Could not replace final Conv3d layer; architecture unexpected.")
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model") or checkpoint
    model.load_state_dict(state_dict)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = _default_device()
    print(f"Using device: {device}")

    set_determinism(args.seed)

    val_ids = load_splits(args.split_json, args.fold)
    index = build_case_index(args.data_root)
    dataset = build_dataset(val_ids, index, args.spacing, args.patch_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    model = initialise_model(args.bundle_name, args.artifacts_dir, device, args.bundle_dir)
    model = replace_output_head(model, NUM_OUTPUT_CHANNELS)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    post_pred = AsDiscrete(argmax=True)
    saver = None
    if SaveImaged is not None:
        saver = SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=str(args.output_dir), output_postfix="", separate_folder=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            case_id = batch["case_id"][0]

            with autocast(enabled=args.amp):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=args.roi_size,
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=args.sw_overlap,
                    mode="gaussian",
                )
            preds = [post_pred(tensor) for tensor in decollate_batch(logits)]
            pred_tensor = preds[0].cpu().to(torch.int16)

            if saver is not None:
                batch_single = {"pred": pred_tensor, "image_meta_dict": batch["image_meta_dict"][0]}
                saver(batch_single)
            else:
                # Fallback: write using nibabel directly
                import nibabel as nib  # lazy import

                meta = batch["image_meta_dict"][0]
                affine = meta.get("affine")
                if affine is None:
                    affine = np.diag(list(meta.get("pixdim", (1, 1, 1))) + [1])
                nib.save(nib.Nifti1Image(pred_tensor.numpy().astype(np.int16), affine), args.output_dir / f"{case_id}.nii.gz")

            output_name = args.output_dir / f"{case_id}.nii.gz"
            if not output_name.exists():
                # SaveImaged writes to path stored in meta; fallback ensures naming
                import nibabel as nib  # lazy import inside loop is negligible for small set

                meta = batch["image_meta_dict"][0]
                affine = meta.get("affine")
                if affine is None:
                    affine = np.diag(list(meta.get("pixdim", (1, 1, 1))) + [1])
                nib.save(nib.Nifti1Image(pred_tensor.numpy().astype(np.int16), affine), output_name)

            print(f"Saved prediction for {case_id} -> {output_name}")

    if args.run_evaluation:
        if args.ground_truth is None:
            raise ValueError("--run-evaluation requires --ground-truth to be specified.")
        eval_dir = args.evaluation_output or (args.output_dir / "../reports")
        eval_dir = Path(eval_dir).resolve()
        eval_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python",
            "scripts/run_full_evaluation.py",
            str(args.ground_truth.resolve()),
            str(args.output_dir.resolve()),
            "--output-dir",
            str(eval_dir),
        ]
        print("Running evaluation:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("Inference complete.")


if __name__ == "__main__":  # pragma: no cover
    main()