"""Run validation inference for the fine-tuned MONAI model and export predictions.

This script reads a nnU-Net dataset.json to identify the test set,
loads a saved checkpoint, performs sliding-window inference, and writes the
resulting segmentations as NIfTI files. Preprocessing (spacing, normalization, etc.)
is aligned with the revised, nnU-Net-style training script.

"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader


try:
    from huggingface_hub import snapshot_download
    from monai.bundle import load as bundle_load
    from monai.data import Dataset, MetaTensor, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        AsDiscrete,
        Compose,
        ConcatItemsd,
        DeleteItemsd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        Spacingd,
        SpatialPadd,
        ResampleToMatchd,
    )
except ImportError as exc:
    raise ImportError(
        "MONAI and huggingface-hub are required. Install them with `pip install monai huggingface-hub`."
    ) from exc

# Modality order aligned with nnU-Net V2 dataset.json for Dataset501_BraTSPostTx
# 0: T2 FLAIR, 1: T1-pre, 2: T1-post, 3: T2
MODALITIES: Sequence[str] = ("t2f", "t1n", "t1c", "t2w")
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
        help="One or more directories containing the BraTS case folders for the test set.",
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
        help="Path to the nnU-Net dataset.json file defining the test set.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="0",
        help='Fold index of the model to use, or "all" to run inference for all 5 folds. Used for formatting checkpoint/output paths. (default: "0").',
    )
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
        type=str,
        required=True,
        help="Path to the checkpoint. If using --fold all, use '{fold}' as a placeholder for the fold number.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for NIfTI predictions. If using --fold all, use '{fold}' as a placeholder.",
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
        default=(128, 128, 128),
        help="Spatial size used for SpatialPadd, aligned with nnU-Net plans (default: 128 128 128).",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        help="Sliding window ROI size, aligned with nnU-Net plans (default: 128 128 128).",
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


def load_test_ids_from_dataset_json(dataset_json: Path) -> Sequence[str]:
    """Loads case identifiers from the 'test' key of a dataset.json."""
    with dataset_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    test_files = data.get("test")
    if not test_files or not isinstance(test_files, list):
        raise ValueError(f"Could not find a valid 'test' list in {dataset_json}")

    case_ids = []
    for f in test_files:
        # Assumes nnU-Net v1/v2 style filenames like 'BraTS-XYZ-123_0000.nii.gz'
        case_id_match = re.match(r"(BraTS-GLI-\d{5}-\d{3})", Path(f).name)
        if case_id_match:
            case_id = case_id_match.group(1)
            if case_id not in case_ids:
                case_ids.append(case_id)

    if not case_ids:
        raise ValueError("Could not extract any case IDs from the test list in dataset.json.")

    return case_ids


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
    for modality in MODALITIES:
        modality_path = case_dir / f"{case_dir.name}-{modality}.nii.gz"
        if not modality_path.exists():
            raise FileNotFoundError(f"Missing modality {modality} for case {case_dir.name} at {modality_path}")
        record[modality] = str(modality_path)
    return record


def build_dataset(case_ids: Sequence[str], index: Mapping[str, Path], spacing: Sequence[float], patch_size: Sequence[int]) -> Dataset:
    records = [case_record(index[case_id]) for case_id in case_ids]

    image_keys = list(MODALITIES)
    # The keys that need to be loaded from disk
    keys_to_load = image_keys + ["label"]

    transforms = Compose(
        [
            # CORRECTED: Ensure 'label' is also loaded from its file path
            LoadImaged(keys=keys_to_load, ensure_channel_first=False),
            # CORRECTED: Ensure 'label' also gets a channel dimension
            EnsureChannelFirstd(keys=keys_to_load),
            ConcatItemsd(keys=image_keys, name="image", dim=0),
            DeleteItemsd(keys=image_keys),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            SpatialPadd(keys=["image"], spatial_size=patch_size, mode="constant"),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )
    return Dataset(data=records, transform=transforms)


def collate_fn(batch: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    meta = {key: [item[key] for item in batch] for key in batch[0] if key not in {"image", "label"}}
    if "label" in batch[0]:
        labels = torch.stack([item["label"] for item in batch], dim=0)
        return {"image": images, "label": labels, **meta}
    return {"image": images, **meta}


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
    components = bundle_load(name=bundle_name, bundle_dir=local_bundle_path)

    if isinstance(components, nn.Module):
        network = components
    else:
        network = components.get("network")
        if network is None:
            raise RuntimeError(f"Failed to load network from MONAI bundle '{bundle_name}'.")
        state_dict = components.get("state_dict") or components.get("model")
        if state_dict:
            network.load_state_dict(state_dict)
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
    # Load checkpoint to CPU first to avoid GPU memory spike, then load into model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model") or checkpoint
    model.load_state_dict(state_dict)


def run_inference_fold(fold: int, args: argparse.Namespace):
    device = _default_device()
    checkpoint_path = Path(args.checkpoint.format(fold=fold))
    output_dir = Path(args.output_dir.format(fold=fold))

    print(f"\n{'='*80}\nRunning inference for fold {fold}\nCheckpoint: {checkpoint_path}\nOutputs will be saved to: {output_dir}\n{'='*80}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint for fold {fold} not found at {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(args.seed + fold)

    test_ids = load_test_ids_from_dataset_json(args.dataset_json)
    index = build_case_index(args.data_root)
    dataset = build_dataset(test_ids, index, args.spacing, args.patch_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    model = initialise_model(args.bundle_name, args.artifacts_dir, device, args.bundle_dir)
    model = replace_output_head(model, NUM_OUTPUT_CHANNELS)
    load_checkpoint(model, checkpoint_path, device)
    model.to(device)

    model.eval()
    post_pred = AsDiscrete(argmax=True)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            # Keep the original label tensor on the CPU to use as a spatial reference
            label_ref = batch["label"]
            case_id = batch["case_id"][0]
            output_path = output_dir / f"{case_id}.nii.gz"

            with autocast('cuda', enabled=args.amp):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=args.roi_size,
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=args.sw_overlap,
                    mode="gaussian",
                )

            logits_tensor = logits.as_tensor() if isinstance(logits, MetaTensor) else logits
            # `decollate_batch` creates a list of tensors. `post_pred` then removes the channel dim.
            # The result is a list containing one tensor of shape [H, W, D].
            pred_tensor = [post_pred(tensor) for tensor in decollate_batch(logits_tensor)][0]

            # --- THE FIX IS HERE ---
            # Add the channel dimension back for the resampling transform.
            # Shape changes from [H, W, D] to [1, H, W, D].
            pred_tensor_with_channel = pred_tensor.unsqueeze(0)
            
            # Create a dictionary for the resampling transform.
            # The label_ref tensor still holds the original spatial metadata.
            resample_data = {"pred": pred_tensor_with_channel, "ref": label_ref}
            
            # Create the resampler. It will resample 'pred' to match the geometry of 'ref'.
            resampler = ResampleToMatchd(keys="pred", key_dst="ref", mode="nearest")
            resampled_dict = resampler(resample_data)

            # The resampled tensor contains the correct data and the correct affine.
            final_pred_tensor = resampled_dict["pred"]
            
            # Extract the data and affine for saving
            final_affine = final_pred_tensor.affine.cpu().numpy()
            # Squeeze the channel dimension before converting to numpy for saving
            final_data = final_pred_tensor.squeeze().cpu().numpy()

            nib.save(nib.Nifti1Image(final_data.astype(np.int16), final_affine), output_path)
            print(f"Saved prediction for {case_id} -> {output_path}")


    if args.run_evaluation:
        if args.ground_truth is None:
            raise ValueError("--run-evaluation requires --ground-truth to be specified.")
        eval_dir = args.evaluation_output or (output_dir.parent / f"reports_fold{fold}")
        eval_dir = Path(eval_dir).resolve()
        eval_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python",
            "scripts/run_full_evaluation.py",
            str(args.ground_truth.resolve()),
            str(output_dir.resolve()),
            "--output-dir",
            str(eval_dir),
        ]
        print("Running evaluation:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"Inference for fold {fold} complete.")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = _default_device()
    print(f"Using device: {device}")

    if args.fold.lower() == "all":
        if "{fold}" not in args.checkpoint or "{fold}" not in args.output_dir:
            raise ValueError("When using --fold all, --checkpoint and --output-dir must contain '{fold}' placeholder.")
        folds_to_run = range(5)
    else:
        try:
            fold_idx = int(args.fold)
            if not 0 <= fold_idx < 5:
                raise ValueError
            folds_to_run = [fold_idx]
        except (ValueError, TypeError):
            raise ValueError(f"Invalid fold specified: '{args.fold}'. Must be an integer 0-4 or 'all'.")

    for fold in folds_to_run:
        run_inference_fold(fold, args)


if __name__ == "__main__":  # pragma: no cover
    main()