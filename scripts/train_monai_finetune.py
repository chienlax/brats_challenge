"""Fine-tune the MONAI BraTS bundle on BraTS post-treatment data.

This script implements the two-stage curriculum described in the project plan:

1. Stage 1 freezes the encoder and trains the decoder and new classification head
   with a relatively high learning rate to adapt to the Resection Cavity class.
2. Stage 2 unfreezes the entire network and continues training end-to-end with
   a much lower learning rate so the encoder can adapt gently to post-treatment
   MRI appearances.

The training pipeline mirrors nnU-Net v2 preprocessing and augmentation choices:
- Orientation is normalised to RAS and volumes are resampled to 1 mm isotropic.
- Modalities are Z-score normalised over the foreground voxels.
- Training crops use a patch size of 128x128x128 and positive/negative sampling.
- Augmentations have been reduced to lower computational load.

The script expects MONAI and huggingface-hub to be installed in the active environment.
It will download the model bundle from Hugging Face on its first run into an `artifacts`
directory. Checkpoints and metrics are written to the chosen output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

try:
    from huggingface_hub import snapshot_download
    from monai.bundle import load as bundle_load
    from monai.data import CacheDataset, Dataset, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.transforms import (
        AsDiscrete,
        Compose,
        ConcatItemsd,
        CropForegroundd,
        DeleteItemsd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        RandAdjustContrastd,
        RandAffined,
        RandBiasFieldd,
        RandCoarseDropoutd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandGaussianNoised,
        RandGaussianSmoothd,
        RandHistogramShiftd,
        RandZoomd,
        Spacingd,
        SpatialPadd,
    )
except ImportError as exc:  # pragma: no cover - surface helpful error
    raise ImportError(
        "MONAI and huggingface-hub are required. Install them with `pip install monai huggingface-hub`."
    ) from exc

try:  # pragma: no cover - SaveImaged may not be used directly but keep compatibility
    from monai.transforms import SaveImaged  # type: ignore
except ImportError:
    SaveImaged = None


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Modality order aligned with nnU-Net V2 dataset.json for Dataset501_BraTSPostTx
# 0: T2 FLAIR, 1: T1-pre, 2: T1-post, 3: T2
MODALITIES: Tuple[str, ...] = ("t2f", "t1n", "t1c", "t2w")
NUM_OUTPUT_CHANNELS = 5  # BG + ET + NETC + SNFH + RC


@dataclass
class StageConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_accum: int
    clip_grad: float


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MONAI BraTS bundle on post-treatment data.")
    parser.add_argument(
        "--data-root",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing BraTS case folders (BraTS-GLI-XXXX-XXX).",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        required=True,
        help="Path to the nnU-Net splits_final.json for Dataset501_BraTSPostTx (or equivalent).",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="0",
        help='Fold index to use for validation, or "all" to run 5-fold cross-validation (default: "0").',
    )
    parser.add_argument(
        "--bundle-name",
        type=str,
        default="brats_mri_segmentation",
        help="Name of the MONAI model bundle on Hugging Face Hub (e.g., 'brats_mri_segmentation').",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store downloaded model bundles (default: ./artifacts).",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        help="Optional path to a local bundle directory, overriding the default in --artifacts-dir.",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=30,
        help="Number of epochs for the decoder/head warm-up stage (default: 30).",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=50,
        help="Number of epochs for the end-to-end fine-tuning stage (default: 50).",
    )
    parser.add_argument(
        "--stage1-lr",
        type=float,
        default=1e-3,
        help="Learning rate for Stage 1 (default: 1e-3).",
    )
    parser.add_argument(
        "--stage2-lr",
        type=float,
        default=1e-5,
        help="Learning rate for Stage 2 (default: 1e-5).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay applied in both stages (default: 1e-5).",
    )
    parser.add_argument(
        "--stage1-accum",
        type=int,
        default=3,
        help="Gradient accumulation steps during Stage 1 (default: 3).",
    )
    parser.add_argument(
        "--stage2-accum",
        type=int,
        default=3,
        help="Gradient accumulation steps during Stage 2 (default: 2).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=12.0,
        help="Gradient clipping threshold (default: 12.0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-iteration batch size. Default 2 to match nnU-Net plans.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of worker processes for the DataLoader (default: 10).",
    )
    parser.add_argument(
        "--cache-rate",
        type=float,
        default=0.1,
        help="Fraction of the training dataset to keep in memory (default: 0.1).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        help="Spatial size for training patches, aligned with nnU-Net plans (default: 128 128 128).",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        help="Target voxel spacing in mm (default: 1.0 1.0 1.0).",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=2,
        help="Validate every N epochs (default: 2).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision training (recommended).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for reproducibility (default: 2024).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/monai_ft"),
        help="Directory where checkpoints and logs will be written (default: outputs/monai_ft).",
    )
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs=NUM_OUTPUT_CHANNELS,
        default=[0.5, 20.0, 1.0, 4.0, 2.5],
        help=f"Class weights (BG, ET, NETC, SNFH, RC) for DiceCE loss. Default is an inverse-frequency based heuristic.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Optional checkpoint path to resume training.",
    )
    parser.add_argument(
        "--load-weights",
        type=Path,
        help="Path to a checkpoint file to load initial model weights from. "
             "Unlike --resume, this only loads the weights and starts a fresh training run.",
    )
    parser.add_argument(
        "--encoder-prefix",
        type=str,
        nargs="*",
        default=("encoder", "backbone", "swinViT", "downsample"),
        help="Parameter name prefixes that belong to the encoder (frozen in Stage 1).",
    )
    parser.add_argument(
        "--head-key",
        type=str,
        default="seg_layers",
        help="Attribute name of the model's final convolutional layer (used when replacing the head).",
    )
    parser.add_argument(
        "--eval-roi",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        help="Sliding window ROI size for validation inference, aligned with nnU-Net plans (default: 128 128 128).",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1,
        help="Sliding window batch size during validation inference (default: 1).",
    )
    parser.add_argument(
        "--sw-overlap",
        type=float,
        default=0.5,
        help="Sliding window overlap fraction (default: 0.5).",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        default=1,
        help="Write checkpoint every N epochs in addition to the best models (default: 10).",
    )
    return parser.parse_args(argv)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_splits(split_json: Path, fold: int) -> Tuple[List[str], List[str]]:
    with split_json.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    if not isinstance(splits, list) or len(splits) <= fold:
        raise ValueError(f"Split index {fold} not found in {split_json}.")
    entry = splits[fold]
    train_ids = [str(case) for case in entry["train"]]
    val_ids = [str(case) for case in entry["val"]]
    return train_ids, val_ids


def build_case_index(data_roots: Sequence[Path]) -> Mapping[str, Path]:
    index: Dict[str, Path] = {}
    for root in data_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Data root {root} does not exist or is not a directory.")
        for case_dir in sorted(root.glob("BraTS-GLI-*")):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            index[case_id] = case_dir
    if not index:
        raise RuntimeError("No BraTS case directories were found in the provided data roots.")
    return index


def case_dict(case_dir: Path) -> Mapping[str, str]:
    record: Dict[str, str] = {"label": str(case_dir / f"{case_dir.name}-seg.nii.gz")}
    if not Path(record["label"]).exists():
        raise FileNotFoundError(f"Missing segmentation for case {case_dir.name}: {record['label']}")
    for modality in MODALITIES:
        modality_path = case_dir / f"{case_dir.name}-{modality}.nii.gz"
        if not modality_path.exists():
            raise FileNotFoundError(f"Missing modality '{modality}' for case {case_dir.name}: {modality_path}")
        record[modality] = str(modality_path)
    record["case_id"] = case_dir.name
    return record


def prepare_datasets(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    case_index: Mapping[str, Path],
    spacing: Sequence[float],
    patch_size: Sequence[int],
    cache_rate: float,
) -> Tuple[CacheDataset, Dataset]:
    train_records = [case_dict(case_index[case_id]) for case_id in train_ids]
    val_records = [case_dict(case_index[case_id]) for case_id in val_ids]

    base_transforms = [
        LoadImaged(keys=list(MODALITIES) + ["label"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=list(MODALITIES) + ["label"]),
        ConcatItemsd(keys=list(MODALITIES), name="image", dim=0),
        DeleteItemsd(keys=list(MODALITIES)),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="label", allow_smaller=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant"),
    ]

    train_aug = Compose(
        base_transforms
        + [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
            # --- Computationally expensive augmentations removed for faster training ---
            # RandAffined(
            #     keys=["image", "label"],
            #     rotate_range=(math.radians(30), math.radians(30), math.radians(30)),
            #     scale_range=(0.1, 0.1, 0.1),
            #     mode=("bilinear", "nearest"),
            #     prob=0.4,
            # ),
            # RandBiasFieldd(keys="image", prob=0.1, coeff_range=(0.02, 0.07)),
            # RandZoomd(
            #     keys=["image", "label"],
            #     min_zoom=0.7,
            #     max_zoom=1.3,
            #     mode=("trilinear", "nearest"),
            #     prob=0.2,
            #     keep_size=True,
            # ),
            RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.05),
            RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.7, 1.5)),
            RandHistogramShiftd(keys="image", prob=0.2, num_control_points=10),
            RandGaussianSmoothd(keys="image", sigma_x=(0.3, 1.2), sigma_y=(0.3, 1.2), sigma_z=(0.3, 1.2), prob=0.15),
            RandCoarseDropoutd(
                keys="image",
                holes=1,
                max_holes=5,
                spatial_size=(32, 32, 32),
                max_spatial_size=(96, 96, 96),
                fill_value=0.0,
                prob=0.25,
            ),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
        ]
    )

    val_transforms = Compose(
        base_transforms
        + [EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64))]
    )

    train_dataset = CacheDataset(data=train_records, transform=train_aug, cache_rate=cache_rate)
    val_dataset = Dataset(data=val_records, transform=val_transforms)
    return train_dataset, val_dataset


def train_collate_fn(batch: List[List[Mapping[str, torch.Tensor]]]) -> Mapping[str, torch.Tensor]:
    batch = [item[0] for item in batch]
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    meta = {key: [item[key] for item in batch] for key in batch[0] if key not in {"image", "label"}}
    return {"image": images, "label": labels, **meta}


def val_collate_fn(batch: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    meta = {key: [item[key] for item in batch] for key in batch[0] if key not in {"image", "label"}}
    return {"image": images, "label": labels, **meta}


def initialise_model(
    bundle_name: str,
    artifacts_dir: Path,
    device: torch.device,
    head_key: str,
    new_out_channels: int,
    bundle_dir: Path | None = None,
) -> nn.Module:
    """Initialises model and replaces the output head, handling nested modules and direct model returns."""
    local_bundle_path = bundle_dir
    if local_bundle_path is None:
        local_bundle_path = artifacts_dir / bundle_name
        if not local_bundle_path.exists():
            repo_id = f"MONAI/{bundle_name}"
            print(f"Bundle not found at {local_bundle_path}. Downloading from Hugging Face: {repo_id}")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=repo_id, local_dir=local_bundle_path, repo_type="model")

    print(f"Loading MONAI bundle from: {local_bundle_path}")
    loaded_bundle = bundle_load(name=bundle_name, bundle_dir=local_bundle_path)

    # Robustly handle direct model return vs. dictionary of components
    network = None
    state_dict = None
    if isinstance(loaded_bundle, dict):
        network = loaded_bundle.get("network")
        state_dict = loaded_bundle.get("state_dict") or loaded_bundle.get("model")
    elif isinstance(loaded_bundle, nn.Module):
        network = loaded_bundle

    if network is None:
        raise RuntimeError(f"Failed to load network from MONAI bundle '{bundle_name}'.")
    if state_dict:
        network.load_state_dict(state_dict)

    # Replace the output head
    try:
        head_module = getattr(network, head_key)
        
        final_conv = None
        if isinstance(head_module, nn.Conv3d):
            final_conv = head_module
        elif isinstance(head_module, nn.Sequential):
            for layer in reversed(head_module):
                if isinstance(layer, nn.Conv3d):
                    final_conv = layer
                    break
        
        if final_conv is None:
            raise TypeError(f"Could not find a Conv3d layer within attribute '{head_key}'.")

        new_conv = nn.Conv3d(
            in_channels=final_conv.in_channels,
            out_channels=new_out_channels,
            kernel_size=final_conv.kernel_size,
            stride=final_conv.stride,
            padding=final_conv.padding,
            dilation=final_conv.dilation,
            groups=final_conv.groups,
            bias=final_conv.bias is not None,
        )
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

        if isinstance(head_module, nn.Conv3d):
            setattr(network, head_key, new_conv)
        elif isinstance(head_module, nn.Sequential):
            for i, layer in enumerate(head_module):
                if layer is final_conv:
                    head_module[i] = new_conv
                    break
        
        print(f"Replaced model head within attribute '{head_key}' for {new_out_channels} classes.")
    except AttributeError:
        raise RuntimeError(f"Could not find attribute '{head_key}' in model to replace the head.")
    
    network.to(device)
    return network

def set_encoder_trainable(model: nn.Module, encoder_prefixes: Sequence[str], trainable: bool) -> None:
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in encoder_prefixes):
            param.requires_grad = trainable


def stage_config_from_args(args: argparse.Namespace) -> Tuple[StageConfig, StageConfig]:
    stage1 = StageConfig(
        epochs=args.stage1_epochs,
        learning_rate=args.stage1_lr,
        weight_decay=args.weight_decay,
        grad_accum=args.stage1_accum,
        clip_grad=args.grad_clip,
    )
    stage2 = StageConfig(
        epochs=args.stage2_epochs,
        learning_rate=args.stage2_lr,
        weight_decay=args.weight_decay,
        grad_accum=args.stage2_accum,
        clip_grad=args.grad_clip,
    )
    return stage1, stage2


def create_optimizer(model: nn.Module, config: StageConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No parameters marked trainable; check encoder prefixes.")
    return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)


def create_loss(class_weights: Sequence[float], device: torch.device) -> DiceCELoss:
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        weight=weight_tensor,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: DiceCELoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: StageConfig,
    device: torch.device,
    amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast('cuda', enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss = loss / config.grad_accum

        scaler.scale(loss).backward()

        if step % config.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.detach().item()

    return float(running_loss * config.grad_accum / len(loader))


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float,
) -> Mapping[str, float]:
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_OUTPUT_CHANNELS)
    post_label = AsDiscrete(to_onehot=NUM_OUTPUT_CHANNELS)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with autocast('cuda', enabled=amp):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )
            preds = [post_pred(tensor)[:, 1:] for tensor in decollate_batch(logits.as_tensor())]
            gts = [post_label(tensor)[:, 1:] for tensor in decollate_batch(labels.as_tensor())]
            dice_metric(y_pred=preds, y=gts)

    dice = dice_metric.aggregate().cpu().numpy()
    dice_metric.reset()

    # Map dice scores to class names (ET, NETC, SNFH, RC)
    # The order depends on the label mapping, assuming 1=ET, 2=NETC, 3=SNFH, 4=RC
    metrics = {"dice_ET": float(dice[0]), "dice_NETC": float(dice[1]), "dice_SNFH": float(dice[2]), "dice_RC": float(dice[3])}
    metrics["dice_mean"] = float(np.mean(dice))
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    stage: str,
    best_metric: float,
    output_dir: Path,
    filename: str,
) -> Path:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "best_metric": best_metric,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save(checkpoint, path)
    return path


def ensure_meta_dirs(output_root: Path) -> Mapping[str, Path]:
    dirs = {
        "checkpoints": output_root / "checkpoints",
        "logs": output_root / "logs",
        "metrics": output_root / "metrics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run_fold(fold: int, args: argparse.Namespace):
    """Run the full training pipeline for a single fold."""
    device = _default_device()
    output_root = args.output_root / f"fold{fold}"
    print(f"\n{'='*80}\nRunning training for fold {fold}\nOutputs will be saved to: {output_root}\n{'='*80}")

    if args.resume and args.load_weights:
        raise ValueError("Cannot use --resume and --load-weights simultaneously.")

    set_determinism(args.seed + fold)

    train_ids, val_ids = load_splits(args.split_json, fold)
    index = build_case_index(args.data_root)
    train_dataset, val_dataset = prepare_datasets(
        train_ids=train_ids,
        val_ids=val_ids,
        case_index=index,
        spacing=args.spacing,
        patch_size=args.patch_size,
        cache_rate=args.cache_rate,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=train_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=val_collate_fn,
    )

    model = initialise_model(
        args.bundle_name,
        args.artifacts_dir,
        device,
        args.head_key,
        NUM_OUTPUT_CHANNELS,
        args.bundle_dir,
    )

    if args.load_weights:
        if not args.load_weights.exists():
            raise FileNotFoundError(f"Weights file not found: {args.load_weights}")
        print(f"Loading initial model weights from: {args.load_weights}")
        ckpt = torch.load(args.load_weights, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    stage1_cfg, stage2_cfg = stage_config_from_args(args)
    loss_fn = create_loss(args.class_weights, device)
    history: List[MutableMapping[str, object]] = []
    dirs = ensure_meta_dirs(output_root)
    scaler = GradScaler(enabled=args.amp)
    optimizer: torch.optim.Optimizer

    start_epoch_s1, start_epoch_s2 = 0, 0
    best_stage1, best_stage2 = -math.inf, -math.inf

    if args.resume:
        print(f"Resuming interrupted run from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        scaler.load_state_dict(ckpt["scaler"])
        stage = ckpt.get("stage", "stage1")

        if stage == "stage1":
            start_epoch_s1 = int(ckpt.get("epoch", 0)) + 1
            best_stage1 = float(ckpt.get("best_metric", -math.inf))
        elif stage == "stage2":
            start_epoch_s1 = stage1_cfg.epochs  # Skip stage 1
            start_epoch_s2 = int(ckpt.get("epoch", 0)) + 1
            best_stage2 = float(ckpt.get("best_metric", -math.inf))

    if start_epoch_s1 < stage1_cfg.epochs:
        print("\n--- Stage 1: decoder/head fine-tuning ---")
        set_encoder_trainable(model, args.encoder_prefix, trainable=False)
        optimizer = create_optimizer(model, stage1_cfg)
        if args.resume and ckpt.get("stage") == "stage1":
            optimizer.load_state_dict(ckpt["optimizer"])

        for epoch in range(start_epoch_s1, stage1_cfg.epochs):
            epoch_start = time.time()
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, stage1_cfg, device, args.amp)
            log: MutableMapping[str, object] = {"epoch": epoch, "stage": "stage1", "train_loss": train_loss}

            if (epoch + 1) % args.val_interval == 0:
                metrics = validate(model, val_loader, device, args.amp, args.eval_roi, args.sw_batch_size, args.sw_overlap)
                log.update(metrics)
                if metrics["dice_mean"] > best_stage1:
                    best_stage1 = metrics["dice_mean"]
                    save_checkpoint(model, optimizer, scaler, epoch, "stage1", best_stage1, dirs["checkpoints"], "stage1_best.pt")
                    print(f"  > Stage1 best improved to {best_stage1:.4f}")

            if (epoch + 1) % args.save_checkpoint_frequency == 0:
                save_checkpoint(model, optimizer, scaler, epoch, "stage1", best_stage1, dirs["checkpoints"], f"stage1_epoch{epoch+1}.pt")

            log["epoch_seconds"] = time.time() - epoch_start
            history.append(log)
            print(f"Epoch {epoch+1}/{stage1_cfg.epochs} | loss={train_loss:.4f} | dice_mean={log.get('dice_mean', 'N/A')}")

    print("\n--- Stage 2: end-to-end fine-tuning ---")
    set_encoder_trainable(model, args.encoder_prefix, trainable=True)

    if 'optimizer' not in locals() or args.resume and ckpt.get("stage") == "stage2":
        # Optimizer needs to be created for stage 2, either from scratch or for resuming
        optimizer = create_optimizer(model, stage2_cfg)
        if args.resume and ckpt.get("stage") == "stage2":
            optimizer.load_state_dict(ckpt["optimizer"])
    else:
        # Reconfigure the existing optimizer from stage 1
        print("Reconfiguring optimizer for Stage 2: adding encoder params and updating LR.")
        optimizer.param_groups.clear()
        optimizer.add_param_group({"params": model.parameters()})
        optimizer.param_groups[0]['lr'] = stage2_cfg.learning_rate
        optimizer.param_groups[0]['weight_decay'] = stage2_cfg.weight_decay

    for epoch in range(start_epoch_s2, stage2_cfg.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, stage2_cfg, device, args.amp)
        log = {"epoch": epoch, "stage": "stage2", "train_loss": train_loss}

        if (epoch + 1) % args.val_interval == 0:
            metrics = validate(model, val_loader, device, args.amp, args.eval_roi, args.sw_batch_size, args.sw_overlap)
            log.update(metrics)
            if metrics["dice_mean"] > best_stage2:
                best_stage2 = metrics["dice_mean"]
                save_checkpoint(model, optimizer, scaler, epoch, "stage2", best_stage2, dirs["checkpoints"], "stage2_best.pt")
                print(f"  > Stage2 best improved to {best_stage2:.4f}")

        if (epoch + 1) % args.save_checkpoint_frequency == 0:
            save_checkpoint(model, optimizer, scaler, epoch, "stage2", best_stage2, dirs["checkpoints"], f"stage2_epoch{epoch+1}.pt")

        log["epoch_seconds"] = time.time() - epoch_start
        history.append(log)
        print(f"Epoch {epoch+1}/{stage2_cfg.epochs} | loss={train_loss:.4f} | dice_mean={log.get('dice_mean', 'N/A')}")

    metrics_path = dirs["metrics"] / "training_log.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Training for fold {fold} complete. Logs written to {metrics_path}")

def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = _default_device()
    print(f"Using device: {device}")

    if args.fold.lower() == "all":
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
        run_fold(fold, args)


if __name__ == "__main__":  # pragma: no cover
    main()