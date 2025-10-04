"""Fine-tune the MONAI BraTS bundle on BraTS post-treatment data.

This script implements the two-stage curriculum described in the project plan:

1. Stage 1 freezes the encoder and trains the decoder and new classification head
   with a relatively high learning rate to adapt to the Resection Cavity class.
2. Stage 2 unfreezes the entire network and continues training end-to-end with
   a much lower learning rate so the encoder can adapt gently to post-treatment
   MRI appearances.

The training pipeline mirrors nnU-Net preprocessing and augmentation choices to
ensure a fair comparison:
- Orientation is normalised to RAS and volumes are resampled to 1 mm isotropic.
- Modalities are normalised with percentile-based intensity scaling.
- Training crops use positive/negative sampling around foreground voxels.
- Augmentations include flips, affine jitter, histogram shifts, bias fields,
  Gaussian smoothing, coarse dropout, and low-resolution simulation.

Usage example (PowerShell):

python scripts/train_monai_finetune.py --data-root training_data training_data_additional `
--split-json nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json --fold 0 `
--output-root outputs/monai_ft `
--bundle-name brats_mri_segmentation

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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
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
        ScaleIntensityRangePercentilesd,
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


MODALITIES: Tuple[str, ...] = ("t1c", "t1n", "t2w", "t2f")
NUM_OUTPUT_CHANNELS = 5  # BG + NETC + ET + SNFH + RC


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
    parser.add_argument("--fold", type=int, default=0, help="Fold index to use for validation (default: 0).")
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
        default=80,
        help="Number of epochs for the decoder/head warm-up stage (default: 80).",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=140,
        help="Number of epochs for the end-to-end fine-tuning stage (default: 140).",
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
        default=2,
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
        default=1,
        help="Per-iteration batch size (default: 1 to fit on 12 GB GPUs).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes for the DataLoader (default: 2).",
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
        default=(224, 224, 144),
        help="Spatial size for training patches (default: 224 224 144).",
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
        nargs=4,
        default=(1.0, 1.0, 1.0, 1.5),
        help="Class weights (ET, NETC, SNFH, RC) for the DiceCE loss (background is excluded).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Optional checkpoint path to resume training.",
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
        default="output_layer",
        help="Attribute name of the model's final convolutional layer (used when replacing the head).",
    )
    parser.add_argument(
        "--eval-roi",
        type=int,
        nargs=3,
        default=(224, 224, 144),
        help="Sliding window ROI size for validation inference (default: 224 224 144).",
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
        default=10,
        help="Write checkpoint every N epochs in addition to the best models (default: 10).",
    )
    return parser.parse_args(argv)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        EnsureChannelFirstd(keys=list(MODALITIES) + ["label"]),  # Added to ensure label has channel dim
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
        # SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant"),
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
            RandAffined(
                keys=["image", "label"],
                rotate_range=(math.radians(30), math.radians(30), math.radians(30)),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                prob=0.4,
            ),
            RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.05),
            RandBiasFieldd(keys="image", prob=0.1, coeff_range=(0.02, 0.07)),
            RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.7, 1.5)),
            RandHistogramShiftd(keys="image", prob=0.2, num_control_points=10),
            RandGaussianSmoothd(keys="image", sigma_x=(0.3, 1.2), sigma_y=(0.3, 1.2), sigma_z=(0.3, 1.2), prob=0.15),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.7,
                max_zoom=1.3,
                mode=("trilinear", "nearest"),
                prob=0.2,
                keep_size=True,
            ),
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


def collate_fn(batch: List[Mapping[str, torch.Tensor]]) -> Mapping[str, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    meta = {key: [item[key] for item in batch] for key in batch[0] if key not in {"image", "label"}}
    return {"image": images, "label": labels, **meta}


def initialise_model(
    bundle_name: str, artifacts_dir: Path, device: torch.device, bundle_dir: Path | None = None
) -> nn.Module:
    """Initialises model, downloading from Hugging Face if not present."""
    # Determine the local path for the bundle
    local_bundle_path = bundle_dir
    if local_bundle_path is None:
        local_bundle_path = artifacts_dir / bundle_name
        # If the bundle doesn't exist locally, download it from Hugging Face
        if not local_bundle_path.exists():
            repo_id = f"MONAI/{bundle_name}"
            print(f"Bundle not found at {local_bundle_path}. Downloading from Hugging Face: {repo_id}")
            # Create parent directory if it doesn't exist
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=repo_id, local_dir=local_bundle_path, repo_type="model")

    print(f"Loading MONAI bundle from: {local_bundle_path}")
    components = bundle_load(
        bundle_dir=local_bundle_path,
    )

    network = components.get("network")
    if network is None:
        raise RuntimeError(f"Failed to load network from MONAI bundle '{bundle_name}'.")

    state_dict = components.get("state_dict") or components.get("model")
    if state_dict:
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
        raise RuntimeError("Could not locate the final Conv3d layer to replace.")
    return model


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


def create_loss(class_weights: Sequence[float]) -> DiceCELoss:
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
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

        with autocast(enabled=amp):
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

            with autocast(enabled=amp):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )
            preds = [post_pred(tensor)[:, 1:] for tensor in decollate_batch(logits)]
            gts = [post_label(tensor)[:, 1:] for tensor in decollate_batch(labels)]
            dice_metric(y_pred=preds, y=gts)

    dice = dice_metric.aggregate().cpu().numpy()
    dice_metric.reset()

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


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler, checkpoint_path: Path) -> Tuple[int, float, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_metric", -math.inf)), str(checkpoint.get("stage", "stage1"))


def ensure_meta_dirs(output_root: Path) -> Mapping[str, Path]:
    dirs = {
        "checkpoints": output_root / "checkpoints",
        "logs": output_root / "logs",
        "metrics": output_root / "metrics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    device = _default_device()
    print(f"Using device: {device}")

    set_determinism(args.seed)

    train_ids, val_ids = load_splits(args.split_json, args.fold)
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
        persistent_workers=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_fn,
    )

    model = initialise_model(args.bundle_name, args.artifacts_dir, device, args.bundle_dir)
    model = replace_output_head(model, NUM_OUTPUT_CHANNELS)

    stage1_cfg, stage2_cfg = stage_config_from_args(args)
    loss_fn = create_loss(args.class_weights)

    history: List[MutableMapping[str, object]] = []
    dirs = ensure_meta_dirs(args.output_root)

    scaler = GradScaler(enabled=args.amp)

    # Stage 1: freeze encoder
    print("\n--- Stage 1: decoder/head fine-tuning ---")
    set_encoder_trainable(model, args.encoder_prefix, trainable=False)
    optimizer = create_optimizer(model, stage1_cfg)
    start_epoch = 0
    best_stage1 = -math.inf

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        last_epoch, best_metric, previous_stage = load_checkpoint(model, optimizer, scaler, args.resume)
        if previous_stage != "stage1":
            raise RuntimeError("Resume checkpoint belongs to a different stage. Use stage-specific checkpoints.")
        start_epoch = last_epoch + 1
        best_stage1 = best_metric

    for epoch in range(start_epoch, stage1_cfg.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, stage1_cfg, device, args.amp)
        log: MutableMapping[str, object] = {"epoch": epoch, "stage": "stage1", "train_loss": train_loss}

        if (epoch + 1) % args.val_interval == 0:
            metrics = validate(model, val_loader, device, args.amp, args.eval_roi, args.sw_batch_size, args.sw_overlap)
            log.update(metrics)
            if metrics["dice_mean"] > best_stage1:
                best_stage1 = metrics["dice_mean"]
                ckpt_path = save_checkpoint(model, optimizer, scaler, epoch, "stage1", best_stage1, dirs["checkpoints"], "stage1_best.pt")
                print(f"  > Stage1 best improved to {best_stage1:.4f} (checkpoint: {ckpt_path.name})")

        if (epoch + 1) % args.save_checkpoint_frequency == 0:
            ckpt_path = save_checkpoint(model, optimizer, scaler, epoch, "stage1", best_stage1, dirs["checkpoints"], f"stage1_epoch{epoch+1}.pt")
            print(f"  > Saved checkpoint {ckpt_path.name}")

        log["epoch_seconds"] = time.time() - epoch_start
        history.append(log)
        print(f"Epoch {epoch+1}/{stage1_cfg.epochs} | loss={train_loss:.4f}")

    # Stage 2: unfreeze encoder
    print("\n--- Stage 2: end-to-end fine-tuning ---")
    set_encoder_trainable(model, args.encoder_prefix, trainable=True)
    optimizer = create_optimizer(model, stage2_cfg)
    best_stage2 = -math.inf

    start_epoch = 0
    if args.resume and args.resume.name.startswith("stage2"):
        last_epoch, best_metric, previous_stage = load_checkpoint(model, optimizer, scaler, args.resume)
        if previous_stage != "stage2":
            raise RuntimeError("Resume checkpoint belongs to a different stage. Use stage-specific checkpoints.")
        start_epoch = last_epoch + 1
        best_stage2 = best_metric

    for epoch in range(start_epoch, stage2_cfg.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, stage2_cfg, device, args.amp)
        log = {"epoch": epoch, "stage": "stage2", "train_loss": train_loss}

        if (epoch + 1) % args.val_interval == 0:
            metrics = validate(model, val_loader, device, args.amp, args.eval_roi, args.sw_batch_size, args.sw_overlap)
            log.update(metrics)
            if metrics["dice_mean"] > best_stage2:
                best_stage2 = metrics["dice_mean"]
                ckpt_path = save_checkpoint(model, optimizer, scaler, epoch, "stage2", best_stage2, dirs["checkpoints"], "stage2_best.pt")
                print(f"  > Stage2 best improved to {best_stage2:.4f} (checkpoint: {ckpt_path.name})")

        if (epoch + 1) % args.save_checkpoint_frequency == 0:
            ckpt_path = save_checkpoint(model, optimizer, scaler, epoch, "stage2", best_stage2, dirs["checkpoints"], f"stage2_epoch{epoch+1}.pt")
            print(f"  > Saved checkpoint {ckpt_path.name}")

        log["epoch_seconds"] = time.time() - epoch_start
        history.append(log)
        print(f"Epoch {epoch+1}/{stage2_cfg.epochs} | loss={train_loss:.4f}")

    metrics_path = dirs["metrics"] / "training_log.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Training complete. Logs written to {metrics_path}")


if __name__ == "__main__":  # pragma: no cover
    main()