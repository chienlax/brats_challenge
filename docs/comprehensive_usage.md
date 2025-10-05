# BraTS Post-treatment Toolkit – Comprehensive CLI Usage

**Last Updated:** October 5, 2025

This document consolidates the end-to-end workflow for every maintained script in the `scripts/` directory. It expands on `docs/usage_guide.md`, the MONAI and nnU-Net installation notes, and the inline help bundled with each CLI. Keep it close whenever you stage data, train models, run inference, or evaluate results.

---

## 1. Before you start

### 1.1 Repository layout refresher

- Source code lives under `apps/` (Dash apps) and `scripts/` (CLI utilities).
- Raw BraTS cases belong in `training_data/`, `training_data_additional/`, or `validation_data/`; keep them out of Git.
- Generated assets go into `outputs/` (ignored by Git) to preserve reproducibility.

### 1.2 Python environments

| Workflow | Interpreter | Commands (PowerShell) |
|----------|-------------|-----------------------|
| Core tooling, visualization, statistics | `.venv\Scripts\python.exe` | ```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
``` |
| nnU-Net dataset prep & evaluation | `.venv_nnunet\Scripts\python.exe` | ```powershell
python -m venv .venv_nnunet
.\.venv_nnunet\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_nnunet.txt --extra-index-url https://download.pytorch.org/whl/cu124
``` |
| MONAI fine-tuning & inference | `.venv_monai\Scripts\python.exe` | ```powershell
python -m venv .venv_monai
.\.venv_monai\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install "monai[all]" nibabel pydicom ipywidgets==8.1.2
pip uninstall -y pytensor
pip install -U filelock
``` |

> **Tip:** Always re-run the appropriate `Activate.ps1` when you open a new shell. If PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### 1.3 Data staging checklist

1. Extract each BraTS case into its own folder (`BraTS-GLI-xxxxx-yyy/`).
2. Confirm the five canonical files exist: `-t1c`, `-t1n`, `-t2f`, `-t2w`, `-seg` (all `.nii.gz`).
3. Mirror the folders into your chosen dataset roots (`training_data*/` for training, `validation_data/` for hold-outs).
4. Keep directories read-only; scripts never mutate raw data.

### 1.4 Environment variables for nnU-Net

When preparing datasets or evaluating nnU-Net results, make sure the storage directories exist and are exported (see `docs/nnU-Net_guide.md` for the rationale):

```powershell
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_results | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\reports\metrics | Out-Null

setx nnUNet_raw "${PWD}\outputs\nnunet\nnUNet_raw"
setx nnUNet_preprocessed "${PWD}\outputs\nnunet\nnUNet_preprocessed"
setx nnUNet_results "${PWD}\outputs\nnunet\nnUNet_results"
```

---

## 2. Script quick reference

| Script | Purpose | Recommended environment | Typical inputs | Primary outputs |
|--------|---------|-------------------------|----------------|-----------------|
| `scripts/analyze_statistics.py` | Launch Dash app for per-case & cohort statistics | `.venv` | Dataset roots | In-browser UI, cached JSON under `outputs/statistics_cache/` |
| `scripts/visualize_volumes.py` | Interactive viewer, static PNGs, GIF generation | `.venv` | BraTS case folders | Dash app, PNG figures, GIF animations |
| `scripts/prepare_nnunet_dataset.py` | Convert BraTS folders into nnU-Net v2 dataset structure | `.venv_nnunet` | Raw case roots | `outputs/nnunet/nnUNet_raw/Dataset*/` tree + `dataset.json` |
| `scripts/train_monai_finetune.py` | Two-stage MONAI fine-tuning aligned with nnU-Net folds | `.venv_monai` | Case roots, `splits_final.json` | Checkpoints, training logs under `outputs/monai_ft*/` |
| `scripts/infer_monai_finetune.py` | Sliding-window inference from MONAI checkpoints | `.venv_monai` | Case roots, `dataset.json`, checkpoints | NIfTI predictions, optional evaluation reports |
| `scripts/compute_brats_lesion_metrics.py` | Lesion-wise BraTS metrics for predictions vs. ground truth | `.venv` or `.venv_nnunet` | Ground-truth + prediction directories | JSON metrics (per-case + summary) |
| `scripts/run_full_evaluation.py` | Drive nnU-Net CLI evaluation + lesion metrics in one step | `.venv_nnunet` | Ground-truth + prediction directories | nnU-Net metrics, lesion metrics, merged manifest |
| `scripts/visualize_architecture.py` | Plot nnU-Net architecture details from `plans.json` | `.venv` (w/ matplotlib) | `plans.json` path | PNG/SVG diagrams, console table |

---

## 3. Detailed usage by script

### 3.1 `scripts/analyze_statistics.py`

**What it does:** hosts the statistics inspector Dash app with two tabs:
- *Case Statistics* renders modality histograms and segmentation label tables.
- *Dataset Statistics* computes aggregated metrics (spacing, intensity percentiles, label distributions) and caches them under `outputs/statistics_cache/` for speedy reloads.

**Run it:**

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/analyze_statistics.py --open-browser
```

**Key arguments:**

| Flag | Description |
|------|-------------|
| `--data-root PATH` (repeatable) | Override the dataset roots (defaults: `training_data`, `training_data_additional`, `validation_data`). |
| `--host`, `--port` | Bind interface/port (default `127.0.0.1:8050`). |
| `--debug` | Enable Dash hot reload for development. |
| `--open-browser` | Auto-launch the default browser after the server starts. |

**Workflow tips:**
- Launch once per session; cached aggregates avoid recomputation between runs.
- The dataset tab exposes `Recompute` and `Load Cached` controls—use `Recompute` after adding new cases.
- All case discovery respects the BraTS naming convention; missing modalities raise a visible error.

### 3.2 `scripts/visualize_volumes.py`

**What it does:** a single entry point for interactive browsing, static figure export, and GIF animation. Three subcommands are exposed through the `--mode` flag.

**Environments:** base `.venv` (depends on matplotlib, imageio, Dash).

#### Interactive mode (Dash app)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/visualize_volumes.py --mode interactive --open-browser
```

- Shares the same Dash backend as `analyze_statistics.py` but focuses on slice viewers.
- Accepts multiple `--data-root` entries; defaults to all known roots.
- Supports `--debug`, `--host`, `--port`, `--open-browser` like the statistics app.

#### Static figure mode

```powershell
python scripts/visualize_volumes.py --mode static `
    --case-dir training_data_additional/BraTS-GLI-02570-100 `
    --layout modalities `
    --axis axial `
    --fraction 0.55 `
    --output outputs/figures/BraTS-GLI-02570-100_modalities.png
```

- `--layout modalities` shows all available modalities on one slice; choose slice via `--fraction` or `--index`.
- `--layout orthogonal` produces sagittal/coronal/axial views for one modality (set with `--modality` and optional `--indices`).
- Overlays default to on; add `--no-overlay` to disable segmentation blending.

#### GIF mode

```powershell
python scripts/visualize_volumes.py --mode gif `
    --root training_data_additional `
    --case BraTS-GLI-02570-100 `
    --axis axial `
    --modality t2f `
    --step 3 `
    --overlay `
    --output-dir outputs/gifs
```

- Processes either a single case (`--case`) or a whole root (optionally capped via `--max-cases`).
- Per-frame rendering honours percentile normalization and optional overlays.
- GIFs appear in `--output-dir` named `<case>_<modality>_<axis>.gif`.

**Troubleshooting:**
- All modes raise fast if a modality or segmentation is missing—fix the dataset rather than ignoring it.
- When running headless, keep `--show` unset for static mode; figures will still be written to disk.

### 3.3 `scripts/prepare_nnunet_dataset.py`

**Purpose:** reorganize BraTS folders into the nnU-Net v2 raw format (`imagesTr/`, `labelsTr/`, `imagesTs/`, `labelsTs/`) and emit a `dataset.json` describing the cohort.

**Run it:**

```powershell
.\.venv_nnunet\Scripts\Activate.ps1
python scripts/prepare_nnunet_dataset.py `
    --clean `
    --train-sources training_data training_data_additional `
    --test-sources validation_data `
    --dataset-id Dataset501_BraTSPostTx
```

**Key behaviour:**
- Creates hard links where possible; falls back to copy if the filesystem forbids linking.
- Validates that every training case contains all four modalities and a segmentation.
- Test cases may omit `-seg` unless `--require-test-labels` is specified (recommended when evaluating against labels).

**Arguments to know:**

| Flag | Notes |
|------|-------|
| `--train-sources`, `--test-sources` | Override default roots. Multiple values allowed. |
| `--dataset-id` | Target folder inside `nnUNet_raw`. Must match downstream training/evaluation IDs. |
| `--nnunet-raw` | Custom raw root (defaults to `outputs/nnunet/nnUNet_raw`). |
| `--clean` | Remove any existing dataset folder before rebuilding (prevents stale cases). |
| `--require-test-labels` | Abort if any test case lacks a segmentation; ensures `labelsTs/` is complete. |

**Outputs:**
- `outputs/nnunet/nnUNet_raw/<dataset-id>/imagesTr/*.nii.gz`
- `outputs/nnunet/nnUNet_raw/<dataset-id>/labelsTr/*.nii.gz`
- Optional `imagesTs/` and `labelsTs/`
- `dataset.json` describing modalities and label schema.

**Follow-up:** always regenerate `nnUNet_preprocessed` by running `nnUNetv2_plan_and_preprocess` after dataset structure changes.

### 3.4 `scripts/train_monai_finetune.py`

**Purpose:** fine-tune the MONAI BraTS bundle using nnU-Net fold definitions in two stages: decoder/head adaptation (Stage 1) and full-network fine-tuning (Stage 2).

**Run it (single fold):**

```powershell
.\.venv_monai\Scripts\Activate.ps1
python scripts/train_monai_finetune.py `
    --data-root training_data training_data_additional `
    --split-json outputs/nnunet/nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json `
    --fold 0 `
    --output-root outputs/monai_ft_nnunet_aligned `
    --amp
```

**Essential inputs:**
- BraTS case roots (segmented cases only).
- `splits_final.json` from nnU-Net preprocessing to align folds.
- Stable GPU-enabled MONAI environment (`.venv_monai`).

**Key arguments:**

| Flag | Description |
|------|-------------|
| `--fold` (`0`-`4` or `all`) | Controls validation fold. `all` loops over all five folds. |
| `--stage1-*` / `--stage2-*` | Epoch counts, learning rates, grad accumulation, etc. |
| `--patch-size`, `--spacing` | Resampling and crop geometry (defaults align with nnU-Net plans). |
| `--class-weights` | DiceCE weights for background + tumor subregions (tuple of 5 values). |
| `--resume` | Resume Stage 1 or Stage 2 from a saved checkpoint. Stage-specific checks enforce safety. |
| `--amp` | Enables PyTorch AMP for faster, memory-friendly runs. |

**Outputs:**
- Checkpoints under `outputs/monai_ft*/fold*/checkpoints/` (stage best + epoch snapshots).
- Metrics log (`metrics/training_log.json`) capturing per-epoch losses and Dice scores.
- Console summary per epoch for quick monitoring.

**Best practices:**
- Monitor GPU memory; adjust `--batch-size`, `--stage*-accum`, or `--patch-size` on smaller cards.
- Use `--save-checkpoint-frequency` to control periodic snapshots (default: every epoch).
- Stage 1 freezes encoder parameters (controlled via `--encoder-prefix`). Stage 2 unfreezes everything.

### 3.5 `scripts/infer_monai_finetune.py`

**Purpose:** run sliding-window inference with the fine-tuned MONAI model, resample predictions back to the original geometry, and optionally trigger evaluation.

**Run it (single fold):**

```powershell
.\.venv_monai\Scripts\Activate.ps1
python scripts/infer_monai_finetune.py `
    --data-root training_data_additional `
    --dataset-json outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/dataset.json `
    --fold 0 `
    --checkpoint outputs/monai_ft_nnunet_aligned/fold0/checkpoints/stage2_best.pt `
    --output-dir outputs/monai_ft_nnunet_aligned/predictions/fold0 `
    --ground-truth outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    --run-evaluation `
    --evaluation-output outputs/monai_ft_nnunet_aligned/reports_fold0 `
    --amp
```

**Key concepts:**
- Reuses `dataset.json` to discover test case IDs and maintain nnU-Net modality ordering.
- `--bundle-name`/`--bundle-dir` control MONAI bundle loading (defaults to Hugging Face `MONAI/brats_mri_segmentation`).
- The script replaces the bundle’s head with a 5-channel Conv3D to match BraTS labels.

**Important flags:**

| Flag | Description |
|------|-------------|
| `--fold` | Accepts `0`-`4` or `all`. When using `all`, embed `{fold}` placeholders in `--checkpoint` and `--output-dir`. |
| `--spacing`, `--patch-size`, `--roi-size`, `--sw-batch-size`, `--sw-overlap` | Align inference geometry with training choices. |
| `--ground-truth`, `--evaluation-output`, `--run-evaluation` | If provided, the script invokes `scripts/run_full_evaluation.py` on completion. |
| `--amp` | Enables mixed-precision inference. |

**Outputs:**
- NIfTI segmentations saved as `<case>.nii.gz` under the chosen output directory.
- Optional evaluation reports if `--run-evaluation` is set.

**Troubleshooting:**
- Ensure the dataset roots include every case listed in `dataset.json`; missing cases raise `FileNotFoundError`.
- When running `--fold all`, verify both the checkpoint pattern and output directory include `{fold}`.

### 3.6 `scripts/compute_brats_lesion_metrics.py`

**Purpose:** compute lesion-wise Dice and 95% Hausdorff distance for each BraTS sub-region following the 2024 challenge definition (connected components with 26-connectivity).

**Run it:**

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/compute_brats_lesion_metrics.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/monai_ft_nnunet_aligned/predictions/fold0 `
    --output outputs/monai_ft_nnunet_aligned/reports_fold0/lesion_metrics.json `
    --pretty
```

**Inputs:** both directories must contain `.nii.gz` files with consistent case IDs. The script tolerates suffix variations (`-seg`, `_seg`, `_pred`).

**Key options:**

| Flag | Description |
|------|-------------|
| `--labels` | Foreground labels to evaluate (default `1 2 3 4`). |
| `--omit-aggregates` | Skip Tumor Core (TC) and Whole Tumor (WT) aggregates. |
| `--pretty` | Pretty-print the JSON output. |

**Outputs:**
- JSON file containing `cases` (per-case metrics & lesion tables) and `summary` (mean/median/min/max/std for each label).

**Notes:**
- HD95 values for missed/spurious lesions are penalised by the volume diagonal length.
- Great companion when predictions are produced outside nnU-Net (e.g., MONAI fine-tuning).

### 3.7 `scripts/run_full_evaluation.py`

**Purpose:** orchestrate two evaluation passes—nnU-Net’s CLI metrics and the bespoke lesion-wise report—in a single command, emitting a consolidated manifest.

**Run it:**

```powershell
.\.venv_nnunet\Scripts\Activate.ps1
python scripts/run_full_evaluation.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/monai_ft_nnunet_aligned/predictions/fold0 `
    --output-dir outputs/monai_ft_nnunet_aligned/reports/fold0 `
    --pretty
```

**What happens:**
1. Invokes `nnUNetv2_evaluate_simple` (or a custom command via `--nnunet-cmd`) unless `--skip-nnunet` is set.
2. Runs the same lesion-wise computation as `compute_brats_lesion_metrics.py`.
3. Writes three JSONs inside `--output-dir` by default: `nnunet_metrics.json`, `lesion_metrics.json`, and `full_metrics.json`.

**Useful flags:**

| Flag | Description |
|------|-------------|
| `--nnunet-output`, `--lesion-output`, `--combined-output` | Override default file locations. |
| `--skip-nnunet` | Skip the nnU-Net CLI invocation (useful if the binary is unavailable). |
| `--labels`, `--omit-aggregates` | Passed through to the lesion metrics module. |

**Requirements:** `nnUNetv2_evaluate_simple` must be discoverable on `PATH` (activate `.venv_nnunet`). If it’s missing, either install nnU-Net v2 or run with `--skip-nnunet`.

### 3.8 `scripts/visualize_architecture.py`

**Purpose:** parse `plans.json` (generated during nnU-Net planning) and render architecture summaries/diagrams for documentation or presentations.

**Run it:**

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/visualize_architecture.py `
    outputs/nnunet/nnUNet_results/Dataset501_BraTSPostTx/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json `
    --config 3d_fullres `
    --output outputs/nnunet/reports/architecture/3d_fullres_overview.png
```

**Features:**
- Smart path resolution: if you pass a directory, filename, or dataset hint, the script searches known nnU-Net result roots and chooses the best match.
- Prints a tabular summary (encoder/decoder stages, kernels, strides, feature widths) to the console.
- Generates both overview plots and detailed block diagrams; can emit raster (`.png`) and vector (`.svg`) outputs.

**Arguments:**

| Flag | Description |
|------|-------------|
| `--config` | Configuration key under `plans.json` (default `3d_fullres`). |
| `--output` | Path for the overview figure (PNG by default). |
| `--detailed-output` | Path for the detailed block diagram (defaults to `<output>_detailed.png`). |
| `--vector-output`, `--detailed-vector-output` | Vector export paths (SVG/PDF depending on suffix). |

**Dependencies:** matplotlib and pandas (already included via `requirements.txt`). Ensure a GUI-less backend is fine—the script defaults to saving figures rather than showing them.

---

## 4. End-to-end workflows

The sections below stitch the individual CLIs into reproducible pipelines. Start with the nnU-Net workflow (which generates the metadata MONAI reuses), then move on to MONAI fine-tuning. Every step shows a ready-to-run command and a table describing the most relevant options.

### 4.1 nnU-Net benchmarking loop

#### Step 1 – Prepare dataset structure (`scripts/prepare_nnunet_dataset.py`)

```powershell
.\.venv_nnunet\Scripts\Activate.ps1
python scripts/prepare_nnunet_dataset.py `
    --clean `
    --train-sources training_data training_data_additional `
    --test-sources validation_data `
    --dataset-id Dataset501_BraTSPostTx `
    --require-test-labels
```

| Option | Description | When to adjust |
|--------|-------------|----------------|
| `--train-sources DIR...` | Labeled case roots that populate `imagesTr/labelsTr`. | Point to alternate storage or subset by site. |
| `--test-sources DIR...` | Roots for held-out cases copied into `imagesTs/`. | Swap for different validation cohorts. |
| `--dataset-id NAME` | Folder name beneath `nnUNet_raw`. | Use unique IDs per experiment (e.g., `Dataset502_Internal`). |
| `--nnunet-raw PATH` | Overrides the default `outputs/nnunet/nnUNet_raw`. | Custom storage or network share. |
| `--clean` | Removes any existing dataset folder before rebuilding. | Always enable when regenerating to avoid stale cases. |
| `--require-test-labels` | Fails if a test case lacks `-seg`; also creates `labelsTs/`. | Turn off when unlabeled test cases are expected. |

#### Step 2 – Plan & preprocess (`nnUNetv2_plan_and_preprocess`)

```powershell
.\.venv_nnunet\Scripts\Activate.ps1
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity
```

| Option | Description | When to adjust |
|--------|-------------|----------------|
| `-d 501` | Dataset ID created in Step&nbsp;1. | Match any custom `--dataset-id`. |
| `--verify_dataset_integrity` | Performs a thorough check before preprocessing. | Disable for faster reruns when data already validated. |
| `--no_preprocessing` | Skips tensor caching when set to `True`. | Keep default (`False`) unless you want a plan-only dry run. |
| `-c CONFIG` | Limit preprocessing to a subset of configurations (e.g., `3d_fullres`). | Use when storage is tight or only one config is needed. |

> **Artifacts:** This step produces `splits_final.json`, `plans.json`, and cached tensors under `outputs/nnunet/nnUNet_preprocessed/`.

#### Step 3 – Train folds (`nnUNetv2_train`)

```powershell
.\.venv_nnunet\Scripts\Activate.ps1
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres 0 --npz --device cuda
```

| Option | Description | When to adjust |
|--------|-------------|----------------|
| `Dataset501_BraTSPostTx` | Dataset ID. | Match custom ID. |
| `3d_fullres` | Configuration key. | Swap to `2d`, `3d_lowres`, etc., for alternate plans. |
| `0` | Fold index (`0`–`4` or `all`). | Run all folds for ensembling. |
| `--npz` | Saves softmax probabilities for visualization (requires `hiddenlayer`). | Disable to save disk space. |
| `--device` | Target device (`cuda`, `cpu`). | Force CPU training on GPU-less nodes. |
| `--continue_training` | Resume an interrupted run. | Set `True` with a matching checkpoint in place. |

Repeat the command for each fold or replace the final argument with `all` for sequential training.

#### Step 4 – Generate predictions (`nnUNetv2_predict`)

```powershell
.\.venv_nnunet\Scripts\Activate.ps1
nnUNetv2_predict -i outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/imagesTs `
    -o outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres `
    -d Dataset501_BraTSPostTx `
    -c 3d_fullres `
    -f 0 `
    --save_probabilities
```

| Option | Description | When to adjust |
|--------|-------------|----------------|
| `-i PATH` | Input directory containing cases to segment. | Swap to `imagesTr` for cross-validation checks. |
| `-o PATH` | Destination for NIfTI predictions. | Point to experiment-specific folders. |
| `-d NAME` | Dataset ID. | Keep consistent with training. |
| `-c CONFIG` | Configuration key. | Use alternative plans (e.g., `2d`). |
| `-f FOLD...` | Folds to use (space-separated). | Provide multiple folds for ensemble prediction. |
| `--save_probabilities` | Stores per-class softmax maps. | Disable to reduce disk footprint. |

#### Step 5 – Consolidate evaluation (`scripts/run_full_evaluation.py`)

```powershell
.\.venv_nnunet\Scripts\Activate.ps1
python scripts/run_full_evaluation.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres `
    --output-dir outputs/nnunet/reports/metrics/3d_fullres_fold0 `
    --pretty
```

| Option | Description | When to adjust |
|--------|-------------|----------------|
| `ground_truth` (positional) | Directory with reference segmentations. | Use `labelsTr` for training-fold evaluation. |
| `predictions` (positional) | Directory with model outputs. | Swap in MONAI predictions for head-to-head comparisons. |
| `--output-dir PATH` | Folder for the three JSON reports. | Separate per fold or per experiment. |
| `--nnunet-cmd NAME` | Override evaluator executable (default `nnUNetv2_evaluate_simple`). | Point to custom builds or wrapper scripts. |
| `--labels LIST` | Foreground labels to score. | Restrict when certain labels are absent. |
| `--omit-aggregates` | Skip Tumor Core/Whole Tumor metrics. | Enable for strict per-label reporting. |
| `--skip-nnunet` | Skip the nnU-Net CLI call. | Use when only lesion-wise metrics are needed. |

Optional extras after evaluation:
- `python scripts/visualize_architecture.py ...` to export architecture diagrams (`plans.json`).
- Dash apps (`analyze_statistics.py`, `visualize_volumes.py`) for qualitative review.

### 4.2 MONAI fine-tuning pipeline

The MONAI workflow reuses the dataset layout, splits, and evaluation harness from the nnU-Net loop. Ensure Steps&nbsp;1–2 above have been completed so that `Dataset501_BraTSPostTx` and `splits_final.json` are in place.

#### Step 1 – Ensure dataset artifacts (`scripts/prepare_nnunet_dataset.py`)

```powershell
.\.venv_nnunet\Scripts\Activate.ps1
python scripts/prepare_nnunet_dataset.py `
    --train-sources training_data training_data_additional `
    --test-sources validation_data `
    --dataset-id Dataset501_BraTSPostTx
```

| Option | Description | When to adjust |
|--------|-------------|----------------|
| `--train-sources DIR...` | Identical to nnU-Net Step&nbsp;1. | Point at whichever cohorts you plan to fine-tune on. |
| `--test-sources DIR...` | Supplies inference cases for MONAI validation/testing. | Swap or omit if MONAI should ignore a cohort. |
| `--dataset-id NAME` | Drives MONAI path defaults (e.g., `dataset.json`). | Keep aligned with nnU-Net artifacts. |
| `--require-test-labels` | Optional guard when using evaluation during inference. | Enable if you expect labels for all MONAI eval cases. |

> **Note:** If Step&nbsp;1 was already run for nnU-Net, rerun with `--clean` only when the raw data changed.

#### Step 2 – Fine-tune the MONAI bundle (`scripts/train_monai_finetune.py`)

```powershell
.\.venv_monai\Scripts\Activate.ps1
python scripts/train_monai_finetune.py `
    --data-root training_data training_data_additional `
    --split-json outputs/nnunet/nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json `
    --fold all `
    --output-root outputs/monai_ft_nnunet_aligned `
    --stage1-epochs 40 `
    --stage2-epochs 120 `
    --amp
```

| Option | Description | When to adjust |
|--------|-------------|----------------|
| `--data-root DIR...` | BraTS roots containing modalities + `-seg`. | Add more roots to enlarge training data. |
| `--split-json PATH` | nnU-Net folds guaranteeing consistent validation splits. | Replace when experimenting with new fold definitions. |
| `--fold {0-4,all}` | Which fold(s) to train. | Use `all` for five sequential runs, or a single index for rapid iteration. |
| `--output-root PATH` | Parent folder for checkpoints/logs. | Separate per experiment (`outputs/monai_ft_baseline`). |
| `--stage1-epochs`, `--stage2-epochs` | Curriculum length for decoder vs. full fine-tuning. | Shorten for smoke tests or lengthen for more convergence time. |
| `--stage1-lr`, `--stage2-lr` | Learning rates for each stage. | Tune when loss diverges or plateaus. |
| `--batch-size`, `--stage*-accum` | Memory controls. | Lower batch size or raise accumulation on smaller GPUs. |
| `--amp` | Enables mixed precision. | Disable only when debugging numerical issues. |

#### Step 3 – Run MONAI inference (`scripts/infer_monai_finetune.py`)

```powershell
.\.venv_monai\Scripts\Activate.ps1
python scripts/infer_monai_finetune.py `
    --data-root training_data_additional `
    --dataset-json outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/dataset.json `
    --fold 0 `
    --checkpoint outputs/monai_ft_nnunet_aligned/fold0/checkpoints/stage2_best.pt `
    --output-dir outputs/monai_ft_nnunet_aligned/predictions/fold0 `
    --spacing 1.0 1.0 1.0 `
    --roi-size 224 224 144 `
    --amp
```

| Option | Description | When to adjust |
|--------|-------------|----------------|
| `--data-root DIR...` | Directories searched for case folders. | Include multiple roots for combined validation sets. |
| `--dataset-json PATH` | Defines case IDs and modality order (from nnU-Net Step&nbsp;1). | Swap if you maintain multiple dataset IDs. |
| `--fold {index,all}` | Controls checkpoint path templating. | Use `all` with `{fold}` placeholders to iterate automatically. |
| `--checkpoint PATH` | Stage 1 or Stage 2 weights to load. | Point to different epochs or experiments. |
| `--output-dir PATH` | Where predictions land. | Separate per fold or augmentation. |
| `--spacing`, `--patch-size`, `--roi-size`, `--sw-batch-size`, `--sw-overlap` | Control resampling and sliding-window inference. | Align with training settings or adjust for memory. |
| `--run-evaluation` | Automatically call Step&nbsp;4 upon completion. | Omit when you plan to evaluate later. |
| `--ground-truth`, `--evaluation-output` | Only required when `--run-evaluation` is active. | Point to labels and report directories. |
| `--amp` | Mixed-precision inference. | Disable on mismatched GPU drivers. |

#### Step 4 – Aggregate metrics (`scripts/run_full_evaluation.py` or inline evaluation)

If Step&nbsp;3 used `--run-evaluation`, the reports are already generated. Otherwise, run the evaluation manually:

```powershell
.\.venv_nnunet\Scripts\Activate.ps1
python scripts/run_full_evaluation.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/monai_ft_nnunet_aligned/predictions/fold0 `
    --output-dir outputs/monai_ft_nnunet_aligned/reports/fold0 `
    --pretty
```

| Option | Description | When to adjust |
|--------|-------------|----------------|
| `ground_truth` | Directory of reference segmentations. | Swap to `training_data` labels for fold-specific validation. |
| `predictions` | Directory containing MONAI outputs. | Point to alternate experiment folders. |
| `--output-dir PATH` | Report location. | Separate per fold or seed. |
| `--labels`, `--omit-aggregates` | Same semantics as Step&nbsp;5 in nnU-Net loop. | Restrict evaluation scope when comparing partial label sets. |
| `--skip-nnunet` | Avoids calling nnU-Net CLI (MONAI-only metrics). | Use if nnU-Net executables are unavailable. |

#### Step 5 – Visual QA (optional)

- Launch `python scripts/analyze_statistics.py --open-browser` to confirm intensity/label distributions post-finetuning.
- Create presentation assets with `python scripts/visualize_volumes.py --mode static ...` or `--mode gif`.

Following these two workflows keeps nnU-Net and MONAI results aligned, enabling apples-to-apples comparisons and rapid iteration on BraTS post-treatment experiments.

### 4.3 Visualization and QA sprint

- Launch `analyze_statistics.py` to validate intensity and label distributions after new data drops.
- Use `visualize_volumes.py --mode interactive` during QA sessions to review tricky cases with overlays.
- Export static panels or GIFs when preparing slide decks or reports for stakeholders.

---

## 5. Troubleshooting cheat sheet

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError` for modalities/segmentations | Dataset directory missing expected files | Re-stage the offending case; ensure exact naming (`Case-Modality.nii.gz`). |
| `Could not locate 'nnUNetv2_evaluate_simple' on PATH` | nnU-Net environment inactive | Activate `.venv_nnunet` or adjust `PATH`; optionally pass `--skip-nnunet`. |
| `Missing segmentation but test labels were requested` | `--require-test-labels` with incomplete test set | Provide the missing `-seg` files or drop `--require-test-labels`. |
| CUDA not detected in MONAI scripts | Wrong environment / driver mismatch | Confirm `.venv_monai` activation, reinstall CUDA 12.4 wheel, check GPU availability script from `docs/MONAI_guide.md`. |
| Dash apps show empty dataset lists | Wrong `--data-root` or permissions | Verify roots exist and are accessible; defaults expect `training_data*` folders in repo root. |

---

## 6. Related resources

- `docs/usage_guide.md` – canonical quick-start and troubleshooting guide.
- `docs/MONAI_guide.md` – GPU environment tips and fine-tuning CLI reference tables.
- `docs/nnU-Net_guide.md` – full nnU-Net setup, environment variable configuration, and training recipes.
- `tests/test_volume_utils.py` & `tests/test_rendering.py` – unit tests covering slice preparation and rendering helpers.

Stay aligned with these workflows to keep the BraTS toolkit reproducible and easy to share across workstations.
