# BraTS Post-treatment

**For detailed setup instruction, go to: [Usage Guide Document](docs/comprehensive_usage.md)**

## Setup

Create and activate the Dash/reporting environment (`.venv/`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For nnU-Net tooling use the dedicated `.venv_nnunet/` environment:

```powershell
python -m venv .venv_nnunet
.\.venv_nnunet\Scripts\Activate.ps1
pip install -r requirements_nnunet.txt
```

For MONAI fine-tuning create a separate `.venv_monai/` environment:

```powershell
python -m venv .venv_monai
.\.venv_monai\Scripts\Activate.ps1
pip install -r requirements_monai.txt `
    --extra-index-url https://download.pytorch.org/whl/cu124
```

> **GPU tip:** Adjust the `--extra-index-url` to match your CUDA toolkit (for
> example `cu118`, `cu121`, etc.). Omit the flag for CPU-only installs.

## MONAI Fine-Tuning Workflow

We provide end-to-end scripts for the two-stage transfer learning strategy:

1. **Training** – run the staged fine-tuning loop.
2. **Inference** – generate fold-specific validation predictions.
3. **Evaluation** – reuse the unified `run_full_evaluation.py` helper to obtain
     Dice, HD95, and lesion-wise metrics aligned with nnU-Net reports.

### Stage 1 & 2 Training *(run from `.venv_monai`)*

```
python scripts/train_monai_finetune.py `
    --data-root training_data training_data_additional `
    --split-json nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json `
    --fold 0 `
    --output-root outputs/monai_ft `
    --amp
```

Key behaviours:

- Loads the MONAI `brats_mri_segmentation` bundle (downloading if necessary).
- Replaces the 3-class head with a 5-class head (BG, ET, NETC, SNFH, RC).
- Stage 1 freezes encoder blocks and trains decoder/head at lr=1e-3.
- Stage 2 unfreezes everything and continues with lr=1e-5.
- Augmentations mirror nnU-Net DA5 (affine jitter, histogram/intensity noise,
    low-resolution simulation, coarse dropout) while remaining 12 GB friendly.
- Best checkpoints are written to `outputs/monai_ft/checkpoints/` and a
    training log is captured in JSON.

### Validation Inference & Evaluation *(run from `.venv_monai`)*

```
python scripts/infer_monai_finetune.py `
    --data-root training_data training_data_additional `
    --split-json nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json `
    --fold 0 `
    --checkpoint outputs/monai_ft/checkpoints/stage2_best.pt `
    --output-dir outputs/monai_ft/predictions/fold0 `
    --amp `
    --run-evaluation `
    --ground-truth training_data `
    --evaluation-output outputs/monai_ft/reports/fold0
```

The inference helper:

- Reapplies the deterministic preprocessing chain and sliding-window inference.
- Saves predictions as `BraTS-GLI-XXXX-XXX.nii.gz` in the requested output
    directory, matching nnU-Net naming conventions.
- Optionally invokes `scripts/run_full_evaluation.py` to compute Dice, HD95, and
    lesion-wise metrics alongside the nnU-Net baseline for a direct comparison.

## Statistics Inspector

Interactive web application for analyzing BraTS case and dataset statistics:

```powershell
python scripts/analyze_statistics.py --open-browser
```

**Features:**

- **Case Statistics Tab**: View intensity histograms and segmentation label volumes for individual cases
- **Dataset Statistics Tab**: Analyze aggregate metrics across entire datasets (shapes, spacings, modalities, intensity summaries, label counts)
- **Smart Caching**: Dataset statistics are pre-computed and cached in `outputs/statistics_cache/`
- **Recompute Option**: Refresh statistics when data changes with a single click

The statistics inspector replaces the previous `generate_case_statistics.py` and `summarize_brats_dataset.py` scripts with a unified browser-based interface.

## Unified Visualization Tool

A single comprehensive tool for all BraTS volume visualization needs with three distinct modes:

### Interactive Mode (Dash Web App)

Launch a browser-based app for real-time slice browsing with segmentation overlays:

```powershell
python scripts/visualize_volumes.py --mode interactive --open-browser
```

Key features: dataset/case dropdowns, modality-aware axis slider, segmentation toggles, orthogonal previews, and one-click PNG export. Toggle between Standard, High (2×), or Best (4×) display quality. View all modalities side-by-side in grid mode.

### Static Mode (Publication Figures)

Generate multi-modal panels or orthogonal planes with segmentation overlays:

```powershell
python scripts/visualize_volumes.py --mode static \
    --case-dir training_data_additional/BraTS-GLI-02405-100 \
    --layout modalities --axis axial --fraction 0.55 \
    --output outputs/BraTS-GLI-02405-100_axial.png
```

Orthogonal views for a single modality:

```powershell
python scripts/visualize_volumes.py --mode static \
    --case-dir training_data_additional/BraTS-GLI-02405-100 \
    --layout orthogonal --modality t1c --indices 90 110 70 \
    --output outputs/BraTS-GLI-02405-100_orthogonal.png
```

**Options:**
- `--layout`: `modalities` or `orthogonal`
- `--axis`: axis for modalities layout (`axial`, `coronal`, `sagittal`)
- `--fraction` / `--index`: control slice location
- `--indices`: specify slice indices for orthogonal layout
- `--no-overlay`: disable segmentation overlay
- `--show`: open interactive matplotlib window

### GIF Mode (Animations)

Create slice-by-slice animations for review meetings:

```powershell
python scripts/visualize_volumes.py --mode gif \
    --root training_data_additional --output-dir outputs/gifs \
    --gif-modality t2f --gif-axis axial --step 3 --overlay --max-cases 2
```

GIFs are written to `outputs/gifs/CASE_modality_axis.gif`.

**Note:** The previous separate scripts (`launch_volume_inspector.py`, `visualize_brats.py`, `generate_case_gifs.py`) have been consolidated into this unified tool and moved to `scripts/deprecated/`.

## Outputs directory

Figures and JSON reports are written to the `outputs/` folder (created on demand).
Common subdirectories include:

- `outputs/gifs/` — animated slice reels.
- `outputs/statistics_cache/` — cached dataset-wide statistics.
- `outputs/volumes/` — Plotly HTML viewers generated by `precompute_volume_visuals.py`.
- `outputs/nnunet/reports/metrics/` — evaluation summaries produced by the nnU-Net helpers.

## nnU-Net evaluation helper

After running `nnUNetv2_predict`, consolidate the standard nnU-Net metrics and the
BraTS lesion-wise report with a single command:

```powershell
python scripts/run_full_evaluation.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres `
    --output-dir outputs/nnunet/reports/metrics/3d_fullres
```

By default the script invokes `nnUNetv2_evaluate_simple` and then generates the
lesion-wise Dice / HD95 tables. All artifacts are written to the chosen
`--output-dir`, yielding:

- `nnunet_metrics.json` — raw nnU-Net CLI output
- `lesion_metrics.json` — lesion-wise breakdown per case and cohort summary
- `full_metrics.json` — combined manifest referencing both reports

Use `--skip-nnunet` when running inside an environment without nnU-Net and
`--omit-aggregates` to disable the Tumor Core / Whole Tumor aggregates.
