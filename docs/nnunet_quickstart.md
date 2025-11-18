# nnU-Net Quick Start Guide

Complete CLI workflow for training and evaluating nnU-Net on BraTS data.

---

## Quick Start: Automated Pipeline

**TL;DR:** Run the entire pipeline with a single command:

```powershell
.\scripts\run_nnunet_pipeline.ps1
```

This script automates all steps below. Use `-Help` to see all options:

```powershell
# See all available options
.\scripts\run_nnunet_pipeline.ps1 -Help

# Run with custom dataset ID
.\scripts\run_nnunet_pipeline.ps1 -DatasetId Dataset502_Custom

# Skip training (use existing model for prediction only)
.\scripts\run_nnunet_pipeline.ps1 -SkipTraining

# Use CPU instead of GPU
.\scripts\run_nnunet_pipeline.ps1 -Device cpu
```

**Continue reading below for manual step-by-step instructions.**

---

## Manual Step-by-Step Instructions

The following sections provide detailed manual commands for each pipeline step.

## 1. Environment Setup

### Create and activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_nnunet.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### Create output directories
```powershell
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_results | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\predictions | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\reports\metrics | Out-Null
```

### Set environment variables
```powershell
$env:nnUNet_raw = "$PWD\outputs\nnunet\nnUNet_raw"
$env:nnUNet_preprocessed = "$PWD\outputs\nnunet\nnUNet_preprocessed"
$env:nnUNet_results = "$PWD\outputs\nnunet\nnUNet_results"

# Make persistent (optional - requires reopening terminal)
setx nnUNet_raw "$PWD\outputs\nnunet\nnUNet_raw"
setx nnUNet_preprocessed "$PWD\outputs\nnunet\nnUNet_preprocessed"
setx nnUNet_results "$PWD\outputs\nnunet\nnUNet_results"
```

---

## 2. Prepare Dataset

```powershell
python scripts/prepare_nnunet_dataset.py `
    --clean `
    --train-sources training_data `
    --test-sources test_data `
    --dataset-id Dataset501_BraTSPostTx `
    --require-test-labels
```

**Key parameters:**
- `--clean`: Remove existing dataset folder before rebuilding
- `--train-sources`: Directory containing training cases
- `--test-sources`: Directory containing test cases
- `--dataset-id`: Unique dataset identifier (e.g., Dataset501, Dataset502)
- `--require-test-labels`: Fail if test cases lack segmentation files

---

## 3. Plan and Preprocess (3D Full Resolution Only)

```powershell
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity --verbose
```

**Key parameters:**
- `-d 501`: Dataset ID (matches number in Dataset501_BraTSPostTx)
- `-c 3d_fullres`: Only preprocess 3D full resolution configuration
- `--verify_dataset_integrity`: Perform sanity checks
- `--no_pp False`: Enable preprocessing (do not skip)
- `--verbose`: Show detailed output

---

## 4. Train Model (All Data)

Train a single model using all training data:

```powershell
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres all --npz --device cuda
```

**Key parameters:**
- `Dataset501_BraTSPostTx`: Dataset identifier
- `3d_fullres`: Configuration to train
- `all`: Train on all data (no cross-validation)
- `--npz`: Save softmax probabilities (required for evaluation)
- `--device cuda`: Use GPU (change to `cpu` if no GPU available)

**Notes:**
- Training uses all available training data in a single model
- Trains for 1000 epochs by default
- Checkpoints saved every 50 epochs
- To resume interrupted training, add `--c` flag
- Use `CUDA_VISIBLE_DEVICES=X` to select specific GPU

---

## 5. Generate Predictions

Predict using the trained model:

```powershell
nnUNetv2_predict `
    -i outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/imagesTs `
    -o outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres_all `
    -d Dataset501_BraTSPostTx `
    -c 3d_fullres `
    -f all `
    --save_probabilities
```

**Key parameters:**
- `-i`: Input directory with test images
- `-o`: Output directory for predictions
- `-d`: Dataset identifier
- `-c`: Configuration used
- `-f all`: Use the model trained on all data
- `--save_probabilities`: Save probability maps (optional)

**Notes:**
- The model trained on all data must be completed before running prediction
- Single model inference (no ensembling)

---

## 6. Evaluate Predictions

```powershell
python scripts/run_full_evaluation.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres_all `
    --output-dir outputs/nnunet/reports/metrics/3d_fullres_all `
    --pretty
```

**Key parameters:**
- First argument: Ground truth segmentation directory
- Second argument: Prediction directory
- `--output-dir`: Where to save evaluation results
- `--pretty`: Format JSON output for readability

**Output files:**
- `nnunet_metrics.json`: Standard nnU-Net Dice and surface distance metrics
- `lesion_metrics.json`: Lesion-wise detection and segmentation metrics
- `combined_metrics.json`: All metrics combined

**Optional parameters:**
- `--labels 1 2 3`: Specify which labels to evaluate
- `--omit-aggregates`: Skip Tumor Core and Whole Tumor metrics
- `--skip-nnunet`: Skip nnU-Net evaluation (lesion metrics only)

---

## Complete Workflow (Copy-Paste Ready)

```powershell
# 1. Setup environment
python -m venv .venv_nnunet
.\.venv_nnunet\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_nnunet.txt --extra-index-url https://download.pytorch.org/whl/cu124

# 2. Create directories
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_results | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\predictions | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\reports\metrics | Out-Null

# 3. Set environment variables
$env:nnUNet_raw = "$PWD\outputs\nnunet\nnUNet_raw"
$env:nnUNet_preprocessed = "$PWD\outputs\nnunet\nnUNet_preprocessed"
$env:nnUNet_results = "$PWD\outputs\nnunet\nnUNet_results"

# 4. Prepare dataset
python scripts/prepare_nnunet_dataset.py `
    --clean `
    --train-sources training_data `
    --test-sources test_data `
    --dataset-id Dataset501_BraTSPostTx `
    --require-test-labels

# 5. Plan and preprocess
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity --no_pp False --verbose

# 6. Train model on all data
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres all --npz --device cuda

# 7. Generate predictions
nnUNetv2_predict `
    -i outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/imagesTs `
    -o outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres_all `
    -d Dataset501_BraTSPostTx `
    -c 3d_fullres `
    -f all `
    --save_probabilities

# 8. Evaluate predictions
python scripts/run_full_evaluation.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres_all `
    --output-dir outputs/nnunet/reports/metrics/3d_fullres_all `
    --pretty
```

---

## Troubleshooting

**PowerShell script execution blocked:**
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Environment variables not found:**
- Reopen terminal after using `setx`
- Or manually set using `$env:VAR_NAME = "value"` in current session

**CUDA not detected:**
- Verify GPU driver is installed
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall with correct CUDA version

**Out of GPU memory during training:**
- Reduce batch size in plans file
- Train on CPU (slower): `--device cpu`
- Use smaller GPU-friendly configuration if available

**Missing test labels error:**
- Remove `--require-test-labels` flag if test set is unlabeled
- Or ensure all test cases have `-seg.nii.gz` files

---

## Expected Output Locations

After completion, you'll find:

```
outputs/nnunet/
├── nnUNet_raw/Dataset501_BraTSPostTx/
│   ├── imagesTr/          # Training images
│   ├── labelsTr/          # Training labels
│   ├── imagesTs/          # Test images
│   ├── labelsTs/          # Test labels
│   └── dataset.json       # Dataset metadata
├── nnUNet_preprocessed/Dataset501_BraTSPostTx/
│   └── nnUNetPlans_3d_fullres/  # Preprocessed data
├── nnUNet_results/Dataset501_BraTSPostTx/
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
│       └── fold_all/      # Model checkpoints, validation results
├── predictions/Dataset501_BraTSPostTx/
│   └── 3d_fullres_all/    # Predictions (.nii.gz)
└── reports/metrics/
    └── 3d_fullres_all/
        ├── nnunet_metrics.json
        ├── lesion_metrics.json
        └── combined_metrics.json
```

---

**Last Updated:** November 18, 2025
