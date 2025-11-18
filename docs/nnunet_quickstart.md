# nnU-Net Quick Start Guide

Complete CLI workflow for training and evaluating nnU-Net on BraTS data.

---

## Quick Start: Automated Pipeline

**TL;DR:** Run the entire pipeline with a single command:

### Windows (PowerShell)
```powershell
.\scripts\run_nnunet_pipeline.ps1
```

### Linux/macOS (Bash)
```bash
chmod +x scripts/run_nnunet_pipeline.sh  # First time only
./scripts/run_nnunet_pipeline.sh
```

---

### Script Options

**Windows PowerShell:**
```powershell
# See all available options
.\scripts\run_nnunet_pipeline.ps1 -Help

# Run with custom dataset ID
.\scripts\run_nnunet_pipeline.ps1 -DatasetId Dataset502_Custom

# Skip training (use existing model for prediction only)
.\scripts\run_nnunet_pipeline.ps1 -SkipTraining

# Use single GPU
.\scripts\run_nnunet_pipeline.ps1 -GpuIds "0"

# Use CPU instead of GPU
.\scripts\run_nnunet_pipeline.ps1 -Device cpu
```

**Linux/macOS Bash:**
```bash
# See all available options
./scripts/run_nnunet_pipeline.sh --help

# Run with custom dataset ID
./scripts/run_nnunet_pipeline.sh --dataset-id Dataset502_Custom

# Skip training (use existing model for prediction only)
./scripts/run_nnunet_pipeline.sh --skip-training

# Use single GPU
./scripts/run_nnunet_pipeline.sh --gpu-ids "0"

# Use CPU instead of GPU
./scripts/run_nnunet_pipeline.sh --device cpu
```

**Continue reading below for manual step-by-step instructions.**

---

## Manual Step-by-Step Instructions

The following sections provide detailed manual commands for each pipeline step.

## 1. Environment Setup

### Windows (PowerShell)

#### Create and activate virtual environment
```powershell
python -m venv .venv_nnunet
.\.venv_nnunet\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_nnunet.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

#### Create output directories
```powershell
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_results | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\predictions | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\reports\metrics | Out-Null
```

#### Set environment variables
```powershell
$env:nnUNet_raw = "$PWD\outputs\nnunet\nnUNet_raw"
$env:nnUNet_preprocessed = "$PWD\outputs\nnunet\nnUNet_preprocessed"
$env:nnUNet_results = "$PWD\outputs\nnunet\nnUNet_results"

# Make persistent (optional - requires reopening terminal)
setx nnUNet_raw "$PWD\outputs\nnunet\nnUNet_raw"
setx nnUNet_preprocessed "$PWD\outputs\nnunet\nnUNet_preprocessed"
setx nnUNet_results "$PWD\outputs\nnunet\nnUNet_results"
```

### Linux/macOS (Bash)

#### Create and activate virtual environment
```bash
python3 -m venv .venv_nnunet
source .venv_nnunet/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_nnunet.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

#### Create output directories
```bash
mkdir -p outputs/nnunet/nnUNet_raw
mkdir -p outputs/nnunet/nnUNet_preprocessed
mkdir -p outputs/nnunet/nnUNet_results
mkdir -p outputs/nnunet/predictions
mkdir -p outputs/nnunet/reports/metrics
```

#### Set environment variables
```bash
export nnUNet_raw="$(pwd)/outputs/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/outputs/nnunet/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/outputs/nnunet/nnUNet_results"

# Make persistent (add to ~/.bashrc or ~/.zshrc)
echo 'export nnUNet_raw="'$(pwd)'/outputs/nnunet/nnUNet_raw"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="'$(pwd)'/outputs/nnunet/nnUNet_preprocessed"' >> ~/.bashrc
echo 'export nnUNet_results="'$(pwd)'/outputs/nnunet/nnUNet_results"' >> ~/.bashrc
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

### Single GPU Training
**Windows:**
```powershell
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres all --npz --device cuda
```

**Linux/macOS:**
```bash
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres all --npz --device cuda
```

### Multi-GPU Training (Recommended for 2x H100)
**Windows:**
```powershell
# Use both GPUs with DataParallel
$env:CUDA_VISIBLE_DEVICES = "0,1"
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres all --npz --device cuda
```

**Linux/macOS:**
```bash
# Use both GPUs with DataParallel
export CUDA_VISIBLE_DEVICES=0,1
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres all --npz --device cuda
```

**Key parameters:**
- `Dataset501_BraTSPostTx`: Dataset identifier
- `3d_fullres`: Configuration to train
- `all`: Train on all data (no cross-validation)
- `--npz`: Save softmax probabilities (required for evaluation)
- `--device cuda`: Use GPU (change to `cpu` if no GPU available)
- `CUDA_VISIBLE_DEVICES`: Set to `0,1` to use both GPUs

**Notes:**
- Training uses all available training data in a single model
- Multi-GPU training automatically uses DataParallel when multiple GPUs are detected
- With 2x H100 GPUs, training will be significantly faster
- Trains for 1000 epochs by default
- Checkpoints saved every 50 epochs
- To resume interrupted training, add `--c` flag
- To use specific GPU only: `CUDA_VISIBLE_DEVICES=0` (or `1`)

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

### Windows (PowerShell)
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

# 6. Train model on all data (with multi-GPU support)
$env:CUDA_VISIBLE_DEVICES = "0,1"  # Use both H100 GPUs
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

### Linux/macOS (Bash)
```bash
# 1. Setup environment
python3 -m venv .venv_nnunet
source .venv_nnunet/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_nnunet.txt --extra-index-url https://download.pytorch.org/whl/cu124

# 2. Create directories
mkdir -p outputs/nnunet/nnUNet_raw
mkdir -p outputs/nnunet/nnUNet_preprocessed
mkdir -p outputs/nnunet/nnUNet_results
mkdir -p outputs/nnunet/predictions
mkdir -p outputs/nnunet/reports/metrics

# 3. Set environment variables
export nnUNet_raw="$(pwd)/outputs/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/outputs/nnunet/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/outputs/nnunet/nnUNet_results"

# 4. Prepare dataset
python scripts/prepare_nnunet_dataset.py \
    --clean \
    --train-sources training_data \
    --test-sources test_data \
    --dataset-id Dataset501_BraTSPostTx \
    --require-test-labels

# 5. Plan and preprocess
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity --no_pp False --verbose

# 6. Train model on all data (with multi-GPU support)
export CUDA_VISIBLE_DEVICES=0,1  # Use both H100 GPUs
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres all --npz --device cuda

# 7. Generate predictions
nnUNetv2_predict \
    -i outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/imagesTs \
    -o outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres_all \
    -d Dataset501_BraTSPostTx \
    -c 3d_fullres \
    -f all \
    --save_probabilities

# 8. Evaluate predictions
python scripts/run_full_evaluation.py \
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs \
    outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres_all \
    --output-dir outputs/nnunet/reports/metrics/3d_fullres_all \
    --pretty
```

---

## Troubleshooting

**PowerShell script execution blocked (Windows):**
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Bash script permission denied (Linux/macOS):**
```bash
chmod +x scripts/run_nnunet_pipeline.sh
```

**Environment variables not found:**
- **Windows**: Reopen terminal after using `setx`, or manually set using `$env:VAR_NAME = "value"` in current session
- **Linux/macOS**: Run `source ~/.bashrc` (or `~/.zshrc`) after adding to shell config, or manually set using `export VAR_NAME="value"`

**CUDA not detected:**
- Verify GPU driver is installed: `nvidia-smi`
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall with correct CUDA version

**Out of GPU memory during training:**
- Reduce batch size in plans file
- Use single GPU instead of both:
  - Windows: `$env:CUDA_VISIBLE_DEVICES = "0"`
  - Linux/macOS: `export CUDA_VISIBLE_DEVICES=0`
- Train on CPU (slower): `--device cpu`
- Use smaller GPU-friendly configuration if available

**Multi-GPU not working:**
- Verify both GPUs are detected: `nvidia-smi`
- Check PyTorch can see both GPUs: `python -c "import torch; print(torch.cuda.device_count())"`
- Ensure CUDA_VISIBLE_DEVICES is set correctly:
  - Windows: `$env:CUDA_VISIBLE_DEVICES = "0,1"`
  - Linux/macOS: `export CUDA_VISIBLE_DEVICES=0,1`

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
