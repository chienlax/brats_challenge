# nnU-Net Complete Pipeline Automation Script
# Usage: .\scripts\run_nnunet_pipeline.ps1 [options]
#
# This script automates the complete nnU-Net workflow:
# 1. Environment setup
# 2. Dataset preparation
# 3. Planning and preprocessing
# 4. Training all 5 folds
# 5. Ensemble prediction
# 6. Evaluation

param(
    [string]$DatasetId = "Dataset501_BraTSPostTx",
    [string]$TrainSources = "training_data",
    [string]$TestSources = "test_data",
    [switch]$SkipEnvSetup,
    [switch]$SkipDataPrep,
    [switch]$SkipPreprocess,
    [switch]$SkipTraining,
    [switch]$SkipPrediction,
    [switch]$SkipEvaluation,
    [string]$Device = "cuda",
    [int]$StartFold = 0,
    [int]$EndFold = 4,
    [switch]$Help
)

# Display help
if ($Help) {
    Write-Host @"
nnU-Net Complete Pipeline Automation Script

Usage: .\scripts\run_nnunet_pipeline.ps1 [options]

Options:
  -DatasetId <string>      Dataset identifier (default: Dataset501_BraTSPostTx)
  -TrainSources <string>   Training data directory (default: training_data)
  -TestSources <string>    Test data directory (default: test_data)
  -Device <string>         Device to use: cuda or cpu (default: cuda)
  -StartFold <int>         Starting fold for training (default: 0)
  -EndFold <int>           Ending fold for training (default: 4)
  -SkipEnvSetup           Skip environment setup
  -SkipDataPrep           Skip dataset preparation
  -SkipPreprocess         Skip planning and preprocessing
  -SkipTraining           Skip model training
  -SkipPrediction         Skip prediction generation
  -SkipEvaluation         Skip evaluation
  -Help                   Show this help message

Examples:
  # Run complete pipeline
  .\scripts\run_nnunet_pipeline.ps1

  # Run with custom dataset ID
  .\scripts\run_nnunet_pipeline.ps1 -DatasetId Dataset502_Custom

  # Skip training (use existing models)
  .\scripts\run_nnunet_pipeline.ps1 -SkipTraining

  # Train only folds 0-2
  .\scripts\run_nnunet_pipeline.ps1 -StartFold 0 -EndFold 2

  # Use CPU instead of GPU
  .\scripts\run_nnunet_pipeline.ps1 -Device cpu
"@
    exit 0
}

# Color output functions
function Write-Step {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Yellow
}

# Extract dataset number from DatasetId
$DatasetNumber = $DatasetId -replace 'Dataset(\d+)_.*', '$1'

# Get repository root
$RepoRoot = Split-Path -Parent $PSScriptRoot

# Define paths
$VenvPath = Join-Path $RepoRoot ".venv_nnunet"
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
$OutputsDir = Join-Path $RepoRoot "outputs\nnunet"
$RawDir = Join-Path $OutputsDir "nnUNet_raw"
$PreprocessedDir = Join-Path $OutputsDir "nnUNet_preprocessed"
$ResultsDir = Join-Path $OutputsDir "nnUNet_results"
$PredictionsDir = Join-Path $OutputsDir "predictions\$DatasetId\3d_fullres_ensemble"
$MetricsDir = Join-Path $OutputsDir "reports\metrics\3d_fullres_ensemble"

Write-Host @"

╔════════════════════════════════════════════════════════════════╗
║         nnU-Net Complete Pipeline Automation Script            ║
╚════════════════════════════════════════════════════════════════╝

Configuration:
  Dataset ID:      $DatasetId
  Dataset Number:  $DatasetNumber
  Train Sources:   $TrainSources
  Test Sources:    $TestSources
  Device:          $Device
  Training Folds:  $StartFold to $EndFold

Pipeline Steps:
  Environment Setup:    $(if ($SkipEnvSetup) { "SKIP" } else { "RUN" })
  Dataset Preparation:  $(if ($SkipDataPrep) { "SKIP" } else { "RUN" })
  Preprocessing:        $(if ($SkipPreprocess) { "SKIP" } else { "RUN" })
  Training:             $(if ($SkipTraining) { "SKIP" } else { "RUN" })
  Prediction:           $(if ($SkipPrediction) { "SKIP" } else { "RUN" })
  Evaluation:           $(if ($SkipEvaluation) { "SKIP" } else { "RUN" })

"@ -ForegroundColor White

# Confirm execution
$confirm = Read-Host "Do you want to proceed? (y/n)"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Info "Pipeline execution cancelled."
    exit 0
}

# Start timing
$StartTime = Get-Date

# ============================================================================
# STEP 1: Environment Setup
# ============================================================================
if (-not $SkipEnvSetup) {
    Write-Step "Step 1: Environment Setup"
    
    # Create virtual environment if it doesn't exist
    if (-not (Test-Path $VenvPath)) {
        Write-Info "Creating virtual environment..."
        python -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Failed to create virtual environment"
            exit 1
        }
        Write-Success "Virtual environment created"
    } else {
        Write-Info "Virtual environment already exists"
    }
    
    # Activate environment and install dependencies
    Write-Info "Activating environment and installing dependencies..."
    & $ActivateScript
    python -m pip install --upgrade pip --quiet
    pip install -r (Join-Path $RepoRoot "requirements_nnunet.txt") --extra-index-url https://download.pytorch.org/whl/cu124 --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to install dependencies"
        exit 1
    }
    Write-Success "Dependencies installed"
    
    # Create output directories
    Write-Info "Creating output directories..."
    New-Item -ItemType Directory -Force $RawDir | Out-Null
    New-Item -ItemType Directory -Force $PreprocessedDir | Out-Null
    New-Item -ItemType Directory -Force $ResultsDir | Out-Null
    New-Item -ItemType Directory -Force (Split-Path $PredictionsDir) | Out-Null
    New-Item -ItemType Directory -Force $MetricsDir | Out-Null
    Write-Success "Output directories created"
    
    # Set environment variables
    Write-Info "Setting environment variables..."
    $env:nnUNet_raw = $RawDir
    $env:nnUNet_preprocessed = $PreprocessedDir
    $env:nnUNet_results = $ResultsDir
    Write-Success "Environment variables set"
    
} else {
    Write-Step "Step 1: Environment Setup (SKIPPED)"
    # Still need to activate environment and set variables
    & $ActivateScript
    $env:nnUNet_raw = $RawDir
    $env:nnUNet_preprocessed = $PreprocessedDir
    $env:nnUNet_results = $ResultsDir
}

# ============================================================================
# STEP 2: Dataset Preparation
# ============================================================================
if (-not $SkipDataPrep) {
    Write-Step "Step 2: Dataset Preparation"
    
    Write-Info "Preparing nnU-Net dataset structure..."
    python (Join-Path $RepoRoot "scripts\prepare_nnunet_dataset.py") `
        --clean `
        --train-sources $TrainSources `
        --test-sources $TestSources `
        --dataset-id $DatasetId `
        --require-test-labels
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Dataset preparation failed"
        exit 1
    }
    Write-Success "Dataset prepared successfully"
} else {
    Write-Step "Step 2: Dataset Preparation (SKIPPED)"
}

# ============================================================================
# STEP 3: Planning and Preprocessing
# ============================================================================
if (-not $SkipPreprocess) {
    Write-Step "Step 3: Planning and Preprocessing"
    
    Write-Info "Running nnUNetv2_plan_and_preprocess..."
    nnUNetv2_plan_and_preprocess -d $DatasetNumber -c 3d_fullres --verify_dataset_integrity --verbose
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Planning and preprocessing failed"
        exit 1
    }
    Write-Success "Planning and preprocessing completed"
} else {
    Write-Step "Step 3: Planning and Preprocessing (SKIPPED)"
}

# ============================================================================
# STEP 4: Training All Folds
# ============================================================================
if (-not $SkipTraining) {
    Write-Step "Step 4: Training All Folds ($StartFold to $EndFold)"
    
    for ($fold = $StartFold; $fold -le $EndFold; $fold++) {
        Write-Info "Training fold $fold..."
        $FoldStartTime = Get-Date
        
        nnUNetv2_train $DatasetId 3d_fullres $fold --npz --device $Device
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Training failed for fold $fold"
            exit 1
        }
        
        $FoldDuration = (Get-Date) - $FoldStartTime
        Write-Success "Fold $fold completed in $($FoldDuration.ToString('hh\:mm\:ss'))"
    }
    
    Write-Success "All folds training completed"
} else {
    Write-Step "Step 4: Training All Folds (SKIPPED)"
}

# ============================================================================
# STEP 5: Generate Ensemble Predictions
# ============================================================================
if (-not $SkipPrediction) {
    Write-Step "Step 5: Generate Ensemble Predictions"
    
    $InputDir = Join-Path $RawDir "$DatasetId\imagesTs"
    
    if (-not (Test-Path $InputDir)) {
        Write-Error-Custom "Input directory not found: $InputDir"
        exit 1
    }
    
    Write-Info "Generating predictions with ensemble of folds $StartFold to $EndFold..."
    $FoldArgs = @()
    for ($fold = $StartFold; $fold -le $EndFold; $fold++) {
        $FoldArgs += $fold
    }
    
    nnUNetv2_predict `
        -i $InputDir `
        -o $PredictionsDir `
        -d $DatasetId `
        -c 3d_fullres `
        -f $FoldArgs `
        --save_probabilities
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Prediction generation failed"
        exit 1
    }
    Write-Success "Ensemble predictions generated"
} else {
    Write-Step "Step 5: Generate Ensemble Predictions (SKIPPED)"
}

# ============================================================================
# STEP 6: Evaluation
# ============================================================================
if (-not $SkipEvaluation) {
    Write-Step "Step 6: Evaluation"
    
    $GroundTruthDir = Join-Path $RawDir "$DatasetId\labelsTs"
    
    if (-not (Test-Path $GroundTruthDir)) {
        Write-Error-Custom "Ground truth directory not found: $GroundTruthDir"
        exit 1
    }
    
    if (-not (Test-Path $PredictionsDir)) {
        Write-Error-Custom "Predictions directory not found: $PredictionsDir"
        exit 1
    }
    
    Write-Info "Running full evaluation..."
    python (Join-Path $RepoRoot "scripts\run_full_evaluation.py") `
        $GroundTruthDir `
        $PredictionsDir `
        --output-dir $MetricsDir `
        --pretty
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Evaluation failed"
        exit 1
    }
    Write-Success "Evaluation completed"
    
    # Display results summary
    Write-Step "Results Summary"
    
    $CombinedMetrics = Join-Path $MetricsDir "combined_metrics.json"
    if (Test-Path $CombinedMetrics) {
        Write-Info "Metrics saved to: $MetricsDir"
        Write-Host "`nMetric files generated:" -ForegroundColor White
        Write-Host "  - nnunet_metrics.json    (Dice, HD95, NSD)"
        Write-Host "  - lesion_metrics.json    (Detection, F1, PQ)"
        Write-Host "  - combined_metrics.json  (All metrics)"
    }
} else {
    Write-Step "Step 6: Evaluation (SKIPPED)"
}

# ============================================================================
# Pipeline Complete
# ============================================================================
$TotalDuration = (Get-Date) - $StartTime

Write-Host @"

╔════════════════════════════════════════════════════════════════╗
║                   Pipeline Completed Successfully!              ║
╚════════════════════════════════════════════════════════════════╝

Total execution time: $($TotalDuration.ToString('hh\:mm\:ss'))

Output locations:
  Preprocessed data:  $PreprocessedDir
  Trained models:     $ResultsDir
  Predictions:        $PredictionsDir
  Evaluation metrics: $MetricsDir

"@ -ForegroundColor Green

Write-Host "To view detailed metrics, check:" -ForegroundColor Yellow
Write-Host "  $MetricsDir\combined_metrics.json" -ForegroundColor White

exit 0
