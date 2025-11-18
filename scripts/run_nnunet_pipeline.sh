#!/bin/bash
# nnU-Net Complete Pipeline Automation Script (Linux/macOS)
# Usage: ./scripts/run_nnunet_pipeline.sh [options]
#
# This script automates the complete nnU-Net workflow:
# 1. Environment setup
# 2. Dataset preparation
# 3. Planning and preprocessing
# 4. Training on all data
# 5. Prediction
# 6. Evaluation

set -e  # Exit on error

# Default parameters
DATASET_ID="Dataset501_BraTSPostTx"
TRAIN_SOURCES="training_data"
TEST_SOURCES="test_data"
SKIP_ENV_SETUP=false
SKIP_DATA_PREP=false
SKIP_PREPROCESS=false
SKIP_TRAINING=false
SKIP_PREDICTION=false
SKIP_EVALUATION=false
DEVICE="cuda"
GPU_IDS="0,1"

# Color output functions
print_step() {
    echo -e "\n\033[0;36m=== $1 ===\033[0m"
}

print_success() {
    echo -e "\033[0;32m✓ $1\033[0m"
}

print_error() {
    echo -e "\033[0;31m✗ $1\033[0m"
}

print_info() {
    echo -e "\033[0;33mℹ $1\033[0m"
}

# Display help
show_help() {
    cat << EOF
nnU-Net Complete Pipeline Automation Script (Linux/macOS)

Usage: ./scripts/run_nnunet_pipeline.sh [options]

Options:
  --dataset-id <string>      Dataset identifier (default: Dataset501_BraTSPostTx)
  --train-sources <string>   Training data directory (default: training_data)
  --test-sources <string>    Test data directory (default: test_data)
  --device <string>          Device to use: cuda or cpu (default: cuda)
  --gpu-ids <string>         GPU IDs to use, comma-separated (default: "0,1" for dual H100)
  --skip-env-setup           Skip environment setup
  --skip-data-prep           Skip dataset preparation
  --skip-preprocess          Skip planning and preprocessing
  --skip-training            Skip model training
  --skip-prediction          Skip prediction generation
  --skip-evaluation          Skip evaluation
  -h, --help                 Show this help message

Examples:
  # Run complete pipeline with dual H100 GPUs
  ./scripts/run_nnunet_pipeline.sh

  # Run with custom dataset ID
  ./scripts/run_nnunet_pipeline.sh --dataset-id Dataset502_Custom

  # Use single GPU only
  ./scripts/run_nnunet_pipeline.sh --gpu-ids "0"

  # Skip training (use existing model)
  ./scripts/run_nnunet_pipeline.sh --skip-training

  # Use CPU instead of GPU
  ./scripts/run_nnunet_pipeline.sh --device cpu
EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-id)
            DATASET_ID="$2"
            shift 2
            ;;
        --train-sources)
            TRAIN_SOURCES="$2"
            shift 2
            ;;
        --test-sources)
            TEST_SOURCES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --skip-env-setup)
            SKIP_ENV_SETUP=true
            shift
            ;;
        --skip-data-prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-prediction)
            SKIP_PREDICTION=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Extract dataset number from DatasetId
DATASET_NUMBER=$(echo "$DATASET_ID" | sed -E 's/Dataset([0-9]+)_.*/\1/')

# Get repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Define paths
VENV_PATH="$REPO_ROOT/.venv_nnunet"
OUTPUTS_DIR="$REPO_ROOT/outputs/nnunet"
RAW_DIR="$OUTPUTS_DIR/nnUNet_raw"
PREPROCESSED_DIR="$OUTPUTS_DIR/nnUNet_preprocessed"
RESULTS_DIR="$OUTPUTS_DIR/nnUNet_results"
PREDICTIONS_DIR="$OUTPUTS_DIR/predictions/$DATASET_ID/3d_fullres_all"
METRICS_DIR="$OUTPUTS_DIR/reports/metrics/3d_fullres_all"

# Display configuration
cat << EOF

╔════════════════════════════════════════════════════════════════╗
║         nnU-Net Complete Pipeline Automation Script            ║
╚════════════════════════════════════════════════════════════════╝

Configuration:
  Dataset ID:      $DATASET_ID
  Dataset Number:  $DATASET_NUMBER
  Train Sources:   $TRAIN_SOURCES
  Test Sources:    $TEST_SOURCES
  Device:          $DEVICE
  GPU IDs:         $([ "$DEVICE" = "cuda" ] && echo "$GPU_IDS" || echo "N/A (CPU mode)")
  Training:        Single model on all data

Pipeline Steps:
  Environment Setup:    $([ "$SKIP_ENV_SETUP" = true ] && echo "SKIP" || echo "RUN")
  Dataset Preparation:  $([ "$SKIP_DATA_PREP" = true ] && echo "SKIP" || echo "RUN")
  Preprocessing:        $([ "$SKIP_PREPROCESS" = true ] && echo "SKIP" || echo "RUN")
  Training:             $([ "$SKIP_TRAINING" = true ] && echo "SKIP" || echo "RUN")
  Prediction:           $([ "$SKIP_PREDICTION" = true ] && echo "SKIP" || echo "RUN")
  Evaluation:           $([ "$SKIP_EVALUATION" = true ] && echo "SKIP" || echo "RUN")

EOF

# Confirm execution
read -p "Do you want to proceed? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Pipeline execution cancelled."
    exit 0
fi

# Start timing
START_TIME=$(date +%s)

# ============================================================================
# STEP 1: Environment Setup
# ============================================================================
if [ "$SKIP_ENV_SETUP" = false ]; then
    print_step "Step 1: Environment Setup"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_PATH" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate environment and install dependencies
    print_info "Activating environment and installing dependencies..."
    source "$VENV_PATH/bin/activate"
    python -m pip install --upgrade pip --quiet
    pip install -r "$REPO_ROOT/requirements_nnunet.txt" --extra-index-url https://download.pytorch.org/whl/cu124 --quiet
    print_success "Dependencies installed"
    
    # Create output directories
    print_info "Creating output directories..."
    mkdir -p "$RAW_DIR"
    mkdir -p "$PREPROCESSED_DIR"
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$(dirname "$PREDICTIONS_DIR")"
    mkdir -p "$METRICS_DIR"
    print_success "Output directories created"
    
    # Set environment variables
    print_info "Setting environment variables..."
    export nnUNet_raw="$RAW_DIR"
    export nnUNet_preprocessed="$PREPROCESSED_DIR"
    export nnUNet_results="$RESULTS_DIR"
    print_success "Environment variables set"
    
else
    print_step "Step 1: Environment Setup (SKIPPED)"
    # Still need to activate environment and set variables
    source "$VENV_PATH/bin/activate"
    export nnUNet_raw="$RAW_DIR"
    export nnUNet_preprocessed="$PREPROCESSED_DIR"
    export nnUNet_results="$RESULTS_DIR"
fi

# ============================================================================
# STEP 2: Dataset Preparation
# ============================================================================
if [ "$SKIP_DATA_PREP" = false ]; then
    print_step "Step 2: Dataset Preparation"
    
    print_info "Preparing nnU-Net dataset structure..."
    python "$REPO_ROOT/scripts/prepare_nnunet_dataset.py" \
        --clean \
        --train-sources "$TRAIN_SOURCES" \
        --test-sources "$TEST_SOURCES" \
        --dataset-id "$DATASET_ID" \
        --require-test-labels
    
    print_success "Dataset prepared successfully"
else
    print_step "Step 2: Dataset Preparation (SKIPPED)"
fi

# ============================================================================
# STEP 3: Planning and Preprocessing
# ============================================================================
if [ "$SKIP_PREPROCESS" = false ]; then
    print_step "Step 3: Planning and Preprocessing"
    
    print_info "Running nnUNetv2_plan_and_preprocess..."
    nnUNetv2_plan_and_preprocess -d "$DATASET_NUMBER" -c 3d_fullres --verify_dataset_integrity --verbose
    
    print_success "Planning and preprocessing completed"
else
    print_step "Step 3: Planning and Preprocessing (SKIPPED)"
fi

# ============================================================================
# STEP 4: Training Model on All Data
# ============================================================================
if [ "$SKIP_TRAINING" = false ]; then
    print_step "Step 4: Training Model on All Data"
    
    # Set GPU configuration
    if [ "$DEVICE" = "cuda" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
        print_info "Using $GPU_COUNT GPU(s): $GPU_IDS"
    fi
    
    print_info "Training single model on all training data..."
    TRAINING_START=$(date +%s)
    
    nnUNetv2_train "$DATASET_ID" 3d_fullres all --npz -device "$DEVICE"
    
    TRAINING_END=$(date +%s)
    TRAINING_DURATION=$((TRAINING_END - TRAINING_START))
    TRAINING_HOURS=$((TRAINING_DURATION / 3600))
    TRAINING_MINUTES=$(((TRAINING_DURATION % 3600) / 60))
    TRAINING_SECONDS=$((TRAINING_DURATION % 60))
    
    print_success "Training completed in $(printf "%02d:%02d:%02d" $TRAINING_HOURS $TRAINING_MINUTES $TRAINING_SECONDS)"
else
    print_step "Step 4: Training Model on All Data (SKIPPED)"
fi

# ============================================================================
# STEP 5: Generate Predictions
# ============================================================================
if [ "$SKIP_PREDICTION" = false ]; then
    print_step "Step 5: Generate Predictions"
    
    INPUT_DIR="$RAW_DIR/$DATASET_ID/imagesTs"
    
    if [ ! -d "$INPUT_DIR" ]; then
        print_error "Input directory not found: $INPUT_DIR"
        exit 1
    fi
    
    print_info "Generating predictions with model trained on all data..."
    
    nnUNetv2_predict \
        -i "$INPUT_DIR" \
        -o "$PREDICTIONS_DIR" \
        -d "$DATASET_ID" \
        -c 3d_fullres \
        -f all \
        --save_probabilities
    
    print_success "Predictions generated"
else
    print_step "Step 5: Generate Predictions (SKIPPED)"
fi

# ============================================================================
# STEP 6: Evaluation
# ============================================================================
if [ "$SKIP_EVALUATION" = false ]; then
    print_step "Step 6: Evaluation"
    
    GROUND_TRUTH_DIR="$RAW_DIR/$DATASET_ID/labelsTs"
    
    if [ ! -d "$GROUND_TRUTH_DIR" ]; then
        print_error "Ground truth directory not found: $GROUND_TRUTH_DIR"
        exit 1
    fi
    
    if [ ! -d "$PREDICTIONS_DIR" ]; then
        print_error "Predictions directory not found: $PREDICTIONS_DIR"
        exit 1
    fi
    
    print_info "Running full evaluation..."
    python "$REPO_ROOT/scripts/run_full_evaluation.py" \
        "$GROUND_TRUTH_DIR" \
        "$PREDICTIONS_DIR" \
        --output-dir "$METRICS_DIR" \
        --pretty
    
    print_success "Evaluation completed"
    
    # Display results summary
    print_step "Results Summary"
    
    COMBINED_METRICS="$METRICS_DIR/combined_metrics.json"
    if [ -f "$COMBINED_METRICS" ]; then
        print_info "Metrics saved to: $METRICS_DIR"
        echo -e "\nMetric files generated:"
        echo "  - nnunet_metrics.json    (Dice, HD95, NSD)"
        echo "  - lesion_metrics.json    (Detection, F1, PQ)"
        echo "  - combined_metrics.json  (All metrics)"
    fi
else
    print_step "Step 6: Evaluation (SKIPPED)"
fi

# ============================================================================
# Pipeline Complete
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

cat << EOF

╔════════════════════════════════════════════════════════════════╗
║                   Pipeline Completed Successfully!              ║
╚════════════════════════════════════════════════════════════════╝

Total execution time: $(printf "%02d:%02d:%02d" $TOTAL_HOURS $TOTAL_MINUTES $TOTAL_SECONDS)

Output locations:
  Preprocessed data:  $PREPROCESSED_DIR
  Trained models:     $RESULTS_DIR
  Predictions:        $PREDICTIONS_DIR
  Evaluation metrics: $METRICS_DIR

EOF

echo -e "\033[0;33mTo view detailed metrics, check:\033[0m"
echo -e "  $METRICS_DIR/combined_metrics.json"

exit 0
