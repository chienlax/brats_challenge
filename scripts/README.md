# Scripts Directory

This directory contains utility scripts for the BraTS Challenge project, organized into subdirectories by functionality.

## Directory Structure

```
scripts/
├── models/              # Model-related scripts
├── visualization/       # Visualization and analysis scripts
└── deprecated/         # Legacy scripts (kept for reference)
```

## Models (`scripts/models/`)

Scripts for model training, evaluation, and dataset preparation:

- **`prepare_nnunet_dataset.py`** - Convert BraTS data to nnU-Net v2 format
  ```bash
  python scripts/models/prepare_nnunet_dataset.py --train-sources training_data
  ```

- **`inspect_bundle.py`** - Inspect MONAI bundle architecture
  ```bash
  python scripts/models/inspect_bundle.py
  ```

- **`visualize_architecture.py`** - Visualize nnU-Net architecture from plans.json
  ```bash
  python scripts/models/visualize_architecture.py <plans.json> --config 3d_fullres
  ```

- **`compute_brats_lesion_metrics.py`** - Compute lesion-wise Dice and HD95 metrics
  ```bash
  python scripts/models/compute_brats_lesion_metrics.py <ground_truth_dir> <predictions_dir>
  ```

- **`run_full_evaluation.py`** - Run complete evaluation (nnU-Net + lesion metrics)
  ```bash
  python scripts/models/run_full_evaluation.py <ground_truth_dir> <predictions_dir> --output-dir outputs/reports
  ```

## Visualization (`scripts/visualization/`)

Scripts for interactive and static visualization:

- **`visualize_volumes.py`** - Unified visualization tool with multiple modes
  ```bash
  # Interactive mode (Dash app)
  python scripts/visualization/visualize_volumes.py --mode interactive --open-browser
  
  # Static image mode
  python scripts/visualization/visualize_volumes.py --mode static --case-dir <case_dir> --output image.png
  
  # GIF animation mode
  python scripts/visualization/visualize_volumes.py --mode gif --root <data_root> --output-dir outputs/gifs
  ```

- **`analyze_statistics.py`** - Launch interactive statistics inspector
  ```bash
  python scripts/visualization/analyze_statistics.py --open-browser
  ```

- **`visualize.ipynb`** - Jupyter notebook for architecture diagram generation

## Deprecated Scripts (`scripts/deprecated/`)

Legacy scripts kept for historical reference. See `scripts/deprecated/README.md` for details.

## Usage Notes

### Running Scripts from Repository Root

All scripts are designed to be run from the repository root directory:

```bash
# From repository root
python scripts/models/prepare_nnunet_dataset.py [args]
python scripts/visualization/visualize_volumes.py [args]
```

### Import Path Handling

Scripts automatically add the repository root to Python's import path, so imports from the `apps` module work correctly regardless of the current working directory.

### Python Environment

Ensure you have activated the appropriate Python environment before running scripts:

```bash
# For nnU-Net scripts
conda activate nnunet_env

# For MONAI/visualization scripts
conda activate monai_env
```
