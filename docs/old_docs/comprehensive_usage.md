# BraTS Post-treatment Toolkit – Comprehensive CLI Usage

**Last Updated:** October 5, 202

---

## 4. End-to-end workflows

### 4.1. Before you start

#### 4.1.1 Repository layout refresher

- Source code lives under `apps/` (Dash apps) and `scripts/` (CLI utilities).
- Raw BraTS cases belong in `training_data/`, `test_data/`, or `validation_data/`; keep them out of Git.
- Generated assets go into `outputs/` (ignored by Git) to preserve reproducibility.

#### 4.1.2 Python environments

- **Core tooling, visualization, statistics**  
    Interpreter: `.venv\Scripts\python.exe`  
    Commands (PowerShell):

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

- **nnU-Net dataset prep & evaluation**  
    Interpreter: `.venv\Scripts\python.exe`  
    Commands (PowerShell):

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements_nnunet.txt --extra-index-url https://download.pytorch.org/whl/cu124
    ```

> **Tip:** Always re-run the appropriate `Activate.ps1` when you open a new shell. If PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

#### 4.1.3 Data staging checklist

1. Extract each BraTS case into its own folder (`BraTS-GLI-xxxxx-yyy/`).
2. Confirm the five canonical files exist: `-t1c`, `-t1n`, `-t2f`, `-t2w`, `-seg` (all `.nii.gz`).
3. Mirror the folders into your chosen dataset roots (`training_data*/` for training, `validation_data/` for hold-outs).
4. Keep directories read-only; scripts never mutate raw data.

#### 4.1.4 Environment variables for nnU-Net

When preparing datasets or evaluating nnU-Net results, make sure the storage directories exist and are exported:

```powershell
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_results | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\reports\metrics | Out-Null

setx nnUNet_raw "${PWD}\outputs\nnunet\nnUNet_raw"
setx nnUNet_preprocessed "${PWD}\outputs\nnunet\nnUNet_preprocessed"
setx nnUNet_results "${PWD}\outputs\nnunet\nnUNet_results"
```

### 4.2 nnU-Net benchmarking loop

#### Step 1 – Prepare dataset structure (`scripts/prepare_nnunet_dataset.py`)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/prepare_nnunet_dataset.py `
    --clean `
    --train-sources training_data `
    --test-sources test_data `
    --dataset-id Dataset501_BraTSPostTx `
    --require-test-labels
```

- `--train-sources DIR...`: labeled case roots that populate `imagesTr/labelsTr`; adjust when you need alternate storage locations or want to subset by site.
- `--test-sources DIR...`: roots for held-out cases copied into `imagesTs/`; change this when targeting different validation cohorts.
- `--dataset-id NAME`: folder name beneath `nnUNet_raw`; pick unique IDs for each experiment (for example `Dataset502_Internal`).
- `--nnunet-raw PATH`: overrides the default `outputs/nnunet/nnUNet_raw`; use when the raw data lives elsewhere (network share, external drive, etc.).
- `--clean`: removes any existing dataset folder before rebuilding; enable whenever you regenerate the dataset to avoid stale cases.
- `--require-test-labels`: fails if a test case lacks `-seg` and also creates `labelsTs/`; disable only when you expect unlabeled test cases.

#### Step 2 – Plan & preprocess (`nnUNetv2_plan_and_preprocess`)

```powershell
.\.venv\Scripts\Activate.ps1
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity --no_pp False --verbose
```

- `-d 501`: dataset ID created in Step&nbsp;1; change it whenever you use a custom `--dataset-id` during preparation.
- `--verify_dataset_integrity`: performs a thorough sanity check before preprocessing; disable only for faster reruns when the data was already validated.
- `--no_preprocessing`: skips tensor caching when set to `True`; keep the default (`False`) unless you just want a plan-only dry run.
- `-c CONFIG`: limits preprocessing to a subset of configurations (for example `3d_fullres`); enable when storage is tight or only one configuration is required.

> **Artifacts:** This step produces `splits_final.json`, `plans.json`, and cached tensors under `outputs/nnunet/nnUNet_preprocessed/`.

#### Step 3 – Train folds (`nnUNetv2_train`)

```powershell
.\.venv\Scripts\Activate.ps1
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres 0 --npz --device cuda
```

- `Dataset501_BraTSPostTx`: dataset ID; align with any custom ID you used during preparation.
- `3d_fullres`: configuration key; swap to `2d`, `3d_lowres`, or another plan when exploring alternatives.
- `0`: fold index (`0`–`4` or `all`); run all folds when you need an ensemble.
- `--npz`: saves softmax probabilities for visualization (requires `hiddenlayer`); disable to conserve disk space.
- `--device`: target device (`cuda` or `cpu`); force CPU training when GPUs are unavailable.
- `--continue_training`: resumes an interrupted run; set to `True` when a matching checkpoint is present.

Repeat the command for each fold or replace the final argument with `all` for sequential training.

#### Step 4 – Generate predictions (`nnUNetv2_predict`)

```powershell
.\.venv\Scripts\Activate.ps1
nnUNetv2_predict -i outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/imagesTs `
    -o outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres `
    -d Dataset501_BraTSPostTx `
    -c 3d_fullres `
    -f 0 `
    --save_probabilities
```

- `-i PATH`: input directory containing cases to segment; swap to `imagesTr` when performing cross-validation checks.
- `-o PATH`: destination for NIfTI predictions; point to experiment-specific folders to keep results organized.
- `-d NAME`: dataset ID; keep it consistent with whatever you used for training.
- `-c CONFIG`: configuration key; choose alternative plans such as `2d` when needed.
- `-f FOLD...`: folds to use (space-separated); supply multiple folds to generate ensemble predictions.
- `--save_probabilities`: stores per-class softmax maps; disable when you need to trim disk usage.

#### Step 5 – Consolidate evaluation (`scripts/run_full_evaluation.py`)

```powershell
.\.venv_nnunet\Scripts\Activate.ps1
python scripts/run_full_evaluation.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres `
    --output-dir outputs/nnunet/reports/metrics/3d_fullres_fold0 `
    --pretty
```

- `ground_truth` (positional): directory with reference segmentations; point to `labelsTr` when evaluating training folds.
- `predictions` (positional): directory containing model outputs; swap in MONAI predictions for head-to-head comparisons.
- `--output-dir PATH`: folder for the three JSON reports; separate directories per fold or per experiment keep results tidy.
- `--nnunet-cmd NAME`: overrides the evaluator executable (default `nnUNetv2_evaluate_simple`); use custom builds or wrapper scripts as needed.
- `--labels LIST`: foreground labels to score; restrict the list when certain labels are absent.
- `--omit-aggregates`: skips Tumor Core and Whole Tumor metrics; enable for strict per-label reporting only.
- `--skip-nnunet`: skips the nnU-Net CLI call; rely on it when you only need lesion-wise metrics.

Optional extras after evaluation:
- `python scripts/visualize_architecture.py ...` to export architecture diagrams (`plans.json`).
- Dash apps (`analyze_statistics.py`, `visualize_volumes.py`) for qualitative review.

### 4.3 Visualization and QA sprint

- Launch `analyze_statistics.py` to validate intensity and label distributions after new data drops.
- Use `visualize_volumes.py --mode interactive` during QA sessions to review tricky cases with overlays.
- Export static panels or GIFs when preparing slide decks or reports for stakeholders.

---

## 5. Troubleshooting cheat sheet

- **Symptom:** `FileNotFoundError` for modalities or segmentations
    - Likely cause: dataset directory is missing the expected files.
    - Fix: re-stage the offending case and ensure file naming matches `Case-Modality.nii.gz` exactly.
- **Symptom:** `Could not locate 'nnUNetv2_evaluate_simple' on PATH`
    - Likely cause: the nnU-Net environment isn’t active.
    - Fix: activate `.venv_nnunet` or adjust `PATH`; as a fallback, run evaluation with `--skip-nnunet`.
- **Symptom:** “Missing segmentation but test labels were requested”
    - Likely cause: `--require-test-labels` is set while the test set is incomplete.
    - Fix: provide the missing `-seg` files or remove the `--require-test-labels` guard.
- **Symptom:** CUDA not detected in MONAI scripts
    - Likely cause: wrong environment or GPU driver mismatch.
    - Fix: confirm `.venv_monai` is active, reinstall the CUDA 12.4 wheel if needed, and run the GPU availability helper in `docs/MONAI_guide.md`.
- **Symptom:** Dash apps show empty dataset lists
    - Likely cause: incorrect `--data-root` or permission issues.
    - Fix: verify the roots exist and are readable; the defaults expect `training_data*` folders in the repository root.

