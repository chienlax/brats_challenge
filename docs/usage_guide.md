# BraTS Post-treatment Toolkit – Usage Guide

Authoritative instructions for setting up the environment, staging BraTS post-treatment data, and operating every tool shipped in this repository.

**Last Updated:** September 30, 2025

---

## Table of contents

- [1. Quick-start checklist](#1-quick-start-checklist)
- [2. Repository map](#2-repository-map)
- [3. Environment setup](#3-environment-setup)
- [4. Data staging & QA hygiene](#4-data-staging--qa-hygiene)
- [5. Workflows & CLI tools](#5-workflows--cli-tools)
	- [5.1 Statistics inspector (`scripts/analyze_statistics.py`)](#51-statistics-inspector-scriptsanalyze_statisticspy)
	- [5.2 Visualization suite (`scripts/visualize_volumes.py`)](#52-visualization-suite-scriptsvisualize_volumespy)
	- [5.3 Lesion-wise metrics (`scripts/compute_brats_lesion_metrics.py`)](#53-lesion-wise-metrics-scriptscompute_brats_lesion_metricspy)
	- [5.4 Legacy scripts (`scripts/deprecated/`)](#54-legacy-scripts-scriptsdeprecated)
	- [5.5 MONAI fine-tuning (`scripts/train_monai_finetune.py`, `scripts/infer_monai_finetune.py`)](#55-monai-fine-tuning-scriptstrain_monai_finetunepy-scriptsinfer_monai_finetunepy)
- [6. Dash app internals](#6-dash-app-internals)
- [7. Outputs, caches, and storage](#7-outputs-caches-and-storage)
- [8. Automation & batching patterns](#8-automation--batching-patterns)
- [9. Testing & validation](#9-testing--validation)
- [10. Troubleshooting cheat sheet](#10-troubleshooting-cheat-sheet)
- [11. Collaboration workflow & guardrails](#11-collaboration-workflow--guardrails)
- [12. Reference materials](#12-reference-materials)

---

## 1. Quick-start checklist

1. **Clone the repository** (code only; datasets are synced separately).
2. **Create and activate** a virtual environment (Section 3).
3. **Install pinned dependencies** from `requirements.txt`.
4. **Mirror the imaging data** into `training_data/`, `training_data_additional/`, or `validation_data/`.
5. **Launch the statistics inspector** to verify dataset health and populate caches.
6. **Run the visualization suite** in the mode you need (interactive/static/GIF).
7. **Keep outputs and raw data local**—only commit code, configs, and documentation.

---

## 2. Repository map

The table below lists every meaningful path and its purpose. Files or folders not listed are either auto-generated (`__pycache__`, `.pytest_cache`) or intentionally ignored (`outputs/`, `training_data*/`).

| Path | Role |
|------|------|
| `README.md` | One-page overview with common commands. |
| `requirements.txt` | Pinned Python dependencies (Python 3.10+). |
| `docs/usage_guide.md` | **This document** – canonical usage instructions. |
| `docs/data_overview.md` | Historical dataset snapshot; refresh after large data drops. |
| `docs/medical_report.md` | Optional clinical context for stakeholders. |
| `apps/__init__.py` | Package marker for the Dash applications. |
| `apps/common/volume_utils.py` | Shared helpers for loading, normalizing, slicing, caching, and labeling BraTS volumes. |
| `apps/volume_inspector/` | Dash app powering the interactive viewer (layout, callbacks, rendering, assets). |
| `apps/statistics_inspector/` | Dash app for dataset/case analytics (repository abstraction, layout, callbacks). |
| `scripts/analyze_statistics.py` | CLI launcher for the statistics inspector Dash app. |
| `scripts/visualize_volumes.py` | Unified CLI for interactive viewing, static figures, and GIF generation. |
| `scripts/deprecated/` | Frozen legacy scripts and migration README for historical reference only. |
| `tests/test_volume_utils.py` | Unit coverage for normalization, slicing, and index helpers. |
| `tests/test_rendering.py` | Unit coverage for Plotly rendering helpers (scaling, overlays). |
| `outputs/` | Git-ignored cache folder for generated artifacts (JSON, PNG, GIF). |
| `training_data*/`, `validation_data/` | Git-ignored raw data roots (symlink or copy from secure storage). |

---

## 3. Environment setup

- **Python:** 3.10 or later (CI-tested with CPython 3.11).
- **OS:** Windows, macOS, and Linux supported; PowerShell snippets are provided below.
- **Disk:** 50 GB+ for datasets plus space for generated GIFs/PNGs/JSON.
- **GPU:** Not required—everything runs comfortably on CPU-only laptops.

### 3.1 Create and activate a virtual environment

```powershell
python -m venv .venv
.\\.venv\Scripts\Activate.ps1
python --version
```

macOS/Linux equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

> **PowerShell tip:** Run `Set-ExecutionPolicy -Scope Process Bypass` if activation scripts are blocked.

### 3.2 Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Optional: create the nnU-Net environment (for evaluation helpers and CLI tooling):

```powershell
python -m venv .venv_nnunet
.\.venv_nnunet\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements_nnunet.txt
```

Optional: create the MONAI fine-tuning environment (for transfer learning scripts):

```powershell
python -m venv .venv_monai
.\.venv_monai\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements_monai.txt
```

Sanity-check a random package:

```powershell
pip list | Select-String dash
```

### 3.3 Recommended IDE configuration

- Install the **Python** and **Jupyter** extensions in VS Code.
- Set the interpreter to `.venv/Scripts/python.exe`.
- For headless servers, export `MPLBACKEND=Agg` or rely on `--output` flags to avoid GUI usage.

---

## 4. Data staging & QA hygiene

1. **Fetch the BraTS archives**.
- Additional training data: [Drive](https://drive.google.com/drive/folders/1bBs_z1IvppZqmtUYOM2iJsD-9g5ZkGKC?usp=drive_link)
- Validation data: [Drive](https://drive.google.com/drive/folders/1uHZ_A4tNVQoLjKkYlh1kMkDJYIWMs9NN?usp=sharing)
2. **Extract** so each case becomes `BraTS-GLI-XXXXX-YYY/` containing `*-{t1c,t1n,t2f,t2w,seg}.nii.gz` files.
3. **Place** folders under `training_data_additional/` (primary set). Legacy cases can live in `training_data/`; validation-only drops go to `validation_data/`.
4. **Verify counts**:
	 ```powershell
	 (Get-ChildItem training_data_additional -Directory).Count
	 ```
	 Expect 270–280 folders depending on the release; record the exact number in `docs/data_overview.md`.
5. **Spot-check one case** to ensure all modalities plus `-seg` are present and spelled correctly.
6. **Keep directories read-only** to avoid accidental edits. Tooling never mutates the dataset roots.

> **Quality gate:** After syncing new data, recompute dataset statistics (Section 5.1) and compare the JSON to prior baselines before green-lighting further work.

---

## 5. Workflows & CLI tools

Two scripts run the entire toolkit. Both respect the same data-root conventions and reuse shared helpers for consistent normalization and color palettes.

### 5.1 Statistics inspector (`scripts/analyze_statistics.py`)

- **What it is:** A Dash app with two tabs—*Case Statistics* and *Dataset Statistics*—replacing the old `generate_case_statistics.py` and `summarize_brats_dataset.py` scripts.
- **Launch:**
	```powershell
	python scripts/analyze_statistics.py --open-browser
	```
- **Key flags:**
	| Flag | Purpose |
	|------|---------|
	| `--data-root PATH` | Repeated flag to add custom dataset directories. Defaults: `training_data`, `training_data_additional`, `validation_data`. |
	| `--host` / `--port` | Network binding (default `127.0.0.1:8050`). Use `--host 0.0.0.0` for LAN demos. |
	| `--debug` | Enables Dash hot reload and verbose logs. |
	| `--open-browser` | Auto-opens the default browser after the server starts. |

#### Case Statistics tab

1. Pick a dataset, then a case ID (dropdown auto-syncs with on-disk contents).
2. Configure histogram bins (10–500, default 100) if you need higher resolution.
3. Click **Compute**.

**Outputs:** Interactive histograms for each available modality (normalized 0–1) and a segmentation label table with voxel counts/percentages. Nothing is persisted to disk; refresh the tab for new runs.

#### Dataset Statistics tab

1. Select the dataset root and optionally set `Max Cases` to limit processing while iterating.
2. Use **Load Cached** to pull from `outputs/statistics_cache/<dataset>_stats.json` if it exists.
3. Use **Recompute** to rebuild the aggregate stats, refreshing the cache on success.

**Metrics collected:**

- Volume shapes, voxel spacing, and orientation counts (affine-derived).
- Modality availability counts and percentages.
- Intensity percentiles (`p01`, `p50`, `p99`, `mean`) aggregated across cases with min/mean/max summaries.
- Voxel totals per segmentation label (0–4) with cohort-wide percentages.
- Under-the-hood dtype distribution for modalities (e.g., `float32` vs `float64`).

**Performance expectations:** A full 270-case run finishes in ~5 minutes on SSD storage. Use the `--max-cases` option for quick smoke tests while developing.

### 5.2 Visualization suite (`scripts/visualize_volumes.py`)

A single CLI handles three visualization scenarios through the `--mode` flag.

#### 5.2.1 Interactive mode (Dash web app)

- **Launch:**
	```powershell
	python scripts/visualize_volumes.py --mode interactive --open-browser
	```
- **Notable flags:** identical to the statistics inspector (`--data-root`, `--host`, `--port`, `--debug`).
- **UI capabilities:**
	- Dataset & case selectors (mirrors disk layout).
	- Modality-aware axis slider with integer marks for `[0, mid, max]` slices.
	- Toggle between *single modality* and *all modalities* view modes.
	- Display quality presets: Standard (×1), High (×2), Best (×4) scaling.
	- Segmentation overlay toggle honoring BraTS color standards.
	- Orthogonal preview rendered from the same volume in single-modality mode.
	- One-click PNG export with timestamped filenames.
- **Caching:** The app memoizes the 16 most recent cases (`volume_utils.cached_volumes`). Restart to clear memory.

#### 5.2.2 Static mode (publication figures)

Generate PNG panels for reports or slide decks.

```powershell
python scripts/visualize_volumes.py --mode static `
		--case-dir training_data_additional/BraTS-GLI-02405-100 `
		--layout modalities --axis axial --fraction 0.55 `
		--output outputs/BraTS-GLI-02405-100_modalities.png
```

Orthogonal alternative:

```powershell
python scripts/visualize_volumes.py --mode static `
		--case-dir training_data_additional/BraTS-GLI-02405-100 `
		--layout orthogonal --modality t1c --indices 90 110 70 `
		--output outputs/BraTS-GLI-02405-100_orthogonal.png
```

| Key argument | Notes |
|--------------|-------|
| `--case-dir` | Required path to the case folder. |
| `--layout` | `modalities` (grid of available modalities) or `orthogonal` (sagittal/coronal/axial). |
| `--axis` / `--fraction` / `--index` | Control slice position for the `modalities` layout. `--index` overrides `--fraction`. |
| `--indices` | Provide explicit `(sagittal coronal axial)` slice indices for orthogonal layout. |
| `--modality` | Select modality for orthogonal layout (default `t1c`). |
| `--no-overlay` | Disable segmentation overlay. |
| `--output` | Required output PNG path; directories are created automatically. |
| `--dpi` | Resolution of the saved figure (default 160). |
| `--show` | Display interactively (omit on servers). |

**Batch example:**

```powershell
Get-ChildItem training_data_additional -Directory | ForEach-Object {
		$case = $_.FullName
		python scripts/visualize_volumes.py --mode static `
				--case-dir $case --layout modalities --fraction 0.5 `
				--output (Join-Path outputs "$(Split-Path $case -Leaf)_modalities.png")
}
```

#### 5.2.3 GIF mode (slice-by-slice animations)

```powershell
python scripts/visualize_volumes.py --mode gif `
		--root training_data_additional --output-dir outputs/gifs `
		--gif-modality t2f --gif-axis axial --step 3 --overlay
```

| Key argument | Notes |
|--------------|-------|
| `--root` | Required dataset directory containing case subfolders. |
| `--case` | Optional single case ID if you do not want to iterate the entire root. |
| `--output-dir` | Required destination directory (auto-created). |
| `--gif-modality` | Modality to animate (`t1c`, `t1n`, `t2f`, `t2w`; default `t2f`). |
| `--gif-axis` | Slice orientation (`sagittal`, `coronal`, `axial`; default `axial`). |
| `--step` | Slice stride (default 2). Increase to shorten GIF length. |
| `--fps` | Playback speed (default 12). |
| `--overlay` | Blend segmentation labels using BraTS colors. |
| `--max-cases` | Process only the first *N* cases when scanning `--root`. |
| `--gif-dpi` | Figure DPI for each frame (default 100). |

Outputs are named `outputs/gifs/<CASE>_<MODALITY>_<AXIS>.gif`. Expect 20–100 MB files depending on stride and DPI.

### 5.3 Lesion-wise metrics (`scripts/compute_brats_lesion_metrics.py`)

This standalone CLI mirrors the official BraTS 2024 evaluation. It scans a folder of
ground-truth segmentations and a folder of model predictions, matches lesions using
26-connectivity, and reports lesion-wise Dice / 95% Hausdorff Distance for the core
sub-regions (ET, NETC, SNFH, RC) plus the aggregate Tumor Core (TC) and Whole Tumor (WT).

```powershell
python scripts/compute_brats_lesion_metrics.py `
	training_data_additional `
	outputs/nnunet/preds_fold0 `
	--output outputs/reports/lesion_metrics.json `
	--pretty
```

| Flag | Purpose |
|------|---------|
| `ground_truth` | Positional argument – directory housing reference segmentations. Files must end with `.nii.gz`. |
| `predictions` | Positional argument – directory with model predictions. Names must mirror the ground truth (e.g., `BraTS-…-seg.nii.gz` or `BraTS-….nii.gz`). |
| `--output PATH` | Destination JSON (directories will be created automatically). |
| `--pretty` | Indents JSON for readability (default is minified). |
| `--labels` | Optional override for foreground labels (default `1 2 3 4`). |
| `--omit-aggregates` | Skip Tumor Core (TC) and Whole Tumor (WT) summaries, reporting only per-label metrics. |

The JSON payload contains per-case metrics, lesion-level breakdown tables, and cohort
summary statistics (mean/median/min/max/std). Missed or spurious detections are
penalised with Dice = 0 and HD95 equal to the volume diagonal to approximate the
challenge scoring rules. Use the lesion tables to audit which component(s) failed.

> **Tip:** Prefer `scripts/run_full_evaluation.py` when you already plan to call
> `nnUNetv2_evaluate_simple`; it orchestrates the nnU-Net CLI and this lesion-wise
> scorer in one step and stores all JSON artifacts alongside each other.

### 5.4 Legacy scripts (`scripts/deprecated/`)

Older commands (`launch_volume_inspector.py`, `visualize_brats.py`, `generate_case_gifs.py`, `generate_case_statistics.py`, `summarize_brats_dataset.py`) are preserved for historical reference in `scripts/deprecated/README.md`. They are **unsupported**; new workflows should use the unified tools above.

### 5.5 MONAI fine-tuning (`scripts/train_monai_finetune.py`, `scripts/infer_monai_finetune.py`)

> **Prerequisites:** Activate `.venv_monai` and install dependencies via
> `pip install -r requirements_monai.txt`. Ensure you also have access to
> `nnUNet_preprocessed/.../splits_final.json` to mirror nnU-Net fold
> memberships.

These scripts implement the staged transfer learning pipeline using MONAI's
`brats_mri_segmentation` bundle as the starting point.

#### 5.5.1 Training (stage 1 & 2)

```powershell
python scripts/train_monai_finetune.py `
		--data-root training_data training_data_additional `
		--split-json nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json `
		--fold 0 `
		--output-root outputs/monai_ft `
		--amp
```

Run this command from the `.venv_monai` environment to ensure MONAI and its
dependencies are available.

Highlights:

- Loads the MONAI bundle, replaces the 3-class head with a 5-class head, and
	caches it under `outputs/monai_ft/checkpoints/`.
- Stage 1 freezes encoder parameters (prefixes configurable via
	`--encoder-prefix`) and trains decoder/head with lr=1e-3 and gradient
	accumulation to accommodate 12 GB VRAM.
- Stage 2 unfreezes everything, drops lr to 1e-5, and continues to refine the
	encoder.
- Augmentations mirror nnU-Net DA5: flips, affine jitter, histogram shifts,
	bias fields, Gaussian noise/smoothing, low-resolution simulation, and coarse
	dropout.
- Validation runs sliding-window inference every `--val-interval` epochs and
	tracks the best Dice score on the fold-0 validation set.
- Training history is exported to `outputs/monai_ft/metrics/training_log.json`.

Key flags:

| Flag | Purpose |
|------|---------|
| `--bundle-name` | Override the MONAI bundle (default `brats_mri_segmentation`). |
| `--stage1-*` / `--stage2-*` | Control epochs, learning rates, and gradient accumulation. |
| `--class-weights` | Set DiceCE weights for ET/NETC/SNFH/RC (background excluded). |
| `--patch-size` / `--spacing` | Tweak patch size and voxel spacing if needed. |
| `--resume` | Resume from a specific checkpoint (stage-specific). |

#### 5.5.2 Inference & evaluation

```powershell
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

Inference should likewise be executed from `.venv_monai`.

Behaviour:

- Re-applies deterministic preprocessing (orientation, spacing, percentile
	scaling, padding) for validation cases.
- Performs sliding-window inference with configurable ROI size, overlap, and
	mixed-precision support.
- Writes predictions as `BraTS-GLI-XXXX-XXX.nii.gz` into the requested output
	directory, matching nnU-Net naming conventions.
- Optional `--run-evaluation` flag reuses `scripts/run_full_evaluation.py` to
	compute nnU-Net + lesion-wise metrics for apples-to-apples comparison.

Additional flags of interest:

| Flag | Purpose |
|------|---------|
| `--roi-size`, `--sw-batch-size`, `--sw-overlap` | Tune sliding-window inference. |
| `--no-download` | Skip bundle download attempts (use cached copy). |
| `--ground-truth`, `--evaluation-output` | Provide evaluation inputs/outputs when using `--run-evaluation`. |
| `--seed` | Control randomness for reproducibility.

---

## 6. Dash app internals

Understanding the architecture helps when extending features or debugging.

- **Shared utilities:** `apps/common/volume_utils.py` centralizes modality labels, segmentation colors, axis handling, slice extraction, normalization (1st–99th percentile), dataset discovery, metadata collection, and a 16-case LRU cache.
- **Volume inspector app (`apps/volume_inspector/`):**
	- `app.py` – Dash factory that wires roots and registers callbacks.
	- `data.py` – `VolumeRepository` indexing dataset roots, cases, and metadata.
	- `layout.py` – Sidebar controls, axis slider, view/quality toggles, orthogonal figure placeholder, segmentation legend.
	- `callbacks.py` – Dropdown linkage, slice updates, figure rendering, and PNG export.
	- `rendering.py` – Plotly figure builders (`create_slice_figure`, `create_modalities_figure`, `create_orthogonal_figure`).
	- `assets/styles.css` – Custom styling for the Dash UI.
- **Statistics inspector app (`apps/statistics_inspector/`):**
	- `data.py` – `StatisticsRepository` for dataset discovery, case summaries, aggregates, and persistent cache management (`outputs/statistics_cache/`).
	- `layout.py` – Bootstrap-based tabs and cards for presenting histograms, tables, and summaries.
	- `callbacks.py` – Handles dropdown population, cache status, dataset recomputes, and error messaging.
	- `app.py` – Dash factory loading Bootstrap theme and wiring callbacks.

---

## 7. Outputs, caches, and storage

| Location | Contents | Refresh strategy |
|----------|----------|------------------|
| `outputs/statistics_cache/` | Dataset-wide JSON caches keyed by dataset name. | Auto-created by the statistics inspector; delete to force full recompute. |
| `outputs/dataset_stats.json` | Legacy single-file summary (optional). | Generated when running historical scripts; can be reproduced via inspector cache. |
| `outputs/gifs/` | GIF animations from visualization suite. | Clean up periodically—files can be large. |
| `outputs/volumes/`, `outputs/stats/` | Optional PNG/HTML exports from scripts or past workflows. | Safe to prune before committing; directory is git-ignored. |

> **Policy reminder:** Never commit `outputs/`, `training_data*/`, or other large artifacts. Git tracks only code, configs, and docs.

---

## 8. Automation & batching patterns

- **Parallel runs:** Tools are single-threaded. If batch throughput is required, launch multiple shells targeting disjoint case subsets. Avoid saturating spinning disks.
- **Logging:** Append `| Tee-Object -FilePath logs/<name>.log` (PowerShell) or `2>&1 | tee logs/<name>.log` (bash) for permanent records.
- **Case allowlists:**
	```powershell
	Get-Content cases.txt | ForEach-Object {
			python scripts/visualize_volumes.py --mode static `
					--case-dir (Join-Path training_data_additional $_) `
					--layout modalities --output (Join-Path outputs "$_.png")
	}
	```
- **Scheduling:** In Task Scheduler or cron, activate the virtual environment inside the job command (`powershell.exe -ExecutionPolicy Bypass -Command "\.venv\Scripts\Activate.ps1; ..."`).

---

## 9. Testing & validation

- **Unit tests:**
	```powershell
	python -m pytest -q
	```
	- `tests/test_volume_utils.py` guards normalization, slice selection, and orientation helpers.
	- `tests/test_rendering.py` validates Plotly rendering scale factors and overlay behavior.
- **Manual smoke:** Run `python scripts/analyze_statistics.py --open-browser` and `python scripts/visualize_volumes.py --mode interactive --open-browser` after installing dependencies to ensure Dash apps start without errors.
- **Pre-release checklist:**
	- Regenerate dataset statistics if data changed; confirm cache JSON values look sane.
	- Render at least one static panel and one GIF to ensure output directories and overlays work.

---

## 10. Troubleshooting cheat sheet

| Symptom | Likely cause | Resolution |
|---------|--------------|------------|
| `FileNotFoundError` for modalities | Missing or misspelled NIfTI files in the case directory. | Re-sync the case folder; ensure filenames follow `<CASE>-<modality>.nii.gz`. |
| Dataset dropdowns are empty | Data roots not found or empty. | Verify paths; pass explicit `--data-root` flags pointing at populated directories. |
| Histograms look flat | Volume contains mostly constant values; normalization clamped. | Inspect raw intensities, try different slice (`--index`), or adjust percentile logic if extending code. |
| PNG export button does nothing | Browser blocked download popup (Safari) or case lacks segmentation. | Allow popups or disable overlay; check console for errors. |
| GIF generation is slow | High DPI or tiny `--step`. | Increase `--step`, reduce `--gif-dpi`, or limit `--max-cases`. |
| Dash app crashes on launch | Missing Dash dependencies. | Re-run `pip install -r requirements.txt`. |
| Orientation looks mirrored | Slice orientation is radiological (patient left appears on image right). | Confirm expected orientation; this is by design and matches BraTS convention. |

---

## 11. Collaboration workflow & guardrails

1. **Sync code** (`git pull origin main`).
2. **Sync data** from secure storage (robocopy/rsync/Drive).
3. **Recompute dataset stats** after each new cohort to populate the cache and validate geometry.
4. **Produce required visuals** with the visualization suite.
5. **Document findings** in `docs/` (update `data_overview.md`, append to this guide if new workflows are introduced).
6. **Run tests** before opening pull requests.
7. **Commit only code/docs**; never add NIfTI files, GIFs, or PNGs.

> **Documentation rule:** Any new flag, mode, or workflow must be reflected here and in `README.md` before merging.

---

## 12. Reference materials

- **Internal docs:**
	- `docs/data_overview.md` – Historical statistics; regenerate after major updates.
	- `docs/medical_report.md` – Clinical interpretation guidelines.
- **External resources:**
	- BraTS challenge hub – <https://www.synapse.org/#!Synapse:syn51156910/wiki/>
	- NiBabel orientation primer – <https://nipy.org/nibabel/orientation.html>
	- Plotly Dash docs – <https://dash.plotly.com/>

Keep this guide authoritative: whenever code, flags, or directory structures change, update the relevant sections immediately.

