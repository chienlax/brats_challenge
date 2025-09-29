# BraTS Post-treatment Toolkit – Team Guide

Your one-stop reference for installing the toolkit, staging data, and running every
analysis/visualization workflow that ships with this repository.

---

## Table of contents

- [1. Quick-start checklist](#1-quick-start-checklist)
- [2. Repository layout](#2-repository-layout)
- [3. Prerequisites & environment setup](#3-prerequisites--environment-setup)
- [4. Data acquisition & directory hygiene](#4-data-acquisition--directory-hygiene)
- [5. Core workflow overview](#5-core-workflow-overview)
- [6. Command-line toolkit reference](#6-command-line-toolkit-reference)
  - [6.1 Dataset summary (`summarize_brats_dataset.py`)](#61-dataset-summary-summarize_brats_datasetpy)
  - [6.2 Visualization panels (`visualize_brats.py`)](#62-visualization-panels-visualize_bratspy)
  - [6.3 Per-case histograms (`generate_case_statistics.py`)](#63-per-case-histograms-generate_case_statisticspy)
  - [6.4 GIF generator (`generate_case_gifs.py`)](#64-gif-generator-generate_case_gifspy)
  - [6.5 Volume pre-renderer (`precompute_volume_visuals.py`)](#65-volume-pre-renderer-precompute_volume_visualspy)
  - [6.6 Interactive volume inspector (`launch_volume_inspector.py`)](#66-interactive-volume-inspector-launch_volume_inspectorpy)
- [7. Interactive notebook (`notebooks/brats_exploration.ipynb`)](#7-interactive-notebook-notebooksbrats_explorationipynb)
- [8. Outputs & artifact management](#8-outputs--artifact-management)
- [9. Automation & batching tips](#9-automation--batching-tips)
- [10. Troubleshooting & FAQ](#10-troubleshooting--faq)
- [11. Collaboration workflow](#11-collaboration-workflow)
- [12. Roadmap & contribution guardrails](#12-roadmap--contribution-guardrails)
- [13. Related resources & contacts](#13-related-resources--contacts)

---

## 1. Quick-start checklist

1. **Clone the repo** (code only; data are synced separately).
2. **Create & activate** a virtual environment (see Section 3).
3. **Install dependencies** from `requirements.txt` (matplotlib, nibabel, numpy, pandas, imageio, ipywidgets).
4. **Mirror the imaging data** into `training_data/`, `training_data_additional/`, and `validation_data/`.
5. **Smoke test the setup** by running the dataset summary script and reviewing `outputs/dataset_stats.json`.
6. **Generate visuals** (panels/GIFs/histograms) or explore interactively via the notebook.
7. **Commit only code & docs**—outputs and raw data stay local per policy.

Need a deeper dive? Read on for step-by-step instructions, advanced flags, and maintenance tips.

---

## 2. Repository layout

```
brats_challenge/
├── README.md                       # Executive summary + quick commands
├── requirements.txt                # Python dependencies (pinned versions)
├── docs/
│   ├── usage_guide.md             # ← you are here
│   ├── data_overview.md           # Aggregated dataset statistics & context
│   └── medical_report.md          # Clinical notes (if provided)
├── notebooks/
│   └── brats_exploration.ipynb    # Interactive exploration playground
├── scripts/                       # Command-line utilities
│   ├── summarize_brats_dataset.py
│   ├── visualize_brats.py
│   ├── generate_case_statistics.py
│   └── generate_case_gifs.py
├── outputs/                       # Generated figures/reports (git-ignored)
├── training_data/                 # Optional baseline dataset (git-ignored)
├── training_data_additional/      # BraTS post-treatment releases (git-ignored)
├── validation_data/               # Held-out cases (git-ignored)
└── .venv/                         # Local virtual environment (git-ignored)
```

> **Data reminder:** Imaging directories are intentionally absent from version control.
> Copy them from the secure storage share before running any scripts. If space is
> tight, symlink directories instead of copying.

---

## 3. Prerequisites & environment setup

- **Python:** 3.10+ (tested with CPython 3.11). Use the same interpreter across scripts and the notebook.
- **OS:** Windows, macOS, and modern Linux distros are supported; examples in this guide use Windows PowerShell.
- **Disk:** ~50 GB for datasets plus room for generated artifacts.
- **Optional:** GPU is *not* required. All utilities run comfortably on CPU-only laptops.

### 3.1 Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version   # confirm interpreter
```

Linux/macOS equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

> Tip: Use `Set-ExecutionPolicy -Scope Process Bypass` if PowerShell blocks the activation script.

### 3.2 Install requirements

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install:

```powershell
pip list | Select-String nibabel
```

### 3.3 IDE & notebook add-ons

- Install the **Jupyter** and **Python** extensions in VS Code for best notebook support.
- `ipywidgets` is bundled already; if widget rendering fails, reinstall via `pip install ipywidgets --force-reinstall`.
- For remote/headless servers, set `MPLBACKEND=Agg` or use `--no-overlay`/`--output` flags to avoid GUI requirements.

---

## 4. Data acquisition & directory hygiene

1. **Retrieve the BraTS archives** from the consortium’s secure storage.
- Addisonal training data: [Google Drive](https://drive.google.com/drive/folders/1bBs_z1IvppZqmtUYOM2iJsD-9g5ZkGKC?usp=sharing)
- Validation data: [Google Drive](https://drive.google.com/drive/folders/1uHZ_A4tNVQoLjKkYlh1kMkDJYIWMs9NN?usp=sharing)
2. **Extract** the archives so each case sits in its own folder: `BraTS-GLI-XXXXX-YYY/`.
3. **Place** those folders under `training_data_additional/` (primary cohort). Legacy data may live in `training_data/`; validation drops belong in `validation_data/`.
4. **Verify counts**:
   ```powershell
   (Get-ChildItem training_data_additional -Directory).Count
   ```
   Expect 271 directories for the March 2025 release (see `docs/data_overview.md`).
5. **Spot-check modality files** inside a case (e.g., `BraTS-GLI-02405-100`) to confirm the pattern `CASEID-{t1c,t1n,t2f,t2w,seg}.nii.gz`.
6. **Keep data read-only** when possible to prevent accidental modifications. Scripts never write inside the dataset trees.

> Quality gate: after every new data delivery, run the dataset summary (Section 6.1) and compare the resulting JSON against the previous baseline. Any shape/spacing/label shifts should be investigated immediately.

---

## 5. Core workflow overview

| Step | Goal | Command | Output |
|------|------|---------|--------|
| 1 | Baseline QA | `summarize_brats_dataset.py` | `outputs/dataset_stats.json` summary of cohort geometry/intensity/labels |
| 2 | Case deep-dive | `generate_case_statistics.py` | Histogram PNGs + label table per case |
| 3 | Publication figures | `visualize_brats.py` | High-res modality panels or orthogonal views |
| 4 | Animation reels | `generate_case_gifs.py` | GIFs sweeping through slices |
| 5 | Interactive inspection | `launch_volume_inspector.py` | Dash web app for multi-modal slice browsing with overlays |
| 6 | Interactive exploration | `brats_exploration.ipynb` | Widgets for slicing, analytics, and quick prototyping |

Follow the steps sequentially for onboarding or run them à la carte depending on your task.

---

## 6. Command-line toolkit reference

All scripts reside in `scripts/` and share common characteristics:

- Implemented in pure Python with pinned dependencies from `requirements.txt`.
- Emit concise progress logs (`Processing CASE (i/N)` or `Saved figure to ...`).
- Accept absolute or relative paths; outputs are created on demand.
- Fail fast with informative error messages if files are missing or malformed.

### 6.1 Dataset summary (`summarize_brats_dataset.py`)

**Purpose:**

- Aggregate geometry, spacing, orientation, modality availability, intensity percentiles, segmentation label counts, and dtype frequencies across the cohort.
- Optionally embed per-case detail snapshots for auditing.

**Basic command:**

```powershell
python scripts/summarize_brats_dataset.py --root training_data_additional --output outputs/dataset_stats.json
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `--root PATH` *(required)* | Dataset folder containing case directories. Works with `training_data`, `training_data_additional`, or a bespoke path. |
| `--max-cases N` | Limit processing to the first `N` case directories—useful for smoke tests. |
| `--sample-cases N` | Include detailed per-case blobs for the first `N` subjects in the JSON output. |
| `--output PATH` | Persist JSON to disk. If omitted, the JSON prints to STDOUT. |

**Interpreting the JSON:**

- `aggregate.num_cases` — total processed cases.
- `aggregate.shape_counts`, `aggregate.voxel_spacing_counts` — catch geometry anomalies.
- `aggregate.modality_presence` — confirm all four modalities are present per case.
- `aggregate.intensity_summary[MODALITY]` — average/min/max of p01/p50/p99/mean intensities.
- `aggregate.label_counts` — total voxel counts per segmentation class.
- `aggregate.dtype_counts` — dtype frequency (watch for unexpected float64 volumes).
- `aggregate.orientation_counts` — orientational consistency (expect `(L, A, S)`).

**Performance tips:**

- Full 271-case runs finish in a few minutes on a laptop (≈3–4 GB RAM peak). Use `--max-cases 5` when iterating on code.
- Redirect logs to file (`... | Tee-Object -FilePath logs/summary.log`) for archival.

**Next actions:** Feed the JSON into dashboards or compare new drops with `data_overview.md` metrics.

### 6.2 Visualization panels (`visualize_brats.py`)

**Purpose:**

- Produce publication-ready figures comparing modalities on a chosen plane or orthogonal views for a single modality.
- Overlay segmentation masks with the challenge color legend.

**Modalities layout example:**

```powershell
python scripts/visualize_brats.py --case-dir training_data_additional/BraTS-GLI-02405-100 `
    --layout modalities --axis axial --fraction 0.55 `
    --output outputs/BraTS-GLI-02405-100_modalities.png
```

**Orthogonal layout example:**

```powershell
python scripts/visualize_brats.py --case-dir training_data_additional/BraTS-GLI-02405-100 `
    --layout orthogonal --modality t1c --indices 90 110 70 `
    --output outputs/BraTS-GLI-02405-100_orthogonal.png
```

**Flags to know:**

| Flag | Description |
|------|-------------|
| `--layout {modalities, orthogonal}` | Choose between modality comparison or three-plane layout. |
| `--axis {axial, coronal, sagittal}` | Slice direction for the modalities layout. |
| `--fraction f` | Relative slice position (0–1). Rounded to the nearest slice when `--index` is absent. |
| `--index n` | Absolute slice index overriding `--fraction` for reproducibility. |
| `--indices a b c` | Explicit sagittal, coronal, axial indices for orthogonal layout. |
| `--modality m` | Modality displayed in orthogonal layout (default `t1c`). |
| `--dpi N` | Control output resolution (default 160). Increase for print figures. |
| `--no-overlay` | Disable segmentation overlay (grayscale only). |
| `--show` | Pop up a GUI window (omit on servers). |
| `--output PATH` | Save figure. Directories auto-create as needed. |

**Under the hood:**

- Intensities are normalized per volume using 1st–99th percentiles to dampen outliers.
- Axial slices follow radiological convention (patient left appears on image right).
- Legends list segmentation labels 1–4 with BraTS-compliant colors.

**Batch production:**

- PowerShell: `Get-ChildItem training_data_additional -Directory | ForEach-Object { python scripts/visualize_brats.py --case-dir $_.FullName --layout modalities --fraction 0.5 --output (Join-Path outputs "${($_.Name)}_modalities.png") }`
- Linux: `for d in training_data_additional/*; do python scripts/visualize_brats.py --case-dir "$d" --layout modalities --fraction 0.5 --output outputs/$(basename "$d")_modalities.png; done`

### 6.3 Per-case histograms (`generate_case_statistics.py`)

**Purpose:**

- Quantify modality intensity distributions and segmentation label volumes per case.
- Produce histogram PNGs for visual QA and a JSON report for downstream analytics.

**Example command:**

```powershell
python scripts/generate_case_statistics.py --root training_data_additional --output-dir outputs/stats --max-cases 10
```

**Outputs:**

- `outputs/stats/<CASE>/<CASE>_hist.png` — side-by-side histograms (normalized 0–1 range).
- `outputs/stats/case_statistics.json` — list of objects with voxel counts and label percentages (`voxels_label_N`, `pct_label_N`).

**Useful flags:**

| Flag | Description |
|------|-------------|
| `--case CASE_ID` | Process a single case (skips directory scan). |
| `--bins N` | Histogram bin count (default 100). |
| `--lower-percentile P`, `--upper-percentile Q` | Adjust normalization percentiles; tighten ranges for noisy scans. |
| `--max-cases N` | Limit number of cases when scanning the root. |

**Workflow tips:**

- Inspect histogram PNGs to detect clipping or bimodal distributions before normalization decisions.
- Use the JSON as input for pandas or BI dashboards: `pandas.read_json("outputs/stats/case_statistics.json")`.
- Missing segmentations raise an error—ideal for catching incomplete case folders.

### 6.4 GIF generator (`generate_case_gifs.py`)

**Purpose:**

- Create slice-by-slice animations for QC reviews or clinician storytelling.
- Supports optional segmentation overlays and stride control for length/size trade-offs.

**Example command:**

```powershell
python scripts/generate_case_gifs.py --root training_data_additional --output-dir outputs/gifs --modality t2f --axis axial --step 4 --overlay --max-cases 3
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--case CASE_ID` | Target a single subject. |
| `--modality {t1c,t1n,t2f,t2w}` | Modality to animate (default `t2f`). |
| `--axis {sagittal, coronal, axial}` | Orientation for traversal. |
| `--step N` | Slice stride (higher = shorter GIF). Minimum enforced at 1. |
| `--fps N` | Frames per second (default 12). Lower to slow down playback. |
| `--overlay` | Add segmentation colors (needs `*-seg.nii.gz`). |
| `--dpi N` | Resolution of intermediate frames. |

**Outputs:**

- `outputs/gifs/<CASE>_<MODALITY>_<AXIS>.gif`
- Console logs summarizing progress (`Rendering GIF for CASE (i/N)` and file path).

**Practical advice:**

- Large GIFs (all slices, low stride) can reach 50–100 MB; adjust `--step` or convert to MP4 externally if needed.
- For remote runs, set `matplotlib` to Agg (done automatically in the script).

### 6.5 Volume pre-renderer (`precompute_volume_visuals.py`)

**Purpose:**

- Generate full slice stacks plus Plotly HTML viewers for lightning-fast auditing outside Python.
- Validate segmentation overlays and modality completeness before downstream processing.

**Example command:**

```powershell
python scripts/precompute_volume_visuals.py --case BraTS-GLI-02405-100 --output-dir outputs/volumes --open
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--case CASE_ID` | Render a single case (skips directory scan). |
| `--root PATH` | Batch every case under the folder. |
| `--axis {axial,coronal,sagittal}` | Slice direction (default `axial`). |
| `--step N` | Skip slices to shrink output (default 1). |
| `--overlay/--no-overlay` | Toggle segmentation overlay (on by default). |
| `--output-dir PATH` | Destination directory (auto-created). |
| `--open` | Launch the default viewer when processing a single case. |

**Notes:**

- Output tree: `outputs/volumes/<CASE>/<AXIS>/slice_###.png` plus interactive HTML overlays.
- Pair with `generate_case_gifs.py` to create animations from the cached PNGs.
- For spinning disks, keep `--step` high or run cases sequentially to avoid IO thrash.

### 6.6 Interactive volume inspector (`launch_volume_inspector.py`)

**Purpose:**

- Spin up a Dash web app for real-time modality navigation with segmentation overlays.
- Browse multiple datasets without editing notebooks or saving intermediate files.
- Export the active slice as a PNG straight from your browser.

**Basic launch:**

```powershell
python scripts/launch_volume_inspector.py --open-browser
```

The server binds to `http://127.0.0.1:8050` by default and auto-indexes `training_data`, `training_data_additional`, and `validation_data` when present.

**Key flags:**

| Flag | Description |
|------|-------------|
| `--data-root PATH` | Add extra dataset folders (repeat the flag as needed). |
| `--host HOST` | Bind address; use `0.0.0.0` for LAN sharing. |
| `--port PORT` | Override the default port (`8050`). |
| `--debug` | Enable Dash hot reload + verbose logging. |
| `--open-browser` | Launch the default browser once the server is ready. |

**UI cheat sheet:**

- Dataset + case dropdowns mirror the on-disk directory structure; missing roots are skipped quietly.
- Modality options reflect only the volumes available for the selected case.
- Axis radio buttons and the slice slider stay within voxel bounds—no more index errors.
- View mode selector toggles between a single-modality pane and a 2×2 grid of T1c/T1n/T2w/T2f (when present).
- Display quality toggle offers Standard, High (2×), and Best (4×) upscaling tiers—bump it up when you need extra detail and don’t mind the extra CPU load.
- Segmentation legend sits under the viewer so you always know the colors: blue (Enhancing Tumor), red (Non-Enhancing Tumor), green (Peritumoral Edema/SNFH), yellow (Resection Cavity).
- Segmentation overlay toggle respects BraTS color conventions and auto-disables when masks are absent.
- The orthogonal preview panel maintains sagittal/coronal/axial context around the chosen slice.
- Export downloads a timestamped PNG with the overlay baked in to your browser's download folder.

**Operational tips:**

- Recent cases are cached in memory (LRU, 16 entries); restart the app to clear the cache.
- Chromium-based browsers (Edge/Chrome) deliver smoother slider interaction thanks to WebGL acceleration.
- When exposing beyond localhost, update firewall rules accordingly—the app intentionally has no authentication layer.

---

## 7. Interactive notebook (`notebooks/brats_exploration.ipynb`)

### Highlights

- **Geometry audit:** Table of shapes, voxel spacing, and orientation; flags inconsistencies.
- **Label analytics:** Aggregated label ratios plus per-case breakdowns mirroring the summary script.
- **Slice viewers:** ipywidgets controls for modality, axis, slice position, and overlay toggles.
- **Histogram explorer:** On-demand plots per case with percentile sliders.
- **GIF helper cells:** Functions that wrap the GIF script for ad-hoc experimentation.

### Launching locally

```powershell
jupyter notebook notebooks/brats_exploration.ipynb
```

- Ensure the virtual environment is active before launching.
- For VS Code users, open the notebook directly and select the `.venv` interpreter.
- When running on a server without GUI forwarding, run `jupyter notebook --no-browser --port 8888` and port-forward via SSH.

### Notebook etiquette

- Keep outputs trimmed before committing (Clear All Outputs in VS Code).
- Parameterize paths at the top of the notebook so others can switch datasets easily.
- Use the notebook to prototype changes, then upstream stable logic into the scripts.

---

## 8. Outputs & artifact management

| Location | Contents | Notes |
|----------|----------|-------|
| `outputs/dataset_stats.json` | Latest dataset-wide summary. | Regenerate after every new data drop; archive dated copies if needed. |
| `outputs/stats/` | Histogram PNGs + per-case JSON. | Consider pruning old runs to save space. |
| `outputs/gifs/` | Animated GIFs per case/modality/axis. | Heavy files—store only the versions you actively share. |
| `outputs/volumes/` | Interactive Plotly HTML viewers per case/modality. | Generated by `precompute_volume_visuals.py`; open directly in a browser. |
| `outputs/*.png` | Visualization panels. | Name files using `<CASE>_<layout>.png` for traceability. |

- The `outputs/` folder is ignored by git; feel free to create subdirectories such as `outputs/reports/` or `outputs/presentations/`.
- When sharing artifacts externally, redact PHI and note whether overlays include segmentation labels.

---

## 9. Automation & batching tips

- **Parallelization:** Scripts are single-threaded by design. For large batches, run multiple PowerShell windows or use GNU Parallel (Linux) with care to avoid saturating disk IO.
- **Logging:** Append `| Tee-Object -FilePath logs/<name>.log` in PowerShell or `2>&1 | tee logs/<name>.log` in Bash to capture console output.
- **Case lists:** Process a subset with a text file: `Get-Content cases.txt | ForEach-Object { python scripts/visualize_brats.py --case-dir (Join-Path training_data_additional $_) ... }`.
- **Scheduled runs:** On Windows Task Scheduler or cron, activate the virtual environment within the job (`powershell.exe -ExecutionPolicy Bypass -Command ".\.venv\Scripts\Activate.ps1; ..."`).

---

## 10. Troubleshooting & FAQ

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError: Missing expected file` | Case folder lacks a modality (naming typo, incomplete sync). | Re-sync the case; confirm filenames follow `CASEID-modality.nii.gz`. |
| JSON missing expected keys | Script terminated early (exception, path wrong). | Re-run with `--max-cases` small to reproduce; inspect console output for the failing case. |
| All-zero or washed-out figures | 1st–99th percentile normalization collapsed (almost constant volume). | Use `--index` to pick a slice with anatomy, or adjust percentiles in `generate_case_statistics.py`. |
| `matplotlib` backend errors on server | GUI backend selected by default. | Avoid `--show`; ensure `MPLBACKEND=Agg` (Linux) or rely on saved outputs. |
| `MemoryError` during histograms | Extremely large simultaneous plots. | Reduce `--max-cases`, close other apps, or process cases sequentially. |
| GIF generation is slow | Rendering hundreds of frames at high DPI. | Increase `--step`, lower `--dpi`, or skip overlays. |
| Widget controls missing in notebook | `ipywidgets` not enabled. | Run `pip install ipywidgets --force-reinstall` and `jupyter nbextension enable --py widgetsnbextension`. |
| Orientation legend unexpected | Affine not canonical (rare). | Check `aggregate.orientation_counts` and confirm the case’s affine in nibabel; update `aff2axcodes` logic if needed. |

Need more help? See Section 13 for contact details.

---

## 11. Collaboration workflow

1. **Sync code** from GitHub (`git pull origin main`).
2. **Sync data** from the secure share (rsync/robocopy depending on platform).
3. **Run `summarize_brats_dataset.py`** to validate geometry/labels for the latest drop.
4. **Produce visuals** needed for the sprint (panels, GIFs, histograms).
5. **Document findings** in `docs/` (update `data_overview.md` or add a short report).
6. **Commit/push** code or documentation changes only; data/outputs remain local.
7. **Open a PR** summarizing the change, referencing any generated artifacts or QA steps performed.

> Keep this guide in sync with any new utilities or workflow tweaks. If you add a script, document it in Section 6.

---

## 12. Roadmap & contribution guardrails

- **Interactive dashboards:** Feed `dataset_stats.json` and `case_statistics.json` into Streamlit/Panel dashboards for live QC.
- **Extended analytics:** Explore percentile-based z-score normalization, longitudinal case tracking, and volume trend charts.
- **Media exports:** Add MP4/WEBM support to the GIF script and expose colorbar customization.
- **Automation:** Package scripts into a CLI bundle (e.g., Typer) or Snakemake pipeline for reproducible batch runs.

Contribution tips:

- Pair code changes with documentation updates (this guide, README, or `data_overview.md`).
- Maintain pinned dependency versions; test upgrades in a feature branch.
- Capture major decisions in `docs/` to onboard the next teammate faster.

---

## 13. Related resources & contacts

- **Docs:**
  - `docs/data_overview.md` — dataset statistics snapshot (keep refreshed).
  - `docs/medical_report.md` — clinical context and interpretation guidelines.
- **External references:**
  - BraTS challenge hub: <https://www.synapse.org/#!Synapse:syn51156910/wiki/>
  - NiBabel orientation primer: <https://nipy.org/nibabel/orientation.html>

- **Team contacts:**
  - *Primary maintainer:* Assign at project kickoff (update here once defined).
  - *Shared channel:* `#brats-post-tx` on Slack for Q&A and status updates.

Keep iterating on this guide—thorough documentation keeps onboarding frictionless and lets the team focus on science.
