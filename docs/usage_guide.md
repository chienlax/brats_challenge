# BraTS Post-treatment Toolkit – Team Guide

This document explains the project structure, configuration steps, and how to run
our analysis/visualization utilities so that every teammate can get productive quickly.

---

## 1. Repository layout

```
brats_challenge/
├── README.md               # Quick-start overview
├── requirements.txt        # Python dependencies (pinned)
├── scripts/
│   ├── summarize_brats_dataset.py
│   └── visualize_brats.py
├── docs/
│   └── usage_guide.md      # ← you are here
├── outputs/                # Generated figures/reports (ignored by git)
├── training_data*/         # Large MRI datasets (ignored by git)
└── .venv/                  # Project virtual environment (ignored by git)
```

> **Note:** All imaging data live under `training_data/`, `training_data_additional/`,
> and `validation_data/`. These directories are excluded from version control via
> `.gitignore`; you must source them from the shared storage location before running the scripts.

---

## 2. Environment and configuration

1. **Create/activate the virtual environment (Windows PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **(Optional) GPU/accelerated plotting:** The current toolkit is CPU-only. If we
   integrate PyTorch/monai later, expand `requirements.txt` accordingly.

4. **Environment variables:** None required today. Future enhancements (e.g., storing
   dataset paths or output buckets) should be wired through a `.env` file and surfaced
   in this guide.

---

## 3. Dataset expectations

- Each case folder follows the BraTS convention: `CASE-ID/CASE-ID-{modality}.nii.gz`.
- Modalities available: `t1c`, `t1n`, `t2f`, `t2w` plus `seg` for labels.
- All volumes should be `182 × 218 × 182` voxels with 1 mm spacing.
- Scripts assume NIfTI files with valid affine matrices; malformed data will raise
  informative exceptions (file not found, dtype mismatch, etc.).

### Data quality checks

Run the dataset summary script (Section 4) to confirm geometry consistency,
modalities present, and segmentation label ranges after receiving new data drops.

---

## 4. Dataset summary script (`summarize_brats_dataset.py`)

### Purpose
- Scan the dataset once and capture aggregate stats (shapes, voxel spacing, intensity
  percentiles, label distributions, modality presence, dtype breakdown).
- Export a JSON report for downstream dashboards or QA review.

### Basic usage
```powershell
python scripts/summarize_brats_dataset.py --root training_data_additional --output outputs/dataset_stats.json
```

### Important flags
| Flag | Description |
|------|-------------|
| `--root` (required) | Path to the dataset folder. |
| `--max-cases N` | Process only the first `N` cases (quick smoke test). |
| `--sample-cases N` | Embed detailed per-case summaries for the first `N` subjects in the report. |
| `--output PATH` | Persist results to `PATH`; prints to STDOUT if omitted. |

### Output interpretation
- `aggregate.num_cases` – total subjects processed.
- `shape_counts` / `voxel_spacing_counts` – detect geometry anomalies.
- `modality_presence` – confirm no modalities are missing.
- `intensity_summary` – mean/min/max of the percentile distributions per modality.
- `label_counts` – global voxel counts per segmentation class.
- `dtype_counts` – track float32 vs float64 volumes (helpful for normalization policies).

The script logs progress (`Processed CASE (i/N)`) to track long runs. Expect the full
271-case scan to take a few minutes on standard hardware.

---

## 5. Visualization script (`visualize_brats.py`)

### Purpose
- Generate publication-ready panels comparing modalities on a chosen slice.
- Produce orthogonal views for a single modality.
- Optionally overlay segmentation masks with a consistent color legend.

### Modalities layout example
```powershell
python scripts/visualize_brats.py --case-dir training_data_additional/BraTS-GLI-02405-100 `
    --layout modalities --axis axial --fraction 0.55 `
    --output outputs/BraTS-GLI-02405-100_modalities.png
```

### Orthogonal layout example
```powershell
python scripts/visualize_brats.py --case-dir training_data_additional/BraTS-GLI-02405-100 `
    --layout orthogonal --modality t1c --indices 90 110 70 `
    --output outputs/BraTS-GLI-02405-100_orthogonal.png
```

### Key options
| Flag | Description |
|------|-------------|
| `--layout {modalities, orthogonal}` | Select panel style. |
| `--axis {axial, coronal, sagittal}` | Slice plane for modalities layout. |
| `--fraction f` | Relative slice position (0–1). |
| `--index n` | Absolute slice index (overrides `--fraction`). |
| `--indices a b c` | Slice indices for `orthogonal` layout (sagittal, coronal, axial). |
| `--modality m` | Modality to visualize in orthogonal layout. |
| `--no-overlay` | Disable segmentation overlay. |
| `--show` | Display interactively (for local exploration). |
| `--output PATH` | Save figure (directories auto-created). |

### Implementation details
- Intensities are normalized per volume using the 1st–99th percentile range, reducing
  outlier impact without clipping clinically relevant signal.
- Segmentation overlays use a transparent colormap aligned with BraTS post-treatment
  labels (Enhancing, Non-Enhancing, Edema, Resection cavity).
- Axial slices are rotated to follow radiological convention (patient left on image right).

---

## 6. Outputs and logging

- All generated figures/JSON land in `outputs/` (ignored by git). Organize subfolders
  (e.g., `outputs/figures`, `outputs/reports`) as needed.
- Scripts print concise progress text; redirect to log files if running in batch mode.

---

## 7. Troubleshooting

| Issue | Resolution |
|-------|------------|
| `FileNotFoundError` for modalities | Confirm dataset path and that the case folder contains expected NIfTI filenames. |
| All-zero figures | Check for invalid normalization (rare if volume contains mostly zeros). Try `--index` to target slices with visible anatomy. |
| Matplotlib backend errors | When running headless (e.g., remote server), omit `--show` and rely on `--output`. |
| Memory limits | Scripts load one volume at a time; if memory issues occur, ensure 64-bit Python and sufficient RAM. |

---

## 8. Collaboration workflow

1. Pull the repository (without data) from GitHub.
2. Obtain imaging data from the secure server and place them in the ignored folders.
3. Install requirements and run the summary script to validate the drop.
4. Generate visualization figures for QC or presentations.
5. Commit only code/docs; outputs and raw data stay local.

---

## 9. Upcoming enhancements (roadmap)

- **Interactive notebooks:** Build Jupyter notebooks showcasing end-to-end exploratory
  analysis (dynamic slice sliders, segmentation overlays, interactive histograms).
- **Heavier statistics:** Automate per-case histograms, intensity distribution plots, and
  label-percent summaries with batch export.
- **GIF/Video generation:** Batch scripts to produce slice-by-slice animations across
  modalities for review meetings.
- **QC dashboards:** Integrate summary JSON into lightweight dashboards (e.g., streamlit).

These items are tracked separately—coordinate with the team lead before starting a new
thread, and capture decisions in this `docs/` folder.

---

## 10. Contact & support

- **Primary maintainer:** *TBD — assign on project kickoff.*
- **Shared questions:** Use the team’s Slack channel `#brats-post-tx`. Update this guide
  whenever new utilities or configuration steps are introduced.

Let’s keep this document current so onboarding the next teammate is frictionless! 💪
