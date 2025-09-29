# BraTS Post-treatment Dataset Overview

This document consolidates the essential facts about the BraTS 2025 post-treatment glioma dataset hosted in this repository, using the analytics scripts and notebook delivered alongside the data. Share it with anyone who needs a one-stop reference on data geometry, modality coverage, label semantics, and available tooling.

---

## 1. Dataset snapshot

- **Cases:** 271 folders under `training_data_additional/` (`BraTS-GLI-XXXXX-YYY`).
- **Modalities per case:** T1 post-contrast (`t1c`), T1 native (`t1n`), T2 FLAIR (`t2f`), T2-weighted (`t2w`), plus a segmentation mask (`seg`).
- **File format:** All volumes and masks are 3D NIfTI (`.nii.gz`).
- **Storage policy:** Imaging directories are excluded from git via `.gitignore`. Acquire them from secure storage before running scripts.

### Recommended validation workflow
1. Install the Python environment (`pip install -r requirements.txt`).
2. Run the dataset summary script to verify geometry/labels.
3. Generate optional per-case histograms and GIFs for deeper QA.
4. Explore interactively via the notebook (slice viewer, histograms, GIF helpers).

---

## 2. Geometry and orientation consistency

Results generated via `scripts/summarize_brats_dataset.py --root training_data_additional --output outputs/dataset_stats.json`:

| Metric | Observation | Source |
|--------|-------------|--------|
| Dimensions | Every modality and mask is `182 × 218 × 182` voxels. | `aggregate.shape_counts` |
| Voxel spacing | All subjects have isotropic `1.0 mm` spacing in x/y/z. | `aggregate.voxel_spacing_counts` |
| Orientation | All affines resolve to `(L, A, S)` (Left → Right, Anterior → Posterior, Superior → Inferior). No flipped volumes detected. | `aggregate.orientation_counts` |
| Modalities present | `t1c`, `t1n`, `t2f`, `t2w` available for 271/271 cases. | `aggregate.modality_presence` |
| Data types | 231 cases per modality in `float32`, 40 in `float64`; plan to cast to `float32` during preprocessing for consistency. | `aggregate.dtype_counts` |

Use the summary JSON as a regression baseline after any new data drop.

---

## 3. Intensity landscape (per modality)

Global percentiles from the summary script (averaged across patients):

| Modality | Mean of 99th percentile | Range of 99th percentile | Mean intensity (whole volume) |
|----------|-------------------------|---------------------------|-------------------------------|
| `t1c` | ~545.6 | 123.1 → 2420.9 | 69.1 |
| `t1n` | ~442.4 | 114.6 → 1915.4 | 57.2 |
| `t2f` | ~434.7 | 48.1 → 1447.1 | 57.6 |
| `t2w` | ~1034.4 | 244.5 → 3150.4 | 108.8 |

Intensity spread motivates the 1st–99th percentile normalization adopted in both `visualize_brats.py` and the histogram/GIF utilities.

---

## 4. Segmentation label distribution

Labels are encoded as integers within the `*-seg.nii.gz` masks.

| Label | Region | Dataset voxel count | Share of labeled voxels |
|-------|--------|---------------------|-------------------------|
| 0 | Background | 1,937,164,818 | — (excluded from percentage) |
| 1 | Enhancing tumor (ET) | 541,043 | **2.74 %** |
| 2 | Non-enhancing tumor (NET/necrosis) | 14,107,863 | **71.49 %** |
| 3 | Peritumoral edema (ED) | 2,990,458 | **15.15 %** |
| 4 | Resection cavity (RC) | 2,095,490 | **10.62 %** |

Percentages are computed relative to all tumor-class voxels (labels 1–4). These numbers are reproduced in `outputs/dataset_stats.json` and in the notebook’s label analytics section.

Color legend (aligned with the official BraTS guidance and used across visualization, notebook, and GIF scripts):

- `1` Enhancing tumor → `#1f77b4` (blue)
- `2` Non-enhancing tumor → `#d62728` (red)
- `3` Peritumoral edema / SNFH → `#2ca02c` (green)
- `4` Resection cavity → `#f1c40f` (yellow)

---

## 5. Tooling index

### `scripts/summarize_brats_dataset.py`
- **Purpose:** Aggregate geometry, intensity percentiles, label counts, orientation, and dtype statistics across the cohort.
- **Key outputs:** `outputs/dataset_stats.json` (global metrics) and optional `sample_cases` block for manual inspection.
- **Tips:** Use `--sample-cases N` to embed per-case snippets in the JSON for code reviews or anomaly triage.

### `scripts/visualize_brats.py`
- **Purpose:** Produce modality comparison panels and orthogonal views with mask overlays.
- **Usage notes:** `--layout modalities` for side-by-side sequence comparison; `--layout orthogonal` for sagittal/coronal/axial exploration; supports `--no-overlay`, slice `--fraction`/`--index`, and per-case legends.

### `scripts/generate_case_statistics.py`
- **Purpose:** Create per-case histogram PNGs plus a JSON table of voxel/percentage statistics by label.
- **Workflow:**
  1. `python scripts/generate_case_statistics.py --root training_data_additional --output-dir outputs/stats`
  2. Review `case_statistics.json` for label prevalence outliers.
  3. Inspect histogram images in `outputs/stats/<CASE>/` for modality intensity anomalies.
- **Config knobs:** `--bins`, `--lower-percentile`, `--upper-percentile` help tune normalization for atypical scans.

### `scripts/generate_case_gifs.py`
- **Purpose:** Batch-create GIFs sweeping through slices along a chosen axis with optional overlays.
- **Workflow:**
  1. `python scripts/generate_case_gifs.py --root training_data_additional --output-dir outputs/gifs --modality t2f --axis axial --step 3 --overlay`
  2. Share resulting files (`CASE_modality_axis.gif`) for radiology review or QC.
- **Options:** `--case` to isolate a subject; `--step` to adjust sampling stride; `--fps` for playback speed; `--overlay` toggle.

### `notebooks/brats_exploration.ipynb`
- **Modules covered:**
  - Metadata table (shapes, spacing, orientation).
  - Orientation & dtype frequency charts.
  - Label analytics table + per-case statistics.
  - Interactive slice viewer with modality/axis widgets.
  - Histogram plots and label bar charts per case.
  - GIF helper routines mirroring the batch script for ad-hoc experiments.
- **Launch:** `jupyter notebook notebooks/brats_exploration.ipynb` (ensure `ipywidgets` is installed).

---

## 6. Suggested QA playbooks

1. **New data drop validation**
   - Run `summarize_brats_dataset.py`; confirm 271 cases, matching shapes/spacing/orientation.
   - Inspect `dtype_counts`; ensure modalities remain predominantly `float32` post-conversion.
   - Review label distribution table for sudden shifts (e.g., missing class 4 voxels).

2. **Case-level triage**
   - Use `generate_case_statistics.py --case CASE_ID` to visualize histograms and label percentages.
   - Open the notebook’s interactive viewer for manual slice checks around suspicious volumes.
   - Produce GIFs focusing on `t2f` or `t1c` with overlays for quick stakeholder review.

3. **Presentation prep**
   - Export high-resolution figures via `visualize_brats.py`.
   - Combine GIF outputs or notebook snapshots into slide decks.

---

## 7. Next steps & extensions

- Automate dashboard ingestion of `dataset_stats.json` and `case_statistics.json` for continuous monitoring.
- Integrate class balancing insights (e.g., under-representation of enhancing tumor) into model training strategies.
- Extend GIF tooling to MP4 exports and add side-by-side modality playback.
- Enrich the notebook with interactive histogram brushing and tumor volume trend charts.

---

## 8. References

- **BraTS challenge page:** https://www.synapse.org/#!Synapse:syn51156910/wiki/ (official guidelines and label definitions).
- **Nibabel orientation primer:** https://nipy.org/nibabel/orientation.html
- **Matplotlib medical imaging tips:** https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html

Keep this document in sync with the outputs of the scripts above. If you update any analytics code, regenerate `outputs/dataset_stats.json` and propagate key numbers here to maintain a trustworthy source of truth.
