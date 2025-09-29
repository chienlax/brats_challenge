# BraTS Post-treatment Visualization Toolkit

Utilities for exploring the BraTS 2025 post-treatment glioma dataset located in
`training_data_additional/`.

## Setup

Create and activate the Python environment (already configured in `.venv/`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset summary

Generate global statistics and (optionally) save them to JSON:

```powershell
python scripts/summarize_brats_dataset.py --root training_data_additional --output outputs/dataset_stats.json
```

Key metrics include:

- Number of cases processed.
- Shape / voxel-spacing consistency.
- Intensity percentiles per modality.
- Label frequencies across segmentation masks.
- Modalities present / missing per case.

## Visualization script

Render multi-modal panels or orthogonal planes with segmentation overlays:

```powershell
python scripts/visualize_brats.py --case-dir training_data_additional/BraTS-GLI-02405-100 --layout modalities --axis axial --fraction 0.55 --output outputs/BraTS-GLI-02405-100_axial.png
```

Alternative example showing orthogonal views for a single modality:

```powershell
python scripts/visualize_brats.py --case-dir training_data_additional/BraTS-GLI-02405-100 --layout orthogonal --modality t1c --indices 90 110 70 --output outputs/BraTS-GLI-02405-100_orthogonal.png
```

### Options

- `--layout`: `modalities` (default) or `orthogonal`.
- `--axis`: axis for `modalities` layout (`axial`, `coronal`, `sagittal`).
- `--fraction` / `--index`: control slice location.
- `--indices`: specify slice indices for orthogonal layout.
- `--no-overlay`: disable segmentation overlay.
- `--show`: open an interactive matplotlib window in addition to saving the figure.

Outputs are saved under the provided `--output` path; if omitted, the figure is
shown interactively.

## Outputs directory

Figures and JSON reports are written to the `outputs/` folder (created on demand).
