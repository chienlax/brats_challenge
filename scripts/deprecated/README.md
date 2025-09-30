# Deprecated Scripts

This folder contains scripts that have been superseded by unified tools.

## Visualization Scripts → `visualize_volumes.py`

The following visualization scripts have been consolidated into `visualize_volumes.py`:

### `launch_volume_inspector.py` → `visualize_volumes.py --mode interactive`

**Old command:**
```powershell
python scripts/launch_volume_inspector.py --open-browser
```

**New command:**
```powershell
python scripts/visualize_volumes.py --mode interactive --open-browser
```

---

### `visualize_brats.py` → `visualize_volumes.py --mode static`

**Old command:**
```powershell
python scripts/visualize_brats.py --case-dir training_data_additional/BraTS-GLI-02405-100 \
    --layout modalities --axis axial --fraction 0.55 --output outputs/case.png
```

**New command:**
```powershell
python scripts/visualize_volumes.py --mode static \
    --case-dir training_data_additional/BraTS-GLI-02405-100 \
    --layout modalities --axis axial --fraction 0.55 --output outputs/case.png
```

---

### `generate_case_gifs.py` → `visualize_volumes.py --mode gif`

**Old command:**
```powershell
python scripts/generate_case_gifs.py --root training_data_additional \
    --output-dir outputs/gifs --modality t2f --overlay --max-cases 3
```

**New command:**
```powershell
python scripts/visualize_volumes.py --mode gif \
    --root training_data_additional --output-dir outputs/gifs \
    --gif-modality t2f --overlay --max-cases 3
```

**Note:** In GIF mode, modality and axis flags are prefixed with `--gif-` to avoid conflicts with static mode flags.

---

## Statistics Scripts → `analyze_statistics.py`

The following statistics scripts have been unified into `analyze_statistics.py` with an interactive web interface:

### `generate_case_statistics.py` → `analyze_statistics.py` (Case Statistics Tab)

**Old command:**
```powershell
python scripts/generate_case_statistics.py --root training_data_additional \
    --output-dir outputs/stats --case BraTS-GLI-02405-100 --bins 100
```

**New workflow:**
```powershell
python scripts/analyze_statistics.py --open-browser
# Then select "Case Statistics" tab in browser
# Choose dataset and case from dropdowns
# Click "Compute" to generate histograms and label statistics
```

---

### `summarize_brats_dataset.py` → `analyze_statistics.py` (Dataset Statistics Tab)

**Old command:**
```powershell
python scripts/summarize_brats_dataset.py --root training_data_additional \
    --output outputs/dataset_stats.json --max-cases 50
```

**New workflow:**
```powershell
python scripts/analyze_statistics.py --open-browser
# Then select "Dataset Statistics" tab in browser
# Choose dataset from dropdown
# Click "Load Cached" to load existing stats or "Recompute" to regenerate
# Statistics are automatically cached to outputs/statistics_cache/
```

**Benefits of the new tool:**
- **Interactive browser interface** - no need to view JSON files manually
- **Cached results** - dataset statistics are computed once and reused
- **Visual presentation** - tables and charts for easier interpretation
- **Unified workflow** - both case and dataset statistics in one place

---

## Why Consolidate?

These separate scripts had overlapping functionality and required users to remember multiple command patterns. The unified tools provide:

- **Single interface** for related workflows
- **Consistent flag naming** and behavior
- **Reduced maintenance** burden
- **Easier onboarding** for new team members
- **Better user experience** with interactive interfaces

These deprecated scripts are kept for reference only and will not receive updates or bug fixes.
