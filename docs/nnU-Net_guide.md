# nnU-Net Installation Guide (Windows)

The following checklist walks you through setting up nnU-Net v2 with GPU-enabled PyTorch on a fresh Windows 10/11 workstation. Every command is PowerShell-ready and assumes you want to mirror the layout used in this repository.

## Quick-start reproducibility checklist

1. Verify the workstation meets the GPU, storage, and OS requirements below.
2. Clone `brats_challenge`, copy BraTS cases into `training_data*/`, and confirm each case contains the four modalities plus `-seg`.
3. Create `.venv_nnunet`, activate it, and install the pinned project Python packages with `python -m pip install -r requirements.txt` before adding PyTorch/nnU-Net.
4. Install the CUDA-enabled PyTorch wheel listed in this guide, then `nnunetv2` and `hiddenlayer`.
5. Set the `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` variables for the current shell **and** persist them with `setx`.
6. Regenerate the dataset with `python scripts\prepare_nnunet_dataset.py --clean` so stale cases never leak into a new machine.
7. Run `nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity` once to populate the preprocessed cache.
8. Train the desired folds (0–4 or `all`) with `nnUNetv2_train ... --npz`.
9. Produce predictions via `nnUNetv2_predict` and score them with `nnUNetv2_evaluate` (commands below).
10. Finish by capturing `python -m pip freeze > nnunet-environment.txt` and archiving `outputs\nnunet\` alongside your results for perfect reproducibility.

---

## 1. Prerequisites

### Hardware
- NVIDIA GPU with **≥6 GB VRAM** (10 GB+ recommended for 3D training).
- SSD with at least **300 GB free** (raw data + preprocessed data + results).
- 32 GB RAM recommended for full-resolution 3D training.

### Software
- **Windows 10/11 64-bit**.
- Latest **NVIDIA driver** + **CUDA 12.x runtime** (PyTorch wheels below use CUDA 12.4). Install from [NVIDIA](https://www.nvidia.com/Download/index.aspx) and reboot.
- **Python 3.12** (the repo uses `.venv`, so install from [python.org](https://www.python.org/downloads/) with “Add Python to PATH”).
- **Git** for cloning this repository.

### Optional but useful
- [7-Zip](https://www.7-zip.org/) for archive extraction.
- [VS Code](https://code.visualstudio.com/) with the Python extension.

---

## 2. Clone the repository & prepare data

Open PowerShell in the directory where you want the project to live and run:

```powershell
git clone https://github.com/chienlax/brats_challenge.git
cd brats_challenge
```

Copy or extract your BraTS post-treatment cases into the folder structure used by this repo (`training_data/` or `training_data_additional/`). Each case must contain:

```
BraTS-GLI-xxxxx-yyy/
	├─ BraTS-GLI-xxxxx-yyy-t1c.nii.gz
	├─ BraTS-GLI-xxxxx-yyy-t1n.nii.gz
	├─ BraTS-GLI-xxxxx-yyy-t2f.nii.gz
	├─ BraTS-GLI-xxxxx-yyy-t2w.nii.gz
	└─ BraTS-GLI-xxxxx-yyy-seg.nii.gz
```

---

## 3. Create the nnU-Net virtual environment

Use a dedicated venv so your main environment stays untouched.

```powershell
python -m venv .venv_nnunet
.\.venv_nnunet\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_nnunet.txt `
    --extra-index-url https://download.pytorch.org/whl/cu124   # optional for CUDA wheels
```

*Tip:* When you open a new shell, re-run `.\.venv_nnunet\Scripts\Activate.ps1` before nnU-Net work.

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once (requires admin approval) and retry the activation step.

---

## 4. Install GPU-enabled PyTorch (CUDA 12.4)

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU detection:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

If you have a different CUDA toolkit requirement, swap the `cu124` URL for the version [listed by PyTorch](https://pytorch.org/get-started/locally/).

---

## 5. Install nnU-Net and helpers

```powershell
python -m pip install nnunetv2
# Optional but recommended for architecture visualizations
python -m pip install hiddenlayer
```

Confirm the CLI entry points are installed:

```powershell
nnUNetv2_predict -h
```

Expect warnings about missing `nnUNet_*` environment variables—that’s the next step.

Capture the exact package set so another workstation can mirror it:

```powershell
python -m pip freeze > nnunet-environment.txt
```

Check the file into your lab’s documentation or deployment notes (keep it out of Git if you do not version binary dependencies).

### Exporting architecture diagrams with hiddenlayer

When `hiddenlayer` is available, every `nnUNetv2_train` run automatically writes a PDF snapshot of the compiled model to the fold output directory. After training finishes, look for:

```
outputs\nnunet\nnUNet_results\Dataset501_BraTSPostTx\<trainer>__<plans>__<configuration>\fold_<N>\network_architecture.pdf
```

**PyTorch 2.4+ compatibility note:** hiddenlayer 0.3.x still references the private
`torch.onnx._optimize_trace` helper, which PyTorch removed. This repository ships a
`sitecustomize.py` compatibility shim that silently reintroduces the missing
attribute whenever you launch nnU-Net commands from the project root. If you run
training from a different working directory, make sure the repository is on your
`PYTHONPATH` or copy `sitecustomize.py` alongside your scripts so the diagrams can
be rendered.

Open the PDF to review encoder/decoder stages, feature widths, and skip connections.

Need to regenerate the diagram without retraining? Re-run the trainer in evaluation mode so nnU-Net rebuilds the network and exports the figure:

```powershell
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres 0 --val
```

`--val` performs validation only; it’s fast and recreates `network_architecture.pdf` if it was missing. For deeper customisations you can load the trained checkpoint in your own Python session and call `hiddenlayer.build_graph` on the instantiated nnU-Net network—refer to the hiddenlayer documentation for advanced usage.

---

## 6. Configure nnU-Net storage directories

nnU-Net expects three folders. This repo keeps them under `outputs/nnunet/`.

```powershell
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_results | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\reports\metrics | Out-Null

setx nnUNet_raw "${PWD}\outputs\nnunet\nnUNet_raw"
setx nnUNet_preprocessed "${PWD}\outputs\nnunet\nnUNet_preprocessed"
setx nnUNet_results "${PWD}\outputs\nnunet\nnUNet_results"
```

Close and reopen PowerShell (or run the lines below) to use them immediately:

```powershell
$env:nnUNet_raw="${PWD}\outputs\nnunet\nnUNet_raw"
$env:nnUNet_preprocessed="${PWD}\outputs\nnunet\nnUNet_preprocessed"
$env:nnUNet_results="${PWD}\outputs\nnunet\nnUNet_results"
```

Verify:

```powershell
[Environment]::GetEnvironmentVariable("nnUNet_raw", "User")
```

On a fresh shell you can double-check all three variables at once:

```powershell
Get-ChildItem Env:nnUNet_*
```

---

## 7. Convert BraTS data into nnU-Net format

The repo supplies `scripts/prepare_nnunet_dataset.py` to reorganize cases and generate `dataset.json` for dataset ID **501**.

```powershell
python scripts\prepare_nnunet_dataset.py --clean --require-test-labels
```

Flags you can override:
- `--train-sources`: labeled cases for `imagesTr/labelsTr` (defaults to `training_data` and `training_data_additional`).
- `--test-sources`: unlabeled cases that should populate `imagesTs` (useful when you keep a held-out evaluation set, e.g. `--test-sources training_data_additional`).
- `--dataset-id`: name of the dataset folder inside `nnUNet_raw`.
- `--nnunet-raw`: alternate raw-data root.
- `--clean`: wipe the existing dataset folder before repopulating (prevents stale cases from previous runs).
- `--require-test-labels`: ensure every test case includes a segmentation; when satisfied the script mirrors them into `labelsTs` for evaluation.

After the script finishes you should have:

```
outputs\nnunet\nnUNet_raw\Dataset501_BraTSPostTx\
	├─ dataset.json
	├─ imagesTr\BraTS-GLI-02405-100_0000.nii.gz
	├─ ...
	├─ labelsTr\BraTS-GLI-02405-100.nii.gz
	├─ imagesTs\BraTS-GLI-02417-101_0000.nii.gz  (if --test-sources was provided)
	└─ labelsTs\BraTS-GLI-02417-101.nii.gz       (if test segmentations were available)
```

When segmentations are present for the test set (for example, a labeled validation cohort), they are hard-linked or copied into `labelsTs` automatically. If you pass `--require-test-labels`, the script fails fast when any case is missing a `-seg` file so evaluation never runs against incomplete data.

### Resetting or migrating the dataset

When you refresh the dataset (for example, migrating to a new PC or pruning cases), clear all derived artifacts first so no stale volumes sneak back in:

```powershell
Remove-Item outputs\nnunet\nnUNet_raw\Dataset501_BraTSPostTx -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item outputs\nnunet\nnUNet_preprocessed\Dataset501_BraTSPostTx -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item outputs\nnunet\nnUNet_results\Dataset501_BraTSPostTx -Recurse -Force -ErrorAction SilentlyContinue
python scripts\prepare_nnunet_dataset.py --clean --require-test-labels
```

The first two `Remove-Item` calls are idempotent and safe to run even when the folders do not exist. The final `--clean` run rebuilds `dataset.json` from scratch.

### Quick integrity checks

Open `outputs\nnunet\nnUNet_raw\Dataset501_BraTSPostTx\dataset.json` and confirm the number of `training` entries matches the count reported by the script.

```powershell
python -c "import json, pathlib; path = pathlib.Path('outputs') / 'nnunet' / 'nnUNet_raw' / 'Dataset501_BraTSPostTx' / 'dataset.json'; data = json.loads(path.read_text()); print('Training cases:', len(data['training'])); print('Test cases:', len(data.get('test', [])))"
```

If `labelsTs` exists, confirm it contains the same number of files as `imagesTs` to ensure evaluation-ready segmentations were copied.

---

## 8. Plan & preprocess with nnU-Net

Run the pipeline planner once to generate configuration files and cached tensors. This step can take 30–40 minutes depending on disk speed.

```powershell
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
```

Outputs land in `outputs\nnunet\nnUNet_preprocessed\Dataset501_BraTSPostTx`. You’ll see notices about which configurations (2d, 3d_fullres, etc.) were created.

---

## 9. Train models

Train each fold (0–4) for the configurations you care about. Example for 3D full resolution:

```powershell
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres 0 --npz
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres 1 --npz
...
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres 4 --npz
nnUNetv2_train Dataset501_BraTSPostTx 3d_fullres all --npz
nnUNetv2_train Dataset501_BraTSPostTx 2d all --npz
nnUNetv2_train Dataset501_BraTSPostTx 2d all --npz

Repeat for `2d` (and `3d_lowres` if the planner generated it). Checkpoints and logs appear under `outputs\nnunet\nnUNet_results`.

Optional: adopt the new residual encoder presets (see [resenc_presets.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md)) by adding `--trainer nnUNetTrainer_ResEncUNet`. Adjust GPU memory expectations accordingly.

---

## 10. Export architecture diagrams

(use .venv)

Capture a static overview of the trained configuration so future runs can review the layer layout without rerunning training. The `visualize_architecture.py` helper consumes the generated `plans.json` and saves both overview and detailed diagrams:

```powershell
$plans = Join-Path $env:nnUNet_preprocessed 'Dataset501_BraTSPostTx\plans.json'
python scripts\visualize_architecture.py $plans --config 3d_fullres --output outputs\nnunet\reports\architecture\3d_fullres_overview.png --detailed-output outputs\nnunet\reports\architecture\3d_fullres_detailed.png
```

```powershell
$plans = Join-Path $env:nnUNet_preprocessed 'Dataset501_BraTSPostTx\plans.json'
python scripts\visualize_architecture.py $plans --config 2d --output outputs\nnunet\reports\architecture\2d_overview.png --detailed-output outputs\nnunet\reports\architecture\2d_detailed.png
```

- The helper now auto-resolves `plans.json` straight from `nnUNet_results` if the
    dataset folder in `nnUNet_preprocessed` does not contain a copy. Passing the
    historical path (as above) still works; you will see a console message pointing
    to the resolved file under the trainer directory.


- Swap `3d_fullres` for any other configuration you trained (for example `2d`).
- Omit `--output` to open the plot interactively, or add `--vector-output`/`--detailed-vector-output` to emit `.svg` assets alongside the PNGs.

Keep the exported figures under `outputs\nnunet\reports\architecture\` (git-ignored) so each experiment bundle includes its architecture snapshot. Hiddenlayer already saves `network_architecture.pdf` inside each fold folder; these PNG/SVG files act as a quick-look summary for reports and slide decks.

---

## 11. Inference

Once training completes, run inference on held-out volumes:

```powershell
nnUNetv2_predict -i <input_folder> -o <output_folder> -d 501 -c 3d_fullres -f all --device cuda
```

Eg:
```powershell
nnUNetv2_predict `
    -i outputs\nnunet\nnUNet_raw\Dataset501_BraTSPostTx\imagesTs `
    -o outputs\nnunet\predictions\Dataset501_BraTSPostTx\2d `
    -d 501 `
    -c 2d `
    -f all `
    -device cuda
```

```
nnUNetv2_predict `
    -i outputs\nnunet\nnUNet_raw\Dataset501_BraTSPostTx\imagesTs `
    -o outputs\nnunet\predictions\Dataset501_BraTSPostTx\3d_fullres `
    -d 501 `
    -c 3d_fullres `
    -f all `
    -device cuda
```

Make sure input volumes follow the same naming scheme as training cases (`CASE_0000.nii.gz`, etc.).

---

## 12. Evaluate predictions

Score predictions directly from the exported masks to validate your pipeline end to end:

```powershell
nnUNetv2_evaluate -i <prediction_folder> -l <ground_truth_folder>
```

Eg:

```powershell
nnUNetv2_evaluate_simple `
    outputs\nnunet\nnUNet_raw\Dataset501_BraTSPostTx\labelsTs `
    outputs\nnunet\predictions\Dataset501_BraTSPostTx\2d `
    -o outputs\nnunet\reports\metrics\2d_metrics.json `
    -l 1 2 3 4
```

- `<prediction_folder>` is the directory produced by `nnUNetv2_predict` (filenames must match the original case IDs).
- `<ground_truth_folder>` should contain reference segmentations with matching filenames (for example `outputs\nnunet\nnUNet_raw\Dataset501_BraTSPostTx\labelsTs` or a clinical hold-out set).

Add dataset- or configuration-specific flags such as `-d 501 -c 3d_fullres -f all --save_json outputs\nnunet\reports\metrics\3d_fullres_metrics.json` if you want the CLI to validate consistency against known metadata or export a metrics file. Storing metrics under `outputs\nnunet\reports\metrics\` keeps evaluation artifacts alongside the architecture figures.

Need to score the built-in nnU-Net cross-validation without rerunning inference? Point the evaluator at the results folder:

```powershell
nnUNetv2_evaluate -d Dataset501_BraTSPostTx -c 3d_fullres -f all --results outputs\nnunet\reports\metrics\3d_fullres_cv
```

Run `nnUNetv2_evaluate -h` on the target machine to review the full flag list—helpful when you need to disable post-processing, restrict labels, or mirror the official nnU-Net cross-validation reports.

### 12.1 One-step evaluation bundle

When you want the standard nnU-Net metrics and the BraTS lesion-wise report in a
single pass, use the orchestration helper shipped with this repository:

```powershell
python scripts/run_full_evaluation.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres `
    --output-dir outputs/nnunet/reports/metrics/3d_fullres
```

```powershell
python scripts/run_full_evaluation.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/nnunet/predictions/Dataset501_BraTSPostTx/2d `
    --output-dir outputs/nnunet/reports/metrics/2d
```

`run_full_evaluation.py` will (1) call `nnUNetv2_evaluate_simple` (unless you pass
`--skip-nnunet`) and (2) run the BraTS lesion-wise scorer. The command writes three
JSON files into the requested `--output-dir`:

- `nnunet_metrics.json` – the raw nnU-Net evaluation output
- `lesion_metrics.json` – lesion-wise Dice / HD95 tables and aggregates
- `full_metrics.json` – combined manifest referencing both reports and the source folders

Use `--omit-aggregates` to hide Tumor Core and Whole Tumor rows or point
`--labels` to a different foreground set when benchmarking alternative label maps.

---

## 13. Troubleshooting tips

| Issue | Likely cause | Fix |
| --- | --- | --- |
| `torch.cuda.is_available() -> False` | Driver/CUDA mismatch or using the wrong environment | Reinstall NVIDIA driver + reboot; ensure PowerShell shows `(\venv_nnunet)` in the prompt.
| `Could not find dataset ...` | Environment variables not picked up | Reopen PowerShell or set `$env:nnUNet_raw=...` in the current shell.
| `Missing modality` errors during dataset prep | Incomplete BraTS case | Verify every folder has four modalities + segmentation.
| Planner warns about new presets | Informational | Consider reading the residual encoder preset doc for potentially better results.

---

## 14. Maintenance & upgrades

- Keep `.venv_nnunet` isolated from other projects; update packages with caution.
- When PyTorch releases new CUDA wheels, upgrade with `python -m pip install --upgrade torch torchvision torchaudio --index-url ...`.
- Track nnU-Net releases on [GitHub](https://github.com/MIC-DKFZ/nnUNet/releases) for bug fixes and new features.
- Back up `outputs\nnunet\` regularly—preprocessing is costly.

With these steps you can replicate the full nnU-Net setup on any Windows workstation with minimal surprises.
