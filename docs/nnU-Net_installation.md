# nnU-Net Installation Guide (Windows)

The following checklist walks you through setting up nnU-Net v2 with GPU-enabled PyTorch on a fresh Windows 10/11 workstation. Every command is PowerShell-ready and assumes you want to mirror the layout used in this repository.

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
```

*Tip:* When you open a new shell, re-run `.\.venv_nnunet\Scripts\Activate.ps1` before nnU-Net work.

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

---

## 6. Configure nnU-Net storage directories

nnU-Net expects three folders. This repo keeps them under `outputs/nnunet/`.

```powershell
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force outputs\nnunet\nnUNet_results | Out-Null

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

---

## 7. Convert BraTS data into nnU-Net format

The repo supplies `scripts/prepare_nnunet_dataset.py` to reorganize cases and generate `dataset.json` for dataset ID **501**.

```powershell
python scripts\prepare_nnunet_dataset.py
```

Flags you can override:
- `--sources`: additional folders to scan for `BraTS-GLI-*` cases.
- `--dataset-id`: name of the dataset folder inside `nnUNet_raw`.
- `--nnunet-raw`: alternate raw-data root.

After the script finishes you should have:

```
outputs\nnunet\nnUNet_raw\Dataset501_BraTSPostTx\
	├─ dataset.json
	├─ imagesTr\BraTS-GLI-02405-100_0000.nii.gz
	├─ ...
	└─ labelsTr\BraTS-GLI-02405-100.nii.gz
```

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
```

Repeat for `2d` (and `3d_lowres` if the planner generated it). Checkpoints and logs appear under `outputs\nnunet\nnUNet_results`.

Optional: adopt the new residual encoder presets (see [resenc_presets.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md)) by adding `--trainer nnUNetTrainer_ResEncUNet`. Adjust GPU memory expectations accordingly.

---

## 10. Inference

Once training completes, run inference on held-out volumes:

```powershell
nnUNetv2_predict -i <input_folder> -o <output_folder> -d 501 -c 3d_fullres -f all --device cuda
```

Make sure input volumes follow the same naming scheme as training cases (`CASE_0000.nii.gz`, etc.).

---

## 11. Troubleshooting tips

| Issue | Likely cause | Fix |
| --- | --- | --- |
| `torch.cuda.is_available() -> False` | Driver/CUDA mismatch or using the wrong environment | Reinstall NVIDIA driver + reboot; ensure PowerShell shows `(\venv_nnunet)` in the prompt.
| `Could not find dataset ...` | Environment variables not picked up | Reopen PowerShell or set `$env:nnUNet_raw=...` in the current shell.
| `Missing modality` errors during dataset prep | Incomplete BraTS case | Verify every folder has four modalities + segmentation.
| Planner warns about new presets | Informational | Consider reading the residual encoder preset doc for potentially better results.

---

## 12. Maintenance & upgrades

- Keep `.venv_nnunet` isolated from other projects; update packages with caution.
- When PyTorch releases new CUDA wheels, upgrade with `python -m pip install --upgrade torch torchvision torchaudio --index-url ...`.
- Track nnU-Net releases on [GitHub](https://github.com/MIC-DKFZ/nnUNet/releases) for bug fixes and new features.
- Back up `outputs\nnunet\` regularly—preprocessing is costly.

With these steps you can replicate the full nnU-Net setup on any Windows workstation with minimal surprises.
