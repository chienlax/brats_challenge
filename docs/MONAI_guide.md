# MONAI Installation Guide (Windows)

I mess up this shit so much, ngl

## 0.1 Create the MONAI virtual environment and install needed library

```powershell
python -m venv .venv_monai
.\.venv_monai\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

pip install "monai[all]" nibabel pydicom ipywidgets==8.1.2

pip uninstall -y pytensor pip install -U filelock
```

Verify GPU detection:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

*Tip:* When you open a new shell, re-run `.\.venv_nnunet\Scripts\Activate.ps1` before nnU-Net work.

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once (requires admin approval) and retry the activation step.

---

## Running the BraTS MONAI fine-tuning workflow

### Prerequisites

- Activate the MONAI environment (`.\.venv_monai\Scripts\Activate.ps1`) every time you start a new PowerShell session.
- Ensure the BraTS post-treatment datasets are unpacked under one or more folders such as `training_data` and `training_data_additional` (case layout `BraTS-GLI-XXXX-XXX`).
- Obtain the nnU-Net `splits_final.json` (for example from `nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json`). The fine-tuning scripts reuse folds defined there.
- Have at least one CUDA-capable GPU available. If no GPU is detected, the scripts automatically fall back to CPU with a significant slowdown.

### Step-by-step: Training (`scripts/train_monai_finetune.py`)

1. **Activate the environment**
   ```powershell
   .\.venv_monai\Scripts\Activate.ps1
   ```
2. **(Optional) Validate GPU availability**
   ```powershell
   python -c "import torch; print('CUDA visible:', torch.cuda.is_available())"
   ```
3. **Launch training with the required paths and preferred options**

   ```powershell
   python scripts/train_monai_finetune.py `
       --data-root training_data training_data_additional `
       --split-json outputs/nnunet/nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json `
       --fold 0 `
       --output-root outputs/monai_ft `
       --bundle-name brats_mri_segmentation `
       --amp
   ```

4. **Monitor progress**: the script prints per-epoch loss and validation Dice scores (when `--val-interval` divides the epoch count). Checkpoints are written to `<output-root>/checkpoints/`.
5. **Inspect logs**: a JSON history is written to `<output-root>/metrics/training_log.json` after Stage 2 completes.

#### CLI reference: `train_monai_finetune.py`

**Core dataset and bundle settings**

| Flag | Type / Default | Description | Example use case |
| --- | --- | --- | --- |
| `--data-root` | `Path` (repeatable) / *required* | One or more directories containing BraTS case folders. | Combine institutional data and BraTS training set: `--data-root training_data training_data_additional`. |
| `--split-json` | `Path` / *required* | Path to nnU-Net fold definition (`splits_final.json`). | Align folds with nnU-Net benchmark results. |
| `--fold` | `int` / `0` | Fold index used for validation. | Train on fold 1: `--fold 1`. |
| `--bundle-name` | `str` / `brats_mri_segmentation` | MONAI bundle identifier to download/load. | Swap to a custom bundle: `--bundle-name my_org/brats_variant`. |
| `--bundle-dir` | `Path` / `None` | Directory containing pre-downloaded bundle assets. | Point to offline cache: `--bundle-dir pretrained/monai_bundle`. |
| `--output-root` | `Path` / `outputs/monai_ft` | Root folder for checkpoints, logs, and metrics. | Store runs per experiment: `--output-root outputs/monai_ft_augmented`. |
| `--no-download` | flag / `False` | Skip network download attempts (useful in offline environments). | Combine with `--bundle-dir` when the bundle is already unpacked. |
| `--seed` | `int` / `2024` | Sets all RNG seeds for reproducibility. | Use a different seed when ensembling: `--seed 1337`. |

**Two-stage training schedule**

| Flag | Type / Default | Description | Example use case |
| --- | --- | --- | --- |
| `--stage1-epochs` | `int` / `80` | Decoder/head adaptation epoch count. | Reduce for a quick pilot: `--stage1-epochs 20`. |
| `--stage1-lr` | `float` / `1e-3` | Learning rate during Stage 1. | Lower if gradients diverge: `--stage1-lr 5e-4`. |
| `--stage1-accum` | `int` / `3` | Gradient accumulation steps in Stage 1. | Increase when using smaller GPUs: `--stage1-accum 6`. |
| `--stage2-epochs` | `int` / `140` | End-to-end fine-tuning epoch count. | Shorten for limited time: `--stage2-epochs 60`. |
| `--stage2-lr` | `float` / `1e-5` | Learning rate during Stage 2. | Raise slightly when encoder adapts slowly: `--stage2-lr 3e-5`. |
| `--stage2-accum` | `int` / `2` | Gradient accumulation in Stage 2. | Match Stage 1 behaviour: `--stage2-accum 3`. |
| `--weight-decay` | `float` / `1e-5` | L2 regularisation for AdamW in both stages. | Increase to curb overfitting: `--weight-decay 3e-5`. |
| `--grad-clip` | `float` / `12.0` | Global gradient norm clipping threshold. | Tighten to avoid spikes: `--grad-clip 5.0`. |
| `--resume` | `Path` / `None` | Resume Stage 1 or Stage 2 from a checkpoint created by this script. | Restart Stage 2 from the best checkpoint: `--resume outputs/monai_ft/checkpoints/stage2_best.pt`. |
| `--save-checkpoint-frequency` | `int` / `10` | Frequency (epochs) for periodic checkpoint snapshots. | Save every 5 epochs: `--save-checkpoint-frequency 5`. |

**Data loading, preprocessing, and patches**

| Flag | Type / Default | Description | Example use case |
| --- | --- | --- | --- |
| `--batch-size` | `int` / `1` | Per-iteration batch size. | Increase to 2 on >24 GB GPUs: `--batch-size 2`. |
| `--num-workers` | `int` / `2` | DataLoader worker processes. | Raise to fully utilise CPU: `--num-workers 6`. |
| `--cache-rate` | `float` / `0.1` | Fraction of dataset cached in RAM by `CacheDataset`. | Cache all samples on fast workstations: `--cache-rate 1.0`. |
| `--patch-size` | `int int int` / `224 224 144` | Training crop size applied after padding. | Match smaller GPUs: `--patch-size 192 192 128`. |
| `--spacing` | `float float float` / `1.0 1.0 1.0` | Target voxel spacing during resampling. | Keep anisotropic resolution for faster runs: `--spacing 1.0 1.0 2.0`. |
| `--class-weights` | `float float float float` / `1.0 1.0 1.0 1.5` | DiceCE loss weights for ET, NETC, SNFH, RC (background excluded). | Emphasise RC even further: `--class-weights 1.0 1.0 1.0 2.0`. |

**Validation and model wiring**

| Flag | Type / Default | Description | Example use case |
| --- | --- | --- | --- |
| `--val-interval` | `int` / `2` | Run validation every *N* epochs. | Validate more frequently near convergence: `--val-interval 1`. |
| `--eval-roi` | `int int int` / `224 224 144` | Sliding-window ROI size during validation inference. | Expand for larger contexts: `--eval-roi 256 256 160`. |
| `--sw-batch-size` | `int` / `1` | Sliding-window batch size for validation inference. | Increase when VRAM allows: `--sw-batch-size 2`. |
| `--sw-overlap` | `float` / `0.5` | Overlap ratio between validation windows. | Reduce to speed up validation: `--sw-overlap 0.25`. |
| `--amp` | flag / `False` | Enable PyTorch automatic mixed precision (saves memory, speeds up). | Recommended for modern GPUs: add `--amp`. |
| `--encoder-prefix` | `str ...` / `encoder backbone swinViT downsample` | Parameter name prefixes regarded as encoder modules (frozen in Stage 1). | Customise when adapting another architecture: `--encoder-prefix encoder stage1`. |
| `--head-key` | `str` / `output_layer` | Attribute name of the bundle’s final convolution; used when regenerating the output head. | Required if your bundle exposes the head under a different name. |

#### Common training workflows
```powershell
python scripts/train_monai_finetune.py `
    --data-root training_data `
    --split-json outputs/nnunet/nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json `
    --batch-size 1 --num-workers 2 `
    --stage1-epochs 2 --stage2-epochs 2 `
    --patch-size 64 64 64 `
    --fold 0 `
    --output-root outputs/monai_ft_nnunet_aligned `
    --amp
```


- **Resume Stage 1 after interruption**:
    ```powershell
    python scripts/train_monai_finetune.py `
            --data-root training_data training_data_additional `
            --split-json outputs/nnunet/nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json `
            --fold 0 `
            --resume outputs/monai_ft_smoke/checkpoints/stage1_epoch40.pt `
            --amp
    ```
- **Resume Stage 2 from its latest checkpoint**:
    ```powershell
    python scripts/train_monai_finetune.py `
            --data-root training_data training_data_additional `
            --split-json outputs/nnunet/nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json `
            --fold 0 `
            --resume outputs/monai_ft_smoke/checkpoints/stage2_epoch120.pt `
            --amp
    ```

### Step-by-step: Inference (`scripts/infer_monai_finetune.py`)

1. **Activate the environment**
   ```powershell
   .\.venv_monai\Scripts\Activate.ps1
   ```
2. **Select the checkpoint produced by training** (for example `outputs/monai_ft/checkpoints/stage2_best.pt`).
3. **Create or choose an output directory** where predicted segmentations will be written.
4. **Run inference**

   ```powershell
   python scripts/infer_monai_finetune.py `
       --data-root training_data_additional `
       --split-json outputs/nnunet/nnUNet_preprocessed/Dataset501_BraTSPostTx/splits_final.json `
       --fold 0 `
       --checkpoint outputs/monai_ft_smoke/fold0/checkpoints/stage2_best.pt `
       --output-dir outputs/monai_ft_smoke/predictions/fold0 `
       --bundle-name brats_mri_segmentation `
       --amp
   ```

5. **Verify outputs**: the script prints one line per case and writes `*.nii.gz` predictions following nnU-Net naming conventions.
6. **(Optional) Trigger evaluation** by passing `--run-evaluation --ground-truth training_data` to automatically call `scripts/run_full_evaluation.py`.

#### CLI reference: `infer_monai_finetune.py`

| Flag | Type / Default | Description | Example use case |
| --- | --- | --- | --- |
| `--data-root` | `Path` (repeatable) / *required* | Directories with BraTS case folders to process. | Combine hospital data with BraTS: `--data-root validation hospital_cases`. |
| `--split-json` | `Path` / *required* | Fold definition reused from nnU-Net. | Keep evaluation consistent across toolchains. |
| `--fold` | `int` / `0` | Choose which validation fold to run. | Generate predictions for fold 2: `--fold 2`. |
| `--checkpoint` | `Path` / *required* | Trained weights (Stage 1 or Stage 2). | Use the best Stage 2 model: `--checkpoint outputs/monai_ft/checkpoints/stage2_best.pt`. |
| `--output-dir` | `Path` / *required* | Destination for NIfTI predictions. | Separate experiments: `--output-dir outputs/monai_ft/predictions/fold0_aug`. |
| `--bundle-name` | `str` / `brats_mri_segmentation` | MONAI bundle identifier. | Match the bundle used during training. |
| `--bundle-dir` | `Path` / `None` | Directory with cached bundle assets. | Avoid redownloading on offline nodes. |
| `--no-download` | flag / `False` | Prevent bundle downloads. | Pair with `--bundle-dir` in air-gapped environments. |
| `--spacing` | `float float float` / `1.0 1.0 1.0` | Resampling spacing before inference. | Match training spacing overrides. |
| `--patch-size` | `int int int` / `224 224 144` | Padding size to guarantee minimum extent. | Reduce for smaller GPUs: `--patch-size 192 192 128`. |
| `--roi-size` | `int int int` / `224 224 144` | Sliding-window ROI size for inference. | Increase when VRAM allows: `--roi-size 256 256 160`. |
| `--sw-batch-size` | `int` / `1` | Sliding-window batch size. | Set to 2 on 48 GB GPUs. |
| `--sw-overlap` | `float` / `0.5` | Overlap between inference windows. | Reduce to 0.25 to speed up inference at slight risk of seams. |
| `--amp` | flag / `False` | Enable automatic mixed precision. | Recommended for faster inference on CUDA. |
| `--ground-truth` | `Path` / `None` | Directory with reference segmentations for evaluation. | Typically `training_data` or held-out labels. |
| `--evaluation-output` | `Path` / `None` | Destination for evaluation reports (requires `--ground-truth`). | Save metrics in a custom folder: `--evaluation-output outputs/monai_ft/reports/fold0`. |
| `--run-evaluation` | flag / `False` | Run `scripts/run_full_evaluation.py` after inference. | Evaluate predictions in one step. |
| `--encoder-prefix` | `str ...` / `encoder backbone swinViT downsample` | Parameter prefixes used when aligning checkpoint parameters. | Override if your custom bundle names encoder modules differently. |
| `--seed` | `int` / `2024` | Seed for deterministic preprocessing and inference. | Produce reproducible predictions across runs. |

#### Evaluation workflow example

```powershell
python scripts/infer_monai_finetune.py `
    --data-root training_data_additional `
    --dataset-json outputs/nnunet/nnUNet_preprocessed/Dataset501_BraTSPostTx/dataset.json `
    --fold 0 `
    --checkpoint outputs/monai_ft_nnunet_aligned/fold0/checkpoints/stage2_best.pt `
    --output-dir outputs/monai_ft_nnunet_aligned/predictions/fold0 `
    --amp `
    --run-evaluation `
    --ground-truth outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs
```

```powershell
python scripts/run_full_evaluation_monai.py `
    outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx/labelsTs `
    outputs/monai_ft_nnunet_aligned/predictions/fold0 `
    --output-dir outputs/reports/monai_fold0 `
    --pretty
```




This command generates NIfTI predictions and immediately writes Dice/Tversky summaries under `outputs/monai_ft/eval_reports`.

---

