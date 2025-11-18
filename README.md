# BraTS Post-treatment

**For detailed setup instruction, go to: [Usage Guide Document](docs/comprehensive_usage.md)**

## Setup

Create and activate the Dash/reporting environment (`venv/`):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For nnU-Net tooling use the dedicated `.venv/` environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_nnunet.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

## Tasks List:
- Model exploration
    + How the 3D data is constructed ?
        + Input?         
        + Process?
        + Output?
    + About U-Net architect (theoretically):
        + What is UNet ?
        + Why it is good for medical task
        + Rationale about the architect. (encoder-decoder end-to-end, comparison to YOLO)

- About training process
    + Preprocess the data (3D model)? 
    + Data augmentation?
    + Loss function?
    + Parameters?
    + Time?

- Performance ?

- Data exploration :
    + What is our problem? Detail about segmentation problem; how it is harder than normal detection tasks.
    + Describe articulately the input data; visualization the prediction output and compare it to ground truth