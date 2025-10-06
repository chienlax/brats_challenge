from pathlib import Path
import torch.nn as nn
from huggingface_hub import snapshot_download
from monai.bundle import load as bundle_load

# --- Configuration ---
BUNDLE_NAME = "brats_mri_segmentation"
ARTIFACTS_DIR = Path("artifacts")
# ---

# Ensure the bundle is downloaded
local_bundle_path = ARTIFACTS_DIR / BUNDLE_NAME
if not local_bundle_path.exists():
    repo_id = f"MONAI/{BUNDLE_NAME}"
    print(f"Downloading bundle to {local_bundle_path}...")
    snapshot_download(repo_id=repo_id, local_dir=local_bundle_path, repo_type="model")

# Load the bundle
print(f"Loading bundle from: {local_bundle_path}")
loaded_bundle = bundle_load(name=BUNDLE_NAME, bundle_dir=local_bundle_path)

# Robustly get the network object
network = None
if isinstance(loaded_bundle, dict):
    network = loaded_bundle.get("network")
elif isinstance(loaded_bundle, nn.Module):
    # The bundle returned the model object directly
    network = loaded_bundle

if network is None:
    raise RuntimeError("Could not extract the network model from the MONAI bundle.")

# Print the entire model architecture
print("\n--- Model Architecture ---")
print(network)
print("--------------------------\n")

print("The final output module in this SegResNet architecture is named 'seg_layers'.")
print("This is the value you must pass to --head-key.")