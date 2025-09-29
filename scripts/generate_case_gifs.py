"""Batch GIF generation for BraTS post-treatment cases.

This utility renders slice-by-slice animations for a selected modality and axis,
optionally overlaying segmentation masks. It reuses the same normalization scheme as
our visualization tools and is intended for quick QC reels or presentations.

Examples:
    python scripts/generate_case_gifs.py --root training_data_additional --output-dir outputs/gifs --modality t2f --max-cases 3
    python scripts/generate_case_gifs.py --case BraTS-GLI-02405-100 --axis axial --step 3 --overlay --output-dir outputs/gifs
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap

MODALITIES = ("t1c", "t1n", "t2f", "t2w")
LABEL_COLORS = {
    0: "#00000000",
    1: "#1f77b4",  # Enhancing Tumor (blue)
    2: "#d62728",  # Non-Enhancing Tumor (red)
    3: "#2ca02c",  # Peritumoral Edema / SNFH (green)
    4: "#f1c40f",  # Resection Cavity (yellow)
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate slice GIFs for BraTS cases.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("training_data_additional"),
        help="Dataset root containing case directories.",
    )
    parser.add_argument("--case", type=str, help="Process a single case id.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store generated GIFs.")
    parser.add_argument("--modality", choices=MODALITIES, default="t2f", help="Modality to animate.")
    parser.add_argument("--axis", choices=("sagittal", "coronal", "axial"), default="axial", help="Slice orientation.")
    parser.add_argument("--step", type=int, default=2, help="Sampling interval between slices.")
    parser.add_argument("--fps", type=int, default=12, help="Animation frame rate.")
    parser.add_argument("--overlay", action="store_true", help="Overlay segmentation labels.")
    parser.add_argument("--max-cases", type=int, help="Limit number of cases when scanning the root.")
    parser.add_argument("--dpi", type=int, default=100, help="Figure DPI for frame rendering.")
    return parser.parse_args(argv)


def list_case_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def infer_case_prefix(case_dir: Path) -> str:
    return case_dir.name


def load_nifti(path: Path) -> nib.Nifti1Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def get_volume(path: Path) -> np.ndarray:
    return load_nifti(path).get_fdata(dtype=np.float32)


def normalize(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    if not finite.any():
        return np.zeros_like(volume)
    data = volume[finite]
    lo, hi = np.percentile(data, (1, 99))
    if np.isclose(lo, hi):
        lo, hi = data.min(), data.max()
    if np.isclose(lo, hi):
        return np.clip(volume, 0.0, 1.0)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0)


def prepare_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    slicer = [slice(None)] * 3
    slicer[axis] = index
    data = volume[tuple(slicer)]
    if axis == 2:
        return np.rot90(data)
    if axis == 1:
        return np.rot90(np.flipud(data))
    return np.rot90(np.flipud(data))


def build_colormap() -> ListedColormap:
    rgba = [plt.matplotlib.colors.to_rgba(LABEL_COLORS.get(label, "#ffffff")) for label in range(max(LABEL_COLORS) + 1)]
    return ListedColormap(rgba)


def render_frame(
    volume: np.ndarray,
    segmentation: np.ndarray | None,
    axis_idx: int,
    index: int,
    overlay: bool,
    dpi: int,
    cmap: ListedColormap,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    ax.imshow(prepare_slice(volume, axis_idx, index), cmap="gray", interpolation="none")
    if overlay and segmentation is not None:
        seg_slice = prepare_slice(segmentation, axis_idx, index)
        mask = np.ma.masked_where(seg_slice < 0.5, seg_slice)
        ax.imshow(mask, cmap=cmap, alpha=0.45, interpolation="none")
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.renderer.buffer_rgba())
    frame = buffer[:, :, :3].copy()
    plt.close(fig)
    return frame


def create_gif_for_case(
    case_dir: Path,
    modality: str,
    axis: str,
    step: int,
    overlay: bool,
    fps: int,
    dpi: int,
    output_dir: Path,
    cmap: ListedColormap,
) -> Path:
    case_id = infer_case_prefix(case_dir)
    volume_path = case_dir / f"{case_id}-{modality}.nii.gz"
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    volume = normalize(get_volume(volume_path))
    segmentation = np.rint(get_volume(seg_path)).astype(np.int16) if overlay and seg_path.exists() else None

    axis_idx = {"sagittal": 0, "coronal": 1, "axial": 2}[axis]

    frames = [
        render_frame(volume, segmentation, axis_idx, idx, overlay, dpi, cmap)
        for idx in range(0, volume.shape[axis_idx], step)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"{case_id}_{modality}_{axis}.gif"
    imageio.mimsave(gif_path, frames, fps=fps)
    return gif_path


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.case:
        case_dirs = [root / args.case]
    else:
        if not root.is_dir():
            raise FileNotFoundError(root)
        case_dirs = list_case_dirs(root)
        if args.max_cases is not None:
            case_dirs = case_dirs[: args.max_cases]

    cmap = build_colormap()  # warm-up to ensure matplotlib caches colormap
    plt.close("all")

    for idx, case_dir in enumerate(case_dirs, start=1):
        if not case_dir.is_dir():
            raise FileNotFoundError(case_dir)
        print(f"Rendering GIF for {case_dir.name} ({idx}/{len(case_dirs)})")
        gif_path = create_gif_for_case(
            case_dir,
            modality=args.modality,
            axis=args.axis,
            step=max(args.step, 1),
            overlay=args.overlay,
            fps=args.fps,
            dpi=args.dpi,
            output_dir=args.output_dir.expanduser().resolve(),
            cmap=cmap,
        )
        print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
