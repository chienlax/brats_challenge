"""Comprehensive BraTS volume visualization tool.

This unified script combines interactive browsing, static image generation, and GIF animation
capabilities into a single command-line interface with multiple modes.

Modes:
    interactive: Launch the Dash web application for interactive volume inspection
    static: Generate static PNG images with modalities or orthogonal views
    gif: Create animated GIF sequences for slice-by-slice visualization

Examples:
    # Interactive mode (Dash app)
    python scripts/visualize_volumes.py --mode interactive --open-browser
    python scripts/visualize_volumes.py --mode interactive --data-root training_data --port 8080

    # Static image mode
    python scripts/visualize_volumes.py --mode static --case-dir training_data_additional/BraTS-GLI-02405-100 \
        --layout modalities --axis axial --fraction 0.55 --output outputs/case_axial.png
    
    python scripts/visualize_volumes.py --mode static --case-dir training_data_additional/BraTS-GLI-02405-100 \
        --layout orthogonal --modality t1c --indices 90 110 70 --output outputs/case_ortho.png

    # GIF animation mode
    python scripts/visualize_volumes.py --mode gif --root training_data_additional --output-dir outputs/gifs \
        --modality t2f --max-cases 3 --overlay
    
    python scripts/visualize_volumes.py --mode gif --case BraTS-GLI-02405-100 \
        --root training_data_additional --axis axial --step 3 --output-dir outputs/gifs
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from threading import Timer
from typing import Iterable, List, Mapping

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


# ============================================================================
# Constants
# ============================================================================

MODALITIES = ("t1c", "t1n", "t2f", "t2w")
LABEL_COLORS = {
    0: "#00000000",
    1: "#1f77b4",  # Enhancing Tumor (blue)
    2: "#d62728",  # Non-Enhancing Tumor (red)
    3: "#2ca02c",  # Peritumoral Edema / SNFH (green)
    4: "#f1c40f",  # Resection Cavity (yellow)
}


# ============================================================================
# Shared Utilities
# ============================================================================

def bootstrap_pythonpath() -> None:
    """Ensure the repository root is on the Python path."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def list_case_dirs(root: Path) -> List[Path]:
    """List all case directories within a dataset root."""
    return sorted([p for p in root.iterdir() if p.is_dir()])


def infer_case_prefix(case_dir: Path) -> str:
    """Extract the case prefix from a case directory name."""
    return case_dir.name


def load_nifti(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI file, raising if not found."""
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def get_volume(path: Path) -> np.ndarray:
    """Load volume data as float32 array."""
    return load_nifti(path).get_fdata(dtype=np.float32)


def normalize(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] using 1st-99th percentile."""
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
    """Extract and orient a 2D slice from a 3D volume."""
    slicer = [slice(None)] * 3
    slicer[axis] = index
    data = volume[tuple(slicer)]
    if axis == 2:
        return np.rot90(data)
    if axis == 1:
        return np.rot90(np.flipud(data))
    return np.rot90(np.flipud(data))


def build_colormap() -> ListedColormap:
    """Construct a colormap for segmentation labels."""
    rgba = [plt.matplotlib.colors.to_rgba(LABEL_COLORS.get(label, "#ffffff")) for label in range(max(LABEL_COLORS) + 1)]
    return ListedColormap(rgba)


# ============================================================================
# Interactive Mode (Dash App)
# ============================================================================

def run_interactive_mode(args: argparse.Namespace) -> None:
    """Launch the interactive Dash volume inspector."""
    bootstrap_pythonpath()
    from apps.volume_inspector import create_dash_app

    data_roots = args.data_root if args.data_root else None

    try:
        app = create_dash_app(data_roots=data_roots)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        sys.exit(2)

    url = f"http://{args.host}:{args.port}"
    if args.open_browser:
        Timer(1.0, lambda: webbrowser.open_new(url)).start()

    print(f"[interactive] Starting Dash app at {url}")
    app.run_server(host=args.host, port=args.port, debug=args.debug)


# ============================================================================
# Static Mode (Single Image)
# ============================================================================

def run_static_mode(args: argparse.Namespace) -> None:
    """Generate a static PNG visualization."""
    bootstrap_pythonpath()
    from apps.common.volume_utils import (
        AXIS_TO_INDEX,
        MODALITY_LABELS,
        build_segmentation_colormap,
        load_segmentation,
        pick_slice_index,
        prepare_slice as volume_utils_prepare_slice,
        prepare_volumes,
        segmentation_legend_handles,
    )

    case_dir = args.case_dir.expanduser().resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory {case_dir} does not exist.")

    case_prefix = infer_case_prefix(case_dir)
    case_id = case_prefix
    volumes = prepare_volumes(case_dir, case_prefix)
    segmentation = load_segmentation(case_dir, case_prefix)

    if args.layout == "modalities":
        reference_volume = next(iter(volumes.values()))
        slice_index = pick_slice_index(reference_volume, args.axis, args.index, args.fraction)
        fig = create_modalities_panel(
            volumes, segmentation, args.axis, slice_index, case_id, overlay=not args.no_overlay, volume_utils_prepare_slice=volume_utils_prepare_slice
        )
    else:
        if args.indices:
            indices = {axis: idx for axis, idx in zip(("sagittal", "coronal", "axial"), args.indices)}
        else:
            vol = next(iter(volumes.values()))
            indices = {
                axis: pick_slice_index(vol, axis, None, args.fraction) for axis in ("sagittal", "coronal", "axial")
            }
        if args.modality not in volumes:
            raise ValueError(f"Requested modality '{args.modality}' was not found for case {case_id}.")
        fig = create_orthogonal_panel(
            volumes[args.modality], segmentation, indices, args.modality, case_id, overlay=not args.no_overlay, volume_utils_prepare_slice=volume_utils_prepare_slice
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"[static] Saved figure to {args.output}")
    if args.show or not args.output:
        plt.show()
    else:
        plt.close(fig)


def create_modalities_panel(
    volumes: Mapping[str, np.ndarray],
    segmentation: np.ndarray | None,
    axis_name: str,
    index: int,
    case_id: str,
    overlay: bool,
    volume_utils_prepare_slice,
) -> plt.Figure:
    """Create a panel showing all modalities for a single slice."""
    bootstrap_pythonpath()
    from apps.common.volume_utils import MODALITY_LABELS, build_segmentation_colormap, segmentation_legend_handles

    cmap_seg, norm_seg = build_segmentation_colormap()
    modalities = [m for m in MODALITY_LABELS if m in volumes]

    fig, axes = plt.subplots(1, len(modalities), figsize=(4.2 * len(modalities), 4.2), constrained_layout=True)
    if len(modalities) == 1:
        axes = [axes]

    for ax, modality in zip(axes, modalities):
        volume = volumes[modality]
        slice_2d = volume_utils_prepare_slice(volume, axis_name, index)
        ax.imshow(slice_2d, cmap="gray", interpolation="none")
        ax.set_title(f"{MODALITY_LABELS[modality]}\n{axis_name.capitalize()} slice {index}")
        ax.axis("off")

        if overlay and segmentation is not None:
            seg_slice = volume_utils_prepare_slice(segmentation, axis_name, index)
            mask = np.ma.masked_where(seg_slice < 0.5, seg_slice)
            ax.imshow(mask, cmap=cmap_seg, norm=norm_seg, alpha=0.45, interpolation="none")

    fig.suptitle(f"BraTS case {case_id}")
    add_segmentation_legend(fig, segmentation_legend_handles)
    return fig


def create_orthogonal_panel(
    volume: np.ndarray,
    segmentation: np.ndarray | None,
    indices: Mapping[str, int],
    modality: str,
    case_id: str,
    overlay: bool,
    volume_utils_prepare_slice,
) -> plt.Figure:
    """Create a panel showing orthogonal views of a single modality."""
    bootstrap_pythonpath()
    from apps.common.volume_utils import MODALITY_LABELS, build_segmentation_colormap, segmentation_legend_handles

    cmap_seg, norm_seg = build_segmentation_colormap()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)

    for ax, axis_name in zip(axes, ("sagittal", "coronal", "axial")):
        idx = indices[axis_name]
        slice_2d = volume_utils_prepare_slice(volume, axis_name, idx)
        ax.imshow(slice_2d, cmap="gray", interpolation="none")
        ax.set_title(f"{axis_name.capitalize()} slice {idx}")
        ax.axis("off")
        if overlay and segmentation is not None:
            seg_slice = volume_utils_prepare_slice(segmentation, axis_name, idx)
            mask = np.ma.masked_where(seg_slice < 0.5, seg_slice)
            ax.imshow(mask, cmap=cmap_seg, norm=norm_seg, alpha=0.45, interpolation="none")

    fig.suptitle(f"BraTS case {case_id} — {MODALITY_LABELS.get(modality, modality)}")
    add_segmentation_legend(fig, segmentation_legend_handles)
    return fig


def add_segmentation_legend(fig: plt.Figure, segmentation_legend_handles_func) -> None:
    """Add a legend for segmentation labels to the figure."""
    handles = segmentation_legend_handles_func()
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)))


# ============================================================================
# GIF Mode (Animation)
# ============================================================================

def run_gif_mode(args: argparse.Namespace) -> None:
    """Generate GIF animations for BraTS cases."""
    root = args.root.expanduser().resolve()
    if args.case:
        case_dirs = [root / args.case]
    else:
        if not root.is_dir():
            raise FileNotFoundError(root)
        case_dirs = list_case_dirs(root)
        if args.max_cases is not None:
            case_dirs = case_dirs[: args.max_cases]

    cmap = build_colormap()
    plt.close("all")

    for idx, case_dir in enumerate(case_dirs, start=1):
        if not case_dir.is_dir():
            raise FileNotFoundError(case_dir)
        print(f"[gif] Rendering GIF for {case_dir.name} ({idx}/{len(case_dirs)})")
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
        print(f"[gif] Saved {gif_path}")


def render_frame(
    volume: np.ndarray,
    segmentation: np.ndarray | None,
    axis_idx: int,
    index: int,
    overlay: bool,
    dpi: int,
    cmap: ListedColormap,
) -> np.ndarray:
    """Render a single frame for GIF animation."""
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
    """Generate a GIF animation for a single case."""
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


# ============================================================================
# CLI Argument Parsing
# ============================================================================

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the unified visualization tool."""
    parser = argparse.ArgumentParser(
        description="Comprehensive BraTS volume visualization tool with multiple modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--mode",
        choices=("interactive", "static", "gif"),
        required=True,
        help="Visualization mode: 'interactive' launches Dash app, 'static' generates PNG, 'gif' creates animations.",
    )

    # Interactive mode arguments
    interactive_group = parser.add_argument_group("interactive mode options")
    interactive_group.add_argument(
        "--data-root",
        type=Path,
        action="append",
        help="Path to BraTS dataset root (repeat for multiple). Defaults to training_data, training_data_additional, validation_data.",
    )
    interactive_group.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1).")
    interactive_group.add_argument("--port", type=int, default=8050, help="Port for Dash app (default: 8050).")
    interactive_group.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    interactive_group.add_argument("--open-browser", action="store_true", help="Automatically open browser.")

    # Static mode arguments
    static_group = parser.add_argument_group("static mode options")
    static_group.add_argument("--case-dir", type=Path, help="Path to a BraTS case directory (required for static mode).")
    static_group.add_argument(
        "--layout",
        choices=("modalities", "orthogonal"),
        default="modalities",
        help="Layout: 'modalities' shows all modalities on one plane, 'orthogonal' shows 3 planes for one modality.",
    )
    static_group.add_argument("--axis", choices=("sagittal", "coronal", "axial"), default="axial", help="Axis for modalities layout.")
    static_group.add_argument("--fraction", type=float, default=0.5, help="Relative slice location [0, 1] (default: 0.5).")
    static_group.add_argument("--index", type=int, help="Explicit slice index for the selected axis.")
    static_group.add_argument(
        "--indices",
        type=int,
        nargs=3,
        metavar=("SAGITTAL", "CORONAL", "AXIAL"),
        help="Explicit slice indices for orthogonal layout (sagittal, coronal, axial).",
    )
    static_group.add_argument("--modality", choices=MODALITIES, default="t1c", help="Modality for orthogonal layout (default: t1c).")
    static_group.add_argument("--no-overlay", action="store_true", help="Disable segmentation overlay.")
    static_group.add_argument("--output", type=Path, help="Output path for saved figure (required for static mode).")
    static_group.add_argument("--dpi", type=int, default=160, help="Resolution for saved figures (default: 160).")
    static_group.add_argument("--show", action="store_true", help="Display figure interactively.")

    # GIF mode arguments
    gif_group = parser.add_argument_group("gif mode options")
    gif_group.add_argument("--root", type=Path, help="Dataset root containing case directories (required for gif mode).")
    gif_group.add_argument("--case", type=str, help="Process a single case ID.")
    gif_group.add_argument("--output-dir", type=Path, help="Directory to store generated GIFs (required for gif mode).")
    gif_group.add_argument("--gif-modality", dest="modality", choices=MODALITIES, default="t2f", help="Modality to animate (default: t2f).")
    gif_group.add_argument("--gif-axis", dest="axis", choices=("sagittal", "coronal", "axial"), default="axial", help="Slice orientation (default: axial).")
    gif_group.add_argument("--step", type=int, default=2, help="Sampling interval between slices (default: 2).")
    gif_group.add_argument("--fps", type=int, default=12, help="Animation frame rate (default: 12).")
    gif_group.add_argument("--overlay", action="store_true", help="Overlay segmentation labels.")
    gif_group.add_argument("--max-cases", type=int, help="Limit number of cases when scanning the root.")
    gif_group.add_argument("--gif-dpi", dest="dpi", type=int, default=100, help="Figure DPI for frame rendering (default: 100).")

    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate mode-specific required arguments."""
    if args.mode == "static":
        if not args.case_dir:
            raise ValueError("--case-dir is required for static mode.")
        if not args.output:
            raise ValueError("--output is required for static mode.")
    elif args.mode == "gif":
        if not args.root:
            raise ValueError("--root is required for gif mode.")
        if not args.output_dir:
            raise ValueError("--output-dir is required for gif mode.")


# ============================================================================
# Main Entry Point
# ============================================================================

def main(argv: Iterable[str] | None = None) -> None:
    """Main entry point for the unified visualization tool."""
    args = parse_args(argv)
    
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"[error] {exc}")
        sys.exit(1)

    try:
        if args.mode == "interactive":
            run_interactive_mode(args)
        elif args.mode == "static":
            run_static_mode(args)
        elif args.mode == "gif":
            run_gif_mode(args)
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
