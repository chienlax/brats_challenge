"""Pre-render interactive 3D BraTS volumes as standalone HTML files.

This utility scans a dataset root for BraTS cases, renders each requested
modality as a Plotly volume, optionally overlays segmentation isosurfaces, and
writes self-contained HTML viewers to an output directory.

Example usage:
    python scripts/precompute_volume_visuals.py --root training_data_additional \
        --output-dir outputs/volumes --modality t1c --with-segmentation --downsample 2

    python scripts/precompute_volume_visuals.py --case BraTS-GLI-02405-100 --output-dir outputs/volumes

"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib   
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

MODALITY_LABELS: Mapping[str, str] = {
    "t1c": "T1-weighted (post-contrast)",
    "t1n": "T1-weighted (pre-contrast)",
    "t2f": "T2 FLAIR",
    "t2w": "T2-weighted",
}

SEGMENTATION_LABELS: Mapping[int, Tuple[str, str]] = {
    0: ("Background", "#00000000"),
    1: ("Enhancing Tumor", "#1f77b4"),
    2: ("Non-Enhancing Tumor", "#d62728"),
    3: ("Peritumoral Edema / SNFH", "#2ca02c"),
    4: ("Resection Cavity", "#f1c40f"),
}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    output_path: Optional[Path]
    error: Optional[str] = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-render interactive 3D BraTS volumes.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("training_data_additional"),
        help="Dataset root containing case directories (default: training_data_additional).",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="Specific case identifier to process. Can be passed multiple times.",
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        help="Optional newline-delimited text file listing case identifiers to process.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Limit the number of discovered cases to process (after any filtering).",
    )
    parser.add_argument(
        "--modality",
        choices=tuple(MODALITY_LABELS),
        default="t1c",
        help="Imaging modality to render (default: t1c).",
    )
    parser.add_argument(
        "--with-segmentation",
        action="store_true",
        help="Overlay segmentation isosurfaces when segmentation volumes are present.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Stride used to subsample voxels before rendering (>=1, default: 2).",
    )
    parser.add_argument(
        "--intensity-opacity",
        type=float,
        default=0.12,
        help="Base opacity for the intensity volume (default: 0.12).",
    )
    parser.add_argument(
        "--segmentation-opacity",
        type=float,
        default=0.35,
        help="Opacity for segmentation isosurfaces (default: 0.35).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/volumes"),
        help="Directory where HTML files will be written (default: outputs/volumes).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases whose HTML output already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HTML files rather than skipping them.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open each generated HTML file in a browser after rendering (use sparingly).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort immediately on the first failure instead of continuing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional details while processing cases.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    step = max(int(args.downsample), 1)

    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite are mutually exclusive.")

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root {root} does not exist.")

    requested_cases = collect_requested_cases(args.cases, args.cases_file)
    case_dirs = discover_case_dirs(root, requested_cases)
    if args.max_cases is not None:
        case_dirs = case_dirs[: max(args.max_cases, 0)]

    if not case_dirs:
        raise FileNotFoundError("No matching case directories were found.")

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[CaseResult] = []
    for idx, case_dir in enumerate(case_dirs, start=1):
        case_id = case_dir.name
        print(f"[{idx}/{len(case_dirs)}] Rendering {case_id}…")
        try:
            result = render_case(
                case_dir=case_dir,
                modality=args.modality,
                include_segmentation=args.with_segmentation,
                downsample=step,
                intensity_opacity=args.intensity_opacity,
                segmentation_opacity=args.segmentation_opacity,
                output_root=output_root,
                skip_existing=args.skip_existing,
                overwrite=args.overwrite,
                auto_open=args.open,
                verbose=args.verbose,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            message = f"Failed to render {case_id}: {exc}"
            print(message, file=sys.stderr)
            results.append(CaseResult(case_id=case_id, output_path=None, error=str(exc)))
            if args.strict:
                raise

    summarize_results(results)


def collect_requested_cases(case_args: Optional[Sequence[str]], cases_file: Optional[Path]) -> Optional[List[str]]:
    cases: List[str] = []
    if case_args:
        cases.extend(case_args)
    if cases_file:
        file_path = cases_file.expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Case list file {file_path} was not found.")
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                clean = line.strip()
                if clean:
                    cases.append(clean)
    if not cases:
        return None
    seen = set()
    unique_cases = []
    for case in cases:
        if case not in seen:
            unique_cases.append(case)
            seen.add(case)
    return unique_cases


def discover_case_dirs(root: Path, requested_cases: Optional[Sequence[str]]) -> List[Path]:
    if requested_cases is not None:
        dirs = []
        for case in requested_cases:
            candidate = (root / case).expanduser().resolve()
            if candidate.is_dir():
                dirs.append(candidate)
            else:
                raise FileNotFoundError(f"Requested case directory {candidate} does not exist.")
        return dirs
    return sorted([p for p in root.iterdir() if p.is_dir()])


def render_case(
    case_dir: Path,
    modality: str,
    include_segmentation: bool,
    downsample: int,
    intensity_opacity: float,
    segmentation_opacity: float,
    output_root: Path,
    skip_existing: bool,
    overwrite: bool,
    auto_open: bool,
    verbose: bool,
) -> CaseResult:
    case_id = case_dir.name
    case_prefix = infer_case_prefix(case_dir)
    volume_path = case_dir / f"{case_prefix}-{modality}.nii.gz"
    if not volume_path.exists():
        raise FileNotFoundError(f"Missing modality file: {volume_path}")

    fig = build_case_figure(
        volume_path=volume_path,
        case_id=case_id,
        modality=modality,
        include_segmentation=include_segmentation,
        segmentation_path=case_dir / f"{case_prefix}-seg.nii.gz",
        downsample=downsample,
        intensity_opacity=intensity_opacity,
        segmentation_opacity=segmentation_opacity,
        verbose=verbose,
    )

    output_path = output_root / f"{case_id}_{modality}_volume.html"
    if output_path.exists() and skip_existing:
        if verbose:
            print(f"  Skipping {case_id}; output already exists at {output_path}")
        return CaseResult(case_id=case_id, output_path=output_path)
    if output_path.exists() and not overwrite and not skip_existing:
        raise FileExistsError(f"Output already exists (use --overwrite or --skip-existing): {output_path}")

    pio.write_html(fig, file=str(output_path), auto_open=auto_open, include_plotlyjs="cdn")
    if verbose:
        print(f"  Wrote HTML viewer to {output_path}")
    return CaseResult(case_id=case_id, output_path=output_path)


def build_case_figure(
    volume_path: Path,
    case_id: str,
    modality: str,
    include_segmentation: bool,
    segmentation_path: Path,
    downsample: int,
    intensity_opacity: float,
    segmentation_opacity: float,
    verbose: bool,
) -> go.Figure:
    volume_data, spacing = load_volume(volume_path)
    normalized_volume = normalize_intensity(volume_data)
    downsampled_volume = downsample_array(normalized_volume, downsample)
    coords = build_coordinate_grid(downsampled_volume.shape, spacing, downsample)

    fig = go.Figure()
    fig.add_trace(
        create_volume_trace(
            downsampled_volume,
            coords,
            modality_label=MODALITY_LABELS.get(modality, modality),
            opacity=intensity_opacity,
        )
    )

    if include_segmentation and segmentation_path.exists():
        segmentation = load_segmentation(segmentation_path)
        downsampled_seg = downsample_array(segmentation, downsample)
        add_segmentation_traces(
            fig,
            downsampled_seg,
            coords,
            opacity=segmentation_opacity,
            verbose=verbose,
        )
    elif include_segmentation and verbose:
        print(f"  Segmentation file missing, skipping overlay: {segmentation_path}")

    fig.update_layout(
        title=f"BraTS case {case_id} — {MODALITY_LABELS.get(modality, modality)}",
        scene=dict(
            xaxis_title="i (voxel)",
            yaxis_title="j (voxel)",
            zaxis_title="k (voxel)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def load_volume(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    return data, spacing


def load_segmentation(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return np.rint(data).astype(np.int16)


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(volume)
    if not np.any(finite_mask):
        return np.zeros_like(volume, dtype=np.float32)
    cleaned = volume[finite_mask]
    lo, hi = np.percentile(cleaned, (1, 99))
    if np.isclose(lo, hi):
        lo, hi = cleaned.min(), cleaned.max()
    if np.isclose(lo, hi):
        return np.clip(volume, 0.0, 1.0)
    normalized = (volume - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)


def downsample_array(array: np.ndarray, step: int) -> np.ndarray:
    if step <= 1:
        return array
    slicer = tuple(slice(None, None, step) for _ in range(array.ndim))
    return array[slicer]


def build_coordinate_grid(
    shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [np.arange(dim, dtype=np.float32) * spacing[idx] * step for idx, dim in enumerate(shape)]
    return np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")


def create_volume_trace(
    volume: np.ndarray,
    coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    modality_label: str,
    opacity: float,
) -> go.Volume:
    x, y, z = [axis.astype(np.float32).ravel() for axis in coords]
    values = volume.astype(np.float32).ravel()
    return go.Volume(
        name=f"{modality_label} intensity",
        x=x,
        y=y,
        z=z,
        value=values,
        opacity=max(min(opacity, 1.0), 0.01),
        surface_count=18,
        colorscale="Gray",
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )


def add_segmentation_traces(
    fig: go.Figure,
    segmentation: np.ndarray,
    coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    opacity: float,
    verbose: bool,
) -> None:
    x, y, z = [axis.astype(np.float32).ravel() for axis in coords]
    flattened = segmentation.ravel()
    unique_labels = [label for label in np.unique(flattened) if label != 0]
    if not unique_labels and verbose:
        print("  Segmentation contains no foreground labels after downsampling.")
    for label in unique_labels:
        name, hex_color = SEGMENTATION_LABELS.get(label, (f"Label {label}", "#FFFFFF"))
        fig.add_trace(
            go.Isosurface(
                name=f"Seg: {name}",
                x=x,
                y=y,
                z=z,
                value=flattened,
                isomin=label - 0.49,
                isomax=label + 0.49,
                surface_count=1,
                opacity=max(min(opacity, 1.0), 0.05),
                colorscale=[[0.0, hex_color], [1.0, hex_color]],
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        )


def infer_case_prefix(case_dir: Path) -> str:
    seg_files = list(case_dir.glob("*-seg.nii.gz"))
    if seg_files:
        return seg_files[0].name.replace("-seg.nii.gz", "")
    # Fallback to directory name when segmentation is absent.
    return case_dir.name


def summarize_results(results: Sequence[CaseResult]) -> None:
    successes = [r for r in results if r.error is None]
    failures = [r for r in results if r.error is not None]
    print()
    print(f"Completed rendering for {len(results)} case(s).")
    print(f"  Successes: {len(successes)}")
    print(f"  Failures : {len(failures)}")
    if failures:
        print("  Failed cases:")
        for result in failures:
            print(f"    - {result.case_id}: {result.error}")


if __name__ == "__main__":
    main()
