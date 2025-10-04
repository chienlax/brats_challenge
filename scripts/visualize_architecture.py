"""Visualize nnU-Net architecture definitions from a plans.json file."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, RegularPolygon


def format_kernel(kernel: List[int]) -> str:
    return "×".join(str(k) for k in kernel)


def format_stride(stride: List[int]) -> str:
    return "×".join(str(s) for s in stride)


def load_plans(plans_path: Path) -> Dict[str, Any]:
    if not plans_path.exists():
        raise FileNotFoundError(f"Plans file not found: {plans_path}")
    with plans_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_architecture(plans: Dict[str, Any], configuration: str) -> Dict[str, Any]:
    try:
        config_block = plans["configurations"][configuration]
    except KeyError as exc:
        raise KeyError(
            f"Configuration '{configuration}' not found in plans file."
        ) from exc

    arch = config_block.get("architecture")
    if arch is None:
        raise KeyError(
            f"No architecture definition found for configuration '{configuration}'."
        )

    arch_kwargs = arch.get("arch_kwargs", {})
    return {
        "network_class_name": arch.get("network_class_name", "unknown"),
        "arch_kwargs": arch_kwargs,
        "normalization": config_block.get("normalization_schemes", []),
        "use_mask_for_norm": config_block.get("use_mask_for_norm", []),
        "spacing": config_block.get("spacing"),
        "patch_size": config_block.get("patch_size"),
        "batch_size": config_block.get("batch_size"),
    }


def architecture_to_frame(arch_kwargs: Dict[str, Any]) -> pd.DataFrame:
    stages = arch_kwargs["n_stages"]
    features = arch_kwargs["features_per_stage"]
    kernels = arch_kwargs["kernel_sizes"]
    strides = arch_kwargs["strides"]
    enc_convs = arch_kwargs.get("n_conv_per_stage", [])
    dec_convs = arch_kwargs.get("n_conv_per_stage_decoder", [])

    rows = []
    for stage_idx in range(stages):
        rows.append(
            {
                "Stage": f"Encoder {stage_idx}",
                "Features": features[stage_idx],
                "Kernel": format_kernel(kernels[stage_idx]),
                "Stride": format_stride(strides[stage_idx]),
                "Conv blocks": enc_convs[stage_idx] if stage_idx < len(enc_convs) else "-",
                "Decoder conv blocks": "-",
            }
        )

    decoder_features = list(reversed(features[:-1]))
    decoder_kernels = list(reversed(kernels[:-1]))
    decoder_strides = list(reversed(strides[1:]))
    for idx, feat in enumerate(decoder_features):
        rows.append(
            {
                "Stage": f"Decoder {idx}",
                "Features": feat,
                "Kernel": format_kernel(decoder_kernels[idx]) if idx < len(decoder_kernels) else "-",
                "Stride": format_stride(decoder_strides[idx]) if idx < len(decoder_strides) else "-",
                "Conv blocks": "-",
                "Decoder conv blocks": dec_convs[idx] if idx < len(dec_convs) else "-",
            }
        )

    return pd.DataFrame(rows)


def build_figure(arch_info: Dict[str, Any], output_path: Path | None = None) -> Path | None:
    arch_kwargs = arch_info["arch_kwargs"]
    df = architecture_to_frame(arch_kwargs)

    encoder_features = arch_kwargs["features_per_stage"]
    decoder_features = list(reversed(arch_kwargs["features_per_stage"][:-1]))

    stages = range(len(encoder_features))
    decoder_stage_positions = list(range(1, len(decoder_features) + 1))

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        f"{arch_info['network_class_name']} architecture overview", fontsize=16, fontweight="bold"
    )

    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.5])

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(stages, encoder_features, marker="o", label="Encoder features")
    if decoder_features:
        ax1.plot(
            decoder_stage_positions,
            decoder_features,
            marker="s",
            label="Decoder features (mirrored)",
        )
    ax1.set_xlabel("Stage")
    ax1.set_ylabel("Feature channels")
    ax1.set_xticks(list(stages))
    xtick_labels = []
    for idx in stages:
        decoder_idx = idx - 1
        decoder_label = (
            f" / D{decoder_idx}" if decoder_idx >= 0 and decoder_idx < len(decoder_features) else ""
        )
        xtick_labels.append(f"E{idx}{decoder_label}")
    ax1.set_xticklabels(xtick_labels)
    ax1.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    ax1.legend()

    ax1.text(
        0.02,
        0.95,
        "\n".join(
            [
                f"Patch size: {arch_info['patch_size']}",
                f"Spacing: {arch_info['spacing']}",
                f"Batch size: {arch_info['batch_size']}",
            ]
        ),
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    table = ax2.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path
    else:
        plt.show()
        return None


def build_detailed_graph(
    arch_info: Dict[str, Any], output_path: Path | None = None
) -> Path | None:
    arch_kwargs = arch_info["arch_kwargs"]
    features = arch_kwargs["features_per_stage"]
    kernels = arch_kwargs["kernel_sizes"]
    strides = arch_kwargs["strides"]
    enc_convs = arch_kwargs.get("n_conv_per_stage", [])

    decoder_features = list(reversed(features[:-1]))
    decoder_kernels = list(reversed(kernels[:-1]))
    decoder_strides = list(reversed(strides[1:]))
    dec_convs = arch_kwargs.get("n_conv_per_stage_decoder", [])

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title(
        f"Detailed {arch_info['network_class_name']} architecture", fontsize=16, fontweight="bold"
    )
    ax.axis("off")

    spacing = 3.0
    block_width = 2.0
    block_height = 1.5
    encoder_y = 2.2
    decoder_y = -2.2

    def draw_block(x: float, y: float, label: str, color: str) -> None:
        rect = FancyBboxPatch(
            (x - block_width / 2, y - block_height / 2),
            block_width,
            block_height,
            boxstyle="round,pad=0.3",
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.25,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    encoder_positions: Dict[int, tuple[float, float]] = {}
    all_positions_x: List[float] = []
    all_positions_y: List[float] = []
    for idx, feat in enumerate(features):
        x = idx * spacing
        stage_name = "Bottleneck" if idx == len(features) - 1 else f"Encoder {idx}"
        stride_str = format_stride(strides[idx])
        kernel_str = format_kernel(kernels[idx])
        conv_count = enc_convs[idx] if idx < len(enc_convs) else "-"
        label_lines = [
            stage_name,
            f"{feat} ch",
            f"k={kernel_str}",
            f"s={stride_str}",
        ]
        if conv_count not in ("-", None):
            label_lines.append(f"conv×{conv_count}")
        draw_block(x, encoder_y, "\n".join(label_lines), "tab:blue")
        encoder_positions[idx] = (x, encoder_y)
        all_positions_x.append(x)
        all_positions_y.append(encoder_y)

    decoder_positions: Dict[int, tuple[float, float]] = {}
    decoder_count = len(decoder_features)
    for idx, feat in enumerate(decoder_features):
        x = (decoder_count - 1 - idx) * spacing
        stride = (
            format_stride(decoder_strides[idx]) if idx < len(decoder_strides) else "1×1×1"
        )
        kernel = (
            format_kernel(decoder_kernels[idx]) if idx < len(decoder_kernels) else "-"
        )
        conv_count = dec_convs[idx] if idx < len(dec_convs) else "-"
        label_lines = [
            f"Decoder {idx}",
            f"{feat} ch",
            f"k={kernel}",
            f"s={stride}",
        ]
        if conv_count not in ("-", None):
            label_lines.append(f"conv×{conv_count}")
        draw_block(x, decoder_y, "\n".join(label_lines), "tab:green")
        decoder_positions[idx] = (x, decoder_y)
        all_positions_x.append(x)
        all_positions_y.append(decoder_y)

    # Downsampling path connections between encoder stages
    downsample_icon_size = 0.3
    skip_color = "#8f3bff"

    for idx in range(len(features) - 1):
        x0, y0 = encoder_positions[idx]
        x1, y1 = encoder_positions[idx + 1]
        ax.annotate(
            "",
            xy=(x1 - block_width / 2, y1),
            xytext=(x0 + block_width / 2, y0),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.6, color="tab:blue"),
        )
        ax.text(
            (x0 + x1) / 2,
            y0 + 0.7,
            f"stride {format_stride(strides[idx + 1])}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="tab:blue",
        )
        # Downsampling icon (triangle pointing down)
        mid_x = (x0 + x1) / 2
        mid_y = y0 - block_height / 2 - 0.2
        triangle = RegularPolygon(
            (mid_x, mid_y),
            numVertices=3,
            radius=downsample_icon_size,
            orientation=-0.5 * math.pi,
            color="tab:blue",
            alpha=0.6,
        )
        ax.add_patch(triangle)

    # Upsampling path connections between decoder stages
    for idx in range(decoder_count - 1):
        x0, y0 = decoder_positions[idx]
        x1, y1 = decoder_positions[idx + 1]
        ax.annotate(
            "",
            xy=(x1 + block_width / 2, y1),
            xytext=(x0 - block_width / 2, y0),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.6, color="tab:green"),
        )
        ax.text(
            (x0 + x1) / 2,
            y0 - 0.7,
            f"stride {format_stride(decoder_strides[idx]) if idx < len(decoder_strides) else '1×1×1'}",
            ha="center",
            va="top",
            fontsize=9,
            color="tab:green",
        )
        # Upsampling icon (triangle pointing up)
        mid_x = (x0 + x1) / 2
        mid_y = y0 + block_height / 2 + 0.2
        triangle = RegularPolygon(
            (mid_x, mid_y),
            numVertices=3,
            radius=downsample_icon_size,
            orientation=0.5 * math.pi,
            color="tab:green",
            alpha=0.6,
        )
        ax.add_patch(triangle)

    # Skip connections from encoder stages to decoder stages
    for enc_idx in range(len(features) - 1):
        dec_idx = decoder_count - 1 - enc_idx
        if dec_idx < 0 or dec_idx >= decoder_count:
            continue
        enc_x, enc_y = encoder_positions[enc_idx]
        dec_x, dec_y = decoder_positions[dec_idx]
        connection = ConnectionPatch(
            xyA=(dec_x, dec_y + block_height / 2),
            xyB=(enc_x, enc_y - block_height / 2),
            coordsA="data",
            coordsB="data",
            axesA=ax,
            axesB=ax,
            arrowstyle="-|>",
            linewidth=1.4,
            linestyle="-",
            color=skip_color,
        )
        ax.add_artist(connection)
        ax.text(
            (enc_x + dec_x) / 2,
            (enc_y + dec_y) / 2,
            "skip",
            fontsize=8,
            color=skip_color,
            rotation=-35 if enc_x > dec_x else 35,
            ha="center",
            va="center",
        )

    # Context box
    info_lines = [
        f"Patch size: {arch_info['patch_size']}",
        f"Spacing: {arch_info['spacing']}",
        f"Batch size: {arch_info['batch_size']}",
        f"Encoder blocks: {len(features)}",
        f"Decoder blocks: {decoder_count}",
    ]
    ax.text(
        -1.2 * spacing,
        0,
        "\n".join(info_lines),
        fontsize=10,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )
    all_positions_x.append(-1.2 * spacing)
    all_positions_y.extend([encoder_y, decoder_y, 0])

    if all_positions_x:
        min_x = min(all_positions_x) - spacing
        max_x = max(all_positions_x) + spacing
    else:
        min_x, max_x = -1, 1

    min_y = min(all_positions_y) - block_height * 2
    max_y = max(all_positions_y) + block_height * 2

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path
    else:
        plt.show()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plans", type=Path, help="Path to plans.json file")
    parser.add_argument(
        "--config",
        type=str,
        default="3d_fullres",
        help="Configuration key to visualize (default: 3d_fullres)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the overview figure (PNG). If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--detailed-output",
        type=Path,
        default=None,
        help="Optional path to save the detailed architecture diagram. Defaults to '<output>_detailed.png' if --output is provided.",
    )
    parser.add_argument(
        "--vector-output",
        type=Path,
        default=None,
        help="Optional path to save the overview figure as a vector graphic (suffix controls format). Defaults to '<output>.svg' if --output is provided.",
    )
    parser.add_argument(
        "--detailed-vector-output",
        type=Path,
        default=None,
        help="Optional path to save the detailed diagram as a vector graphic (suffix controls format). Defaults to '<detailed-output>.svg' if either output flag is provided.",
    )
    args = parser.parse_args()

    plans = load_plans(args.plans)
    arch_info = extract_architecture(plans, args.config)

    df = architecture_to_frame(arch_info["arch_kwargs"])
    print(f"Configuration: {args.config}")
    print(df.to_string(index=False))

    result_path = build_figure(arch_info, args.output)
    if result_path:
        print(f"Visualization written to: {result_path}")

    vector_output = args.vector_output
    if vector_output is None and args.output is not None:
        vector_output = args.output.with_suffix(".svg")

    if vector_output:
        vector_path = build_figure(arch_info, vector_output)
        if vector_path:
            print(f"Vector overview written to: {vector_path}")

    detailed_output = args.detailed_output
    if detailed_output is None and args.output is not None:
        detailed_output = args.output.with_name(
            f"{args.output.stem}_detailed{args.output.suffix}"
        )

    detailed_path = build_detailed_graph(arch_info, detailed_output)
    if detailed_path:
        print(f"Detailed visualization written to: {detailed_path}")

    detailed_vector_output = args.detailed_vector_output
    if detailed_vector_output is None and detailed_output is not None:
        detailed_vector_output = detailed_output.with_suffix(".svg")

    if detailed_vector_output:
        detailed_vector_path = build_detailed_graph(arch_info, detailed_vector_output)
        if detailed_vector_path:
            print(f"Detailed vector visualization written to: {detailed_vector_path}")


if __name__ == "__main__":
    main()
