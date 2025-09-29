from __future__ import annotations

import numpy as np

from apps.volume_inspector import rendering


def test_compose_rgba_slice_scale_multiplier() -> None:
    volume = np.linspace(0.0, 1.0, num=16, dtype=np.float32).reshape(4, 4)
    rgba_standard = rendering.compose_rgba_slice(volume, None, overlay=False, scale=1)
    rgba_high = rendering.compose_rgba_slice(volume, None, overlay=False, scale=2)
    rgba_best = rendering.compose_rgba_slice(volume, None, overlay=False, scale=4)

    assert rgba_standard.shape[:2] == (4, 4)
    assert rgba_high.shape[:2] == (8, 8)
    assert rgba_best.shape[:2] == (16, 16)
    # Alpha channel preserved
    assert rgba_standard.shape[2] == 4
    assert rgba_high.shape[2] == 4
    assert rgba_best.shape[2] == 4


def test_compose_rgba_slice_scales_segmentation() -> None:
    volume = np.zeros((2, 2), dtype=np.float32)
    segmentation = np.array([[0, 1], [2, 0]], dtype=np.int16)
    rgba = rendering.compose_rgba_slice(volume, segmentation, overlay=True, scale=2)

    # Expect the two labeled regions to expand with scaling and retain non-zero alpha
    assert rgba.shape[:2] == (4, 4)
    alpha_channel = rgba[..., 3]
    assert np.any(alpha_channel > 0)


def test_compose_modalities_grid_arranges_all_modalities() -> None:
    base = np.linspace(0.0, 1.0, num=64, dtype=np.float32).reshape(4, 4, 4)
    volumes = {
        "t1c": base,
        "t1n": base + 0.1,
        "t2w": base + 0.2,
        "t2f": base + 0.3,
    }
    segmentation = np.zeros_like(base, dtype=np.int16)
    canvas, locations, cell_h, cell_w = rendering.compose_modalities_grid(
        volumes,
        segmentation,
        axis_name="axial",
        index=1,
        overlay=False,
        scale=1,
    )

    assert canvas.shape == (8, 8, 4)
    assert len(list(locations)) == 4
    assert cell_h == 4 and cell_w == 4
