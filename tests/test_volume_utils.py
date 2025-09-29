from __future__ import annotations

import numpy as np

from apps.common import volume_utils as vu


def test_normalize_intensity_scales_between_zero_and_one() -> None:
    volume = np.linspace(-100, 100, num=1000, dtype=np.float32).reshape(10, 10, 10)
    normalized = vu.normalize_intensity(volume)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_pick_slice_index_from_fraction() -> None:
    volume = np.zeros((20, 30, 40), dtype=np.float32)
    index = vu.pick_slice_index(volume, "axial", index=None, fraction=0.5)
    assert index == 20  # 0-based index nearest to the midpoint of 40 slices


def test_prepare_slice_returns_expected_shape() -> None:
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    axial_slice = vu.prepare_slice(volume, "axial", 0)
    coronal_slice = vu.prepare_slice(volume, "coronal", 0)
    sagittal_slice = vu.prepare_slice(volume, "sagittal", 0)

    assert axial_slice.shape == (3, 2)
    assert coronal_slice.shape == (4, 2)
    assert sagittal_slice.shape == (4, 3)
