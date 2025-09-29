"""Configuration constants for the volume inspector UI."""

from __future__ import annotations

DISPLAY_MODES = {
    "standard": {
        "label": "Standard",
        "scale": 1,
        "description": "Fast rendering",
    },
    "high": {
        "label": "High (2×)",
        "scale": 2,
        "description": "Sharper slices (higher CPU)",
    },
}

DEFAULT_DISPLAY_MODE = "standard"
VIEW_MODES = {
    "single": {
        "label": "Single modality",
    },
    "all": {
        "label": "All modalities",
    },
}
DEFAULT_VIEW_MODE = "single"


def display_mode_scale(mode_key: str | None) -> int:
    """Resolve the pixel scale multiplier for a display mode key."""
    if not mode_key:
        return DISPLAY_MODES[DEFAULT_DISPLAY_MODE]["scale"]
    config = DISPLAY_MODES.get(mode_key)
    if not config:
        return DISPLAY_MODES[DEFAULT_DISPLAY_MODE]["scale"]
    return int(config["scale"])


__all__ = [
    "DISPLAY_MODES",
    "DEFAULT_DISPLAY_MODE",
    "display_mode_scale",
    "VIEW_MODES",
    "DEFAULT_VIEW_MODE",
]
