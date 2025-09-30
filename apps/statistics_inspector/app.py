"""Main Dash application factory for statistics inspector."""
from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc

from .callbacks import register_callbacks
from .data import StatisticsRepository
from .layout import create_layout


def create_dash_app(data_roots: list[Path] | None = None) -> dash.Dash:
    """Create and configure the statistics inspector Dash application.
    
    Args:
        data_roots: Optional list of dataset root directories. If None, uses defaults.
    
    Returns:
        Configured Dash application instance.
    """
    # Initialize data repository
    repo = StatisticsRepository(data_roots=data_roots)
    datasets = repo.get_datasets()
    
    if not datasets:
        raise FileNotFoundError(
            "No valid dataset roots found. Ensure at least one of training_data, "
            "training_data_additional, or validation_data exists."
        )
    
    # Create Dash app with Bootstrap theme
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="BraTS Statistics Inspector",
    )
    
    # Set layout
    app.layout = create_layout(datasets)
    
    # Register callbacks
    register_callbacks(app, repo)
    
    return app
