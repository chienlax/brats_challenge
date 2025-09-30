"""Callback functions for the statistics inspector."""
from __future__ import annotations

from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

from .data import StatisticsRepository
from .layout import create_case_statistics_display, create_dataset_statistics_display


def register_callbacks(app, repo: StatisticsRepository):
    """Register all Dash callbacks for the statistics inspector."""

    @app.callback(
        Output("case-dropdown", "options"),
        Output("case-dropdown", "value"),
        Input("case-dataset-dropdown", "value"),
    )
    def update_case_dropdown(dataset_name):
        """Update case dropdown when dataset changes."""
        if not dataset_name:
            return [], None
        
        cases = repo.get_cases_for_dataset(dataset_name)
        options = [{"label": case, "value": case} for case in cases]
        value = cases[0] if cases else None
        return options, value

    @app.callback(
        Output("case-stats-output", "children"),
        Input("compute-case-btn", "n_clicks"),
        State("case-dataset-dropdown", "value"),
        State("case-dropdown", "value"),
        State("histogram-bins-input", "value"),
        prevent_initial_call=True,
    )
    def compute_case_statistics(n_clicks, dataset_name, case_id, bins):
        """Compute and display case statistics."""
        if not dataset_name or not case_id:
            raise PreventUpdate
        
        try:
            stats = repo.compute_case_statistics(
                dataset_name=dataset_name,
                case_id=case_id,
                bins=bins or 100,
            )
            return create_case_statistics_display(stats)
        except Exception as e:
            return f"Error computing statistics: {str(e)}"

    @app.callback(
        Output("cache-status", "children"),
        Input("dataset-dropdown", "value"),
    )
    def update_cache_status(dataset_name):
        """Update cache status message."""
        if not dataset_name:
            return ""
        
        if repo.has_cached_dataset_stats(dataset_name):
            cache_path = repo.get_dataset_cache_path(dataset_name)
            return f"✓ Cached statistics available at {cache_path.name}"
        else:
            return "⚠ No cached statistics found. Click 'Recompute' to generate."

    @app.callback(
        Output("dataset-stats-output", "children"),
        Input("load-cached-btn", "n_clicks"),
        Input("recompute-btn", "n_clicks"),
        State("dataset-dropdown", "value"),
        State("max-cases-input", "value"),
        prevent_initial_call=True,
    )
    def handle_dataset_statistics(load_clicks, recompute_clicks, dataset_name, max_cases):
        """Handle dataset statistics loading or recomputing."""
        if not dataset_name:
            raise PreventUpdate
        
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        try:
            if button_id == "load-cached-btn":
                # Try to load from cache
                if repo.has_cached_dataset_stats(dataset_name):
                    stats = repo.load_cached_dataset_stats(dataset_name)
                    return create_dataset_statistics_display(stats)
                else:
                    return "No cached statistics found. Click 'Recompute' to generate."
            
            elif button_id == "recompute-btn":
                # Recompute statistics
                stats = repo.compute_dataset_statistics(
                    dataset_name=dataset_name,
                    max_cases=max_cases if max_cases and max_cases > 0 else None,
                )
                return create_dataset_statistics_display(stats)
        
        except Exception as e:
            return f"Error: {str(e)}"
        
        raise PreventUpdate
