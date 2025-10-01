"""UI layout for the statistics inspector Dash application."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from .data import LABEL_MAP


def create_layout(datasets: list[str]) -> html.Div:
    """Create the main application layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("BraTS Statistics Inspector", className="text-center mb-4"),
                html.P("Analyze per-case or dataset-wide statistics for BraTS cohorts", 
                       className="text-center text-muted"),
            ], width=12),
        ]),
        
        # Mode selection tabs
        dbc.Tabs([
            dbc.Tab(label="Case Statistics", tab_id="case", children=[
                html.Div(id="case-stats-content", children=[
                    create_case_statistics_controls(datasets),
                    html.Hr(),
                    html.Div(id="case-stats-output"),
                ])
            ]),
            dbc.Tab(label="Dataset Statistics", tab_id="dataset", children=[
                html.Div(id="dataset-stats-content", children=[
                    create_dataset_statistics_controls(datasets),
                    html.Hr(),
                    html.Div(id="dataset-stats-output"),
                ])
            ]),
        ], id="stats-mode-tabs", active_tab="case", className="mb-4"),
        
    ], fluid=True, className="p-4")


def create_case_statistics_controls(datasets: list[str]) -> html.Div:
    """Create controls for case statistics mode."""
    return dbc.Row([
        dbc.Col([
            html.Label("Dataset:", className="fw-bold"),
            dcc.Dropdown(
                id="case-dataset-dropdown",
                options=[{"label": ds, "value": ds} for ds in datasets],
                value=datasets[0] if datasets else None,
                clearable=False,
            ),
        ], width=4),
        dbc.Col([
            html.Label("Case:", className="fw-bold"),
            dcc.Dropdown(
                id="case-dropdown",
                options=[],
                value=None,
                clearable=False,
            ),
        ], width=4),
        dbc.Col([
            html.Label("Histogram Bins:", className="fw-bold"),
            dcc.Input(
                id="histogram-bins-input",
                type="number",
                value=100,
                min=10,
                max=500,
                step=10,
                className="form-control",
            ),
        ], width=2),
        dbc.Col([
            html.Label("\u00A0", className="fw-bold"),  # nbsp for alignment
            dbc.Button("Compute", id="compute-case-btn", color="primary", className="w-100"),
        ], width=2),
    ], className="g-3 mb-3")


def create_dataset_statistics_controls(datasets: list[str]) -> html.Div:
    """Create controls for dataset statistics mode."""
    return dbc.Row([
        dbc.Col([
            html.Label("Dataset:", className="fw-bold"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[{"label": ds, "value": ds} for ds in datasets],
                value=datasets[0] if datasets else None,
                clearable=False,
            ),
        ], width=4),
        dbc.Col([
            html.Label("Max Cases (optional):", className="fw-bold"),
            dcc.Input(
                id="max-cases-input",
                type="number",
                value=None,
                min=1,
                step=1,
                placeholder="All cases",
                className="form-control",
            ),
        ], width=3),
        dbc.Col([
            html.Label("\u00A0", className="fw-bold"),
            dbc.Button("Load Cached", id="load-cached-btn", color="info", className="w-100 me-2"),
        ], width=2),
        dbc.Col([
            html.Label("\u00A0", className="fw-bold"),
            dbc.Button("Recompute", id="recompute-btn", color="warning", className="w-100"),
        ], width=2),
        dbc.Col([
            html.Div(id="cache-status", className="mt-4 text-muted small"),
        ], width=12),
    ], className="g-3 mb-3")


def create_case_statistics_display(stats: dict) -> html.Div:
    """Create display components for case statistics."""
    if not stats:
        return html.Div("No statistics to display.", className="text-muted")

    components = [
        html.H3(f"Case: {stats['case_id']}", className="mb-3"),
        html.H5(f"Dataset: {stats['dataset']}", className="text-muted mb-4"),
    ]

    # Histogram figures
    components.append(html.H4("Modality Histograms", className="mt-4 mb-3"))
    histogram_data = stats.get("histogram_data", {})
    
    if histogram_data:
        figures = []
        for modality in sorted(histogram_data.keys()):
            hist_info = histogram_data[modality]
            counts = hist_info["counts"]
            bin_edges = hist_info["bin_edges"]
            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
            
            fig = {
                "data": [{
                    "x": bin_centers,
                    "y": counts,
                    "type": "bar",
                    "name": modality.upper(),
                    "marker": {"color": "steelblue"},
                }],
                "layout": {
                    "title": f"{modality.upper()} Intensity Distribution",
                    "xaxis": {"title": "Normalized Intensity"},
                    "yaxis": {"title": "Voxel Count"},
                    "hovermode": "closest",
                }
            }
            figures.append(dbc.Col([dcc.Graph(figure=fig)], width=6))
        
        components.append(dbc.Row(figures, className="g-3"))
    
    # Label statistics table
    components.append(html.H4("Segmentation Label Statistics", className="mt-5 mb-3"))
    label_stats = stats.get("label_stats", {})
    
    if label_stats:
        table_rows = []
        for label_id, label_name in LABEL_MAP.items():
            voxels = label_stats.get(f"voxels_label_{label_id}", 0)
            pct = label_stats.get(f"pct_label_{label_id}", 0.0)
            table_rows.append(html.Tr([
                html.Td(str(label_id)),
                html.Td(label_name),
                html.Td(f"{voxels:,}"),
                html.Td(f"{pct:.2f}%"),
            ]))
        
        table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Label"),
                html.Th("Name"),
                html.Th("Voxels"),
                html.Th("Percentage"),
            ])),
            html.Tbody(table_rows),
        ], bordered=True, hover=True, responsive=True, striped=True)
        
        components.append(table)
    
    return html.Div(components)


def create_dataset_statistics_display(stats: dict) -> html.Div:
    """Create display components for dataset statistics."""
    if not stats:
        return html.Div("No statistics to display.", className="text-muted")

    aggregate = stats.get("aggregate", {})
    num_cases = stats.get("num_cases_processed", 0)
    
    components = [
        html.H3(f"Dataset: {stats.get('dataset_name', 'Unknown')}", className="mb-3"),
        html.H5(f"Cases Processed: {num_cases}", className="text-muted mb-4"),
    ]

    # Shape and spacing statistics
    components.append(html.H4("Volume Geometry", className="mt-4 mb-3"))
    
    shape_counts = aggregate.get("shape_counts", {})
    spacing_counts = aggregate.get("voxel_spacing_counts", {})
    orientation_counts = aggregate.get("orientation_counts", {})
    
    geometry_cards = []
    
    if shape_counts:
        shape_list = html.Ul([html.Li(f"{shape}: {count} cases") for shape, count in shape_counts.items()])
        geometry_cards.append(dbc.Col([
            dbc.Card([
                dbc.CardHeader("Volume Shapes"),
                dbc.CardBody(shape_list),
            ])
        ], width=4))
    
    if spacing_counts:
        spacing_list = html.Ul([html.Li(f"{spacing}: {count} cases") for spacing, count in spacing_counts.items()])
        geometry_cards.append(dbc.Col([
            dbc.Card([
                dbc.CardHeader("Voxel Spacing"),
                dbc.CardBody(spacing_list),
            ])
        ], width=4))
    
    if orientation_counts:
        orient_list = html.Ul([html.Li(f"{orient}: {count} cases") for orient, count in orientation_counts.items()])
        geometry_cards.append(dbc.Col([
            dbc.Card([
                dbc.CardHeader("Orientations"),
                dbc.CardBody(orient_list),
            ])
        ], width=4))
    
    if geometry_cards:
        components.append(dbc.Row(geometry_cards, className="g-3 mb-4"))

    # Modality presence
    components.append(html.H4("Modality Availability", className="mt-4 mb-3"))
    modality_presence = aggregate.get("modality_presence", {})
    
    if modality_presence:
        modality_rows = []
        for modality, count in sorted(modality_presence.items()):
            pct = (count / num_cases * 100) if num_cases > 0 else 0
            modality_rows.append(html.Tr([
                html.Td(modality.upper()),
                html.Td(f"{count} / {num_cases}"),
                html.Td(f"{pct:.1f}%"),
            ]))
        
        modality_table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Modality"),
                html.Th("Present"),
                html.Th("Percentage"),
            ])),
            html.Tbody(modality_rows),
        ], bordered=True, hover=True, responsive=True, striped=True, className="mb-4")
        
        components.append(modality_table)

    # Intensity statistics
    components.append(html.H4("Intensity Statistics (Aggregate)", className="mt-4 mb-3"))
    intensity_summary = aggregate.get("intensity_summary", {})
    
    if intensity_summary:
        intensity_rows = []
        for modality in sorted(intensity_summary.keys()):
            stats_dict = intensity_summary[modality]
            for stat_name in ["p01", "p50", "p99", "mean"]:
                if stat_name in stats_dict:
                    stat_values = stats_dict[stat_name]
                    intensity_rows.append(html.Tr([
                        html.Td(modality.upper()),
                        html.Td(stat_name),
                        html.Td(f"{stat_values.get('mean', 0):.2f}"),
                        html.Td(f"{stat_values.get('min', 0):.2f}"),
                        html.Td(f"{stat_values.get('max', 0):.2f}"),
                    ]))
        
        intensity_table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Modality"),
                html.Th("Statistic"),
                html.Th("Mean"),
                html.Th("Min"),
                html.Th("Max"),
            ])),
            html.Tbody(intensity_rows),
        ], bordered=True, hover=True, responsive=True, striped=True, className="mb-4")
        
        components.append(intensity_table)

    # Segmentation label counts
    components.append(html.H4("Segmentation Label Totals", className="mt-4 mb-3"))
    label_counts_raw = aggregate.get("label_counts", {})
    label_counts = {}
    for key, value in label_counts_raw.items():
        try:
            label_key = int(key)
        except (TypeError, ValueError):
            continue
        label_counts[label_key] = int(value)
    
    if label_counts:
        label_rows = []
        total_voxels = sum(label_counts.values())
        for label_id, label_name in LABEL_MAP.items():
            count = label_counts.get(label_id, 0)
            pct = (count / total_voxels * 100) if total_voxels > 0 else 0
            label_rows.append(html.Tr([
                html.Td(str(label_id)),
                html.Td(label_name),
                html.Td(f"{count:,}"),
                html.Td(f"{pct:.2f}%"),
            ]))
        
        label_table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Label"),
                html.Th("Name"),
                html.Th("Total Voxels"),
                html.Th("Percentage"),
            ])),
            html.Tbody(label_rows),
        ], bordered=True, hover=True, responsive=True, striped=True)
        
        components.append(label_table)

    return html.Div(components)
