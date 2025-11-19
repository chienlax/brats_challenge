"""
Interactive visualization of nnU-Net predictions vs ground truth.
Run this script to launch a Dash app for inspecting results.
"""

import sys
from pathlib import Path
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Any

# Ensure we can import from the apps package
sys.path.append(str(Path(__file__).parent.parent))

import dash
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.common.volume_utils import AXIS_TO_INDEX, MODALITY_LABELS, SEGMENTATION_LABELS, normalize_intensity, prepare_slice
from apps.volume_inspector.rendering import compose_rgba_slice

# Determine base path
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    if hasattr(sys, '_MEIPASS'):
        # One-file mode
        BASE_DIR = Path(sys._MEIPASS)
    else:
        # One-dir mode (PyInstaller 6+ puts dependencies in _internal)
        BASE_DIR = Path(sys.executable).parent / "_internal"
else:
    # Running as script
    BASE_DIR = Path(__file__).parent.parent

# Dataset configuration
PREDICTIONS_DIR = BASE_DIR / "outputs/nnunet/predictions/Dataset501_BraTSPostTx/3d_fullres_all"
RAW_DATA_DIR = BASE_DIR / "outputs/nnunet/nnUNet_raw/Dataset501_BraTSPostTx"
IMAGES_TS_DIR = RAW_DATA_DIR / "imagesTs"
LABELS_TS_DIR = RAW_DATA_DIR / "labelsTs"
SUMMARY_PATH = PREDICTIONS_DIR / "summary.json"

# Channel mapping from dataset.json
CHANNEL_MAP = {
    0: "t2f",
    1: "t1n",
    2: "t1c",
    3: "t2w",
}

CHANNEL_ORDER = ["t2f", "t1n", "t1c", "t2w"]
CHANNEL_TITLES = ["T2 FLAIR", "T1", "T1c", "T2"]

@dataclass
class CaseData:
    case_id: str
    prediction_path: Path
    image_paths: Dict[str, Path]
    label_path: Optional[Path]

class PredictionRepository:
    def __init__(self):
        self.cases: Dict[str, CaseData] = {}
        self.metrics: Dict[str, Any] = self._load_metrics()
        self._discover_cases()

    def _load_metrics(self) -> Dict[str, Any]:
        if not SUMMARY_PATH.exists():
            print(f"Warning: Metrics file not found: {SUMMARY_PATH}")
            return {}
        
        try:
            with open(SUMMARY_PATH, 'r') as f:
                data = json.load(f)
            
            # Index per-case metrics by case_id
            per_case = {}
            for entry in data.get("metric_per_case", []):
                # Extract case_id from file path (support both keys)
                file_path = entry.get("prediction_file") or entry.get("reference_file")
                if not file_path:
                    continue
                    
                path_obj = Path(file_path)
                case_id = path_obj.name.replace(".nii.gz", "")
                per_case[case_id] = entry
                
            return {
                "global": data.get("foreground_mean", {}),
                "per_case": per_case
            }
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return {}

    def _discover_cases(self):
        if not PREDICTIONS_DIR.exists():
            print(f"Warning: Predictions directory not found: {PREDICTIONS_DIR}")
            return

        # Find all prediction files
        pred_files = list(PREDICTIONS_DIR.glob("*.nii.gz"))
        
        for pred_path in pred_files:
            if pred_path.name == "summary.json": continue
            case_id = pred_path.name.replace(".nii.gz", "")
            
            # Find corresponding images
            image_paths = {}
            for channel_idx, modality in CHANNEL_MAP.items():
                img_name = f"{case_id}_{channel_idx:04d}.nii.gz"
                img_path = IMAGES_TS_DIR / img_name
                if img_path.exists():
                    image_paths[modality] = img_path
            
            # Find ground truth label
            label_path = LABELS_TS_DIR / f"{case_id}.nii.gz"
            if not label_path.exists():
                label_path = None

            if image_paths:
                self.cases[case_id] = CaseData(
                    case_id=case_id,
                    prediction_path=pred_path,
                    image_paths=image_paths,
                    label_path=label_path
                )
        
        print(f"Found {len(self.cases)} cases with predictions and images.")

    def get_case_ids(self) -> List[str]:
        return sorted(list(self.cases.keys()))

    def load_case(self, case_id: str) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Returns (images, ground_truth, prediction)"""
        case = self.cases.get(case_id)
        if not case:
            raise ValueError(f"Case {case_id} not found")

        images = {}
        for mod, path in case.image_paths.items():
            img = nib.load(str(path))
            data = img.get_fdata().astype(np.float32)
            images[mod] = normalize_intensity(data)

        ground_truth = None
        if case.label_path:
            gt_img = nib.load(str(case.label_path))
            ground_truth = gt_img.get_fdata().astype(np.uint8)

        pred_img = nib.load(str(case.prediction_path))
        prediction = pred_img.get_fdata().astype(np.uint8)

        return images, ground_truth, prediction

    def get_shape(self, case_id: str) -> Tuple[int, int, int]:
        case = self.cases.get(case_id)
        if not case:
            return (0, 0, 0)
        first_img = next(iter(case.image_paths.values()))
        img = nib.load(str(first_img))
        return img.shape

repo = PredictionRepository()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="BraTS Prediction Viewer", suppress_callback_exceptions=True)

def build_layout():
    case_ids = repo.get_case_ids()
    default_case = case_ids[0] if case_ids else None

    sidebar = html.Div([
        html.H2("BraTS Viewer", className="display-6"),
        html.Hr(),
        
        html.Label("Select Case:"),
        dcc.Dropdown(
            id="case-select",
            options=[{"label": cid, "value": cid} for cid in case_ids],
            value=default_case,
            clearable=False,
            className="mb-3"
        ),
        
        html.Label("Select Axis:"),
        dbc.RadioItems(
            id="axis-select",
            options=[
                {"label": "Axial", "value": "axial"},
                {"label": "Coronal", "value": "coronal"},
                {"label": "Sagittal", "value": "sagittal"},
            ],
            value="axial",
            className="mb-3"
        ),
        
        html.Label("Slice Index:"),
        dcc.Slider(
            id="slice-slider",
            min=0,
            max=100,
            step=1,
            value=50,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
            className="mb-3"
        ),
        
        dbc.Checklist(
            id="overlay-toggle",
            options=[{"label": "Show Overlay", "value": "show"}],
            value=["show"],
            switch=True,
            className="mb-3"
        ),

        html.Hr(),
        html.H4("Metrics", className="h5"),
        html.Div(id="metrics-content")

    ], style={
        "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "30rem",
        "padding": "2rem 1rem", "backgroundColor": "#f8f9fa", "overflowY": "auto"
    })

    content = html.Div([
        dcc.Graph(id="main-graph", style={"height": "95vh"})
    ], style={"marginLeft": "32rem", "padding": "1rem"})

    return html.Div([sidebar, content])

app.layout = build_layout

@app.callback(
    Output("slice-slider", "max"),
    Output("slice-slider", "value"),
    Input("case-select", "value"),
    Input("axis-select", "value"),
    State("slice-slider", "value")
)
def update_slider(case_id, axis, current_val):
    if not case_id:
        return 0, 0
    
    shape = repo.get_shape(case_id)
    axis_idx = AXIS_TO_INDEX[axis]
    max_slice = shape[axis_idx] - 1
    
    # Try to keep relative position or center
    new_val = current_val if current_val <= max_slice else max_slice // 2
    
    return max_slice, new_val

@app.callback(
    Output("metrics-content", "children"),
    Input("case-select", "value")
)
def update_metrics(case_id):
    if not case_id:
        return "No case selected."

    per_case = repo.metrics.get("per_case", {})
    case_data = per_case.get(case_id, {})
    
    if not case_data:
        return html.Div("No metrics available for this case.")

    # Define metrics to show
    metric_keys = ["Dice", "IoU", "TP", "FP", "FN", "TN", "n_pred", "n_ref"]
    
    # Define classes
    classes = ["1", "2", "3", "4"]
    class_names = {
        "1": "Enhancing",
        "2": "Non-Enhancing",
        "3": "Edema",
        "4": "Cavity"
    }

    # Get metrics dictionary
    metrics_data = case_data.get("metrics", {})

    # Build header
    header_row = [html.Th("Metric")]
    for c in classes:
        header_row.append(html.Th(f"{class_names[c]} ({c})"))
    
    thead = html.Thead(html.Tr(header_row))

    # Build rows
    rows = []
    for m in metric_keys:
        row_cells = [html.Td(m, style={"fontWeight": "bold"})]
        for c in classes:
            val = "-"
            if c in metrics_data:
                raw_val = metrics_data[c].get(m, 0)
                
                # Handle NaN
                if isinstance(raw_val, float) and math.isnan(raw_val):
                    val = "N/A"
                elif m in ["Dice", "IoU"]:
                    val = f"{raw_val:.4f}"
                else:
                    val = f"{raw_val:,.0f}"
            row_cells.append(html.Td(val))
        rows.append(html.Tr(row_cells))

    table = dbc.Table([
        thead,
        html.Tbody(rows)
    ], bordered=True, size="sm", style={"fontSize": "0.75rem", "textAlign": "center"})

    return table

@app.callback(
    Output("main-graph", "figure"),
    Input("case-select", "value"),
    Input("axis-select", "value"),
    Input("slice-slider", "value"),
    Input("overlay-toggle", "value")
)
def update_graph(case_id, axis, slice_idx, overlay_val):
    if not case_id:
        return go.Figure()

    images, ground_truth, prediction = repo.load_case(case_id)
    show_overlay = "show" in (overlay_val or [])

    # Create 2x4 subplot grid
    # Row 1: Ground Truth (4 channels)
    # Row 2: Prediction (4 channels)
    fig = make_subplots(
        rows=2, cols=4,
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.02, horizontal_spacing=0.01,
        subplot_titles=[f"{t} (GT)" for t in CHANNEL_TITLES] + [f"{t} (Pred)" for t in CHANNEL_TITLES]
    )

    for i, mod in enumerate(CHANNEL_ORDER):
        col = i + 1
        vol = images.get(mod)
        if vol is None:
            continue
            
        slice_2d = prepare_slice(vol, axis, slice_idx)
        
        # Row 1: Ground Truth
        gt_slice = None
        if ground_truth is not None and show_overlay:
            gt_slice = prepare_slice(ground_truth, axis, slice_idx)
        
        rgba_gt = compose_rgba_slice(slice_2d, gt_slice, show_overlay and gt_slice is not None)
        fig.add_trace(go.Image(z=rgba_gt), row=1, col=col)

        # Row 2: Prediction
        pred_slice = None
        if prediction is not None and show_overlay:
            pred_slice = prepare_slice(prediction, axis, slice_idx)
            
        rgba_pred = compose_rgba_slice(slice_2d, pred_slice, show_overlay and pred_slice is not None)
        fig.add_trace(go.Image(z=rgba_pred), row=2, col=col)

    # Add dummy traces for legend
    for label_val, (label_name, color_hex) in SEGMENTATION_LABELS.items():
        if label_val == 0: continue
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color_hex),
            name=f"{label_name} ({label_val})",
            showlegend=True
        ), row=1, col=1)

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")

    return fig

if __name__ == "__main__":
    print("Starting server...")
    # Disable debug mode when running as frozen app to avoid reloader issues
    is_frozen = getattr(sys, 'frozen', False)
    app.run_server(debug=not is_frozen, port=8051)
