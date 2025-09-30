"""Launch the BraTS statistics inspector web application.

This unified tool replaces the separate generate_case_statistics.py and
summarize_brats_dataset.py scripts with an interactive web interface that
supports both per-case and dataset-wide statistics analysis.

Features:
- Interactive browser-based interface
- Case statistics: histograms and label volumes for individual cases
- Dataset statistics: aggregate analysis across entire datasets
- Caching: pre-compute dataset statistics and save to outputs/
- Recompute option: refresh statistics when data changes

Examples:
    # Launch with default datasets
    python scripts/analyze_statistics.py --open-browser

    # Launch with custom dataset roots
    python scripts/analyze_statistics.py --data-root training_data --data-root validation_data

    # Launch on custom port
    python scripts/analyze_statistics.py --port 8060 --host 0.0.0.0

    # Launch in debug mode
    python scripts/analyze_statistics.py --debug --open-browser
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from threading import Timer
from typing import Iterable


def bootstrap_pythonpath() -> None:
    """Ensure the repository root is on the Python path."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


bootstrap_pythonpath()

from apps.statistics_inspector import create_dash_app


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch the interactive BraTS statistics inspector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        help="Path to a BraTS dataset root (repeat to add multiple roots). "
             "Defaults to training_data, training_data_additional, validation_data.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind. Default is 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to serve the Dash app on. Default is 8050.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode with hot reload.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open the app in your default browser.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Main entry point for the statistics inspector."""
    args = parse_args(argv)
    data_roots = args.data_root if args.data_root else None

    try:
        app = create_dash_app(data_roots=data_roots)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        sys.exit(2)

    url = f"http://{args.host}:{args.port}"
    print(f"[statistics-inspector] Starting application at {url}")
    print(f"[statistics-inspector] Navigate to the 'Case Statistics' tab for per-case analysis")
    print(f"[statistics-inspector] Navigate to the 'Dataset Statistics' tab for cohort-wide summaries")
    print(f"[statistics-inspector] Cached statistics are stored in outputs/statistics_cache/")
    
    if args.open_browser:
        Timer(1.5, lambda: webbrowser.open_new(url)).start()

    app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
