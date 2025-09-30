"""Launch the BraTS interactive volume inspection Dash app."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from threading import Timer
from typing import Iterable


def bootstrap_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


bootstrap_pythonpath()

from apps.volume_inspector import create_dash_app


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the interactive BraTS volume inspector site.")
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        help="Path to a BraTS dataset root (repeat to add multiple roots). Defaults to training_data, training_data_additional, validation_data.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Default is 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8050, help="Port to serve the Dash app on.")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    parser.add_argument("--open-browser", action="store_true", help="Automatically open the app in your default browser.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    data_roots = args.data_root if args.data_root else None

    try:
        app = create_dash_app(data_roots=data_roots)
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        sys.exit(2)

    url = f"http://{args.host}:{args.port}"
    if args.open_browser:
        Timer(1.0, lambda: webbrowser.open_new(url)).start()

    app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
