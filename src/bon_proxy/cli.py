"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from bon_proxy.app import create_app
from bon_proxy.config import ConfigLoadError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bon-proxy")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
    except ConfigLoadError as exc:
        raise SystemExit(str(exc)) from exc

    logging.basicConfig(
        level=getattr(logging, config.server.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
        workers=1,
    )
