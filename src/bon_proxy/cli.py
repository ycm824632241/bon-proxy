"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from bon_proxy.app import create_app
from bon_proxy.config import ConfigLoadError, load_config
from bon_proxy.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Best-of-N vLLM proxy")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Override server.log_file; process logs are still printed to stderr",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
    except ConfigLoadError as exc:
        raise SystemExit(str(exc)) from exc

    log_file = args.log_file or config.server.log_file
    configure_logging(config.server.log_level, log_file)
    if args.log_file:
        logging.getLogger(__name__).info("log_file override=%s", args.log_file)
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
        workers=1,
    )


if __name__ == "__main__":
    main()
