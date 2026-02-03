#!/usr/bin/env python3
"""Start the GameBench benchmark API server."""

import argparse
import logging
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="GameBench Benchmark API Server")
    parser.add_argument(
        "--config", default="configs/api_server.yaml",
        help="Path to server config YAML (default: configs/api_server.yaml)",
    )
    parser.add_argument("--host", default=None, help="Override host")
    parser.add_argument("--port", type=int, default=None, help="Override port")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    server_cfg = config.get("server", {})
    host = args.host or server_cfg.get("host", "0.0.0.0")
    port = args.port or server_cfg.get("port", 8000)

    # Import and initialize app
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install -e '.[api]'", file=sys.stderr)
        sys.exit(1)

    from davechess.benchmark.api.server import app
    from davechess.benchmark.api.dependencies import init_app

    init_app(app, config)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
