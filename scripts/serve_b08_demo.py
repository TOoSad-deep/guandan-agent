#!/usr/bin/env python3
"""Read-only B08 display fixtures; no launch configurations, keys or model calls."""
import argparse
from pathlib import Path

from guandan_agent.app import create_web_application


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    app = create_web_application(recording_root=root / "artifacts/b08/demo-recordings",
                                 allowed_config_names=(), archive_root=root / "tests/fixtures/web",
                                 allow_report_generation=False, port=args.port)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, reload=False, access_log=False)


if __name__ == "__main__":
    main()
