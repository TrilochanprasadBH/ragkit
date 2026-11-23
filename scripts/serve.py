#!/usr/bin/env python3
"""Run the service.

    python scripts/serve.py                 # http://127.0.0.1:8000
    PORT=9000 python scripts/serve.py

Endpoints:

    GET  /health   liveness plus the index manifest
    POST /ask      {"question": "..."} -> answer, citations, timings, cost

Usage is appended to logs/usage.jsonl, one JSON object per request.
"""

from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / os.environ.get("IMPL", "src")))

from ragkit.server import serve  # noqa: E402

from scripts.bench import build_service


def main() -> int:
    port = int(os.environ.get("PORT", "8000"))
    service = build_service()
    service.usage_path = ROOT / "logs" / "usage.jsonl"

    manifest_path = ROOT / "reports" / "index.json"
    service.write_manifest(manifest_path)

    server = serve(service, port=port)
    print(f" listening on http://127.0.0.1:{port}")
    print(f" manifest  {manifest_path}")
    print(f" usage log {service.usage_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
