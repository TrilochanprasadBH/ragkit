"""Stage 08: a minimal HTTP server over the service.

This module is **given**. It uses only the standard library so that the project runs
anywhere with no installation step.

In production you would put the same `Service.ask()` behind FastAPI, Flask, or whatever
your platform already runs. Nothing in `service.py` would change; only this file would.
The lesson shows the twelve-line FastAPI equivalent.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok", **service.manifest()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/ask":
                self._send(404, {"error": "not found"})
                return

            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "body must be JSON"})
                return

            question = str(payload.get("question", "")).strip()
            if not question:
                self._send(400, {"error": "question is required"})
                return

            self._send(200, service.ask(question))

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass  # silence the default per-request stderr line

    return Handler


def serve(service, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(service))
