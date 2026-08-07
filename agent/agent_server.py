"""
agent_server.py  --  a tiny local server for the grounded HPC diagnostician demo.

Serves ui/index.html and a POST /ask endpoint that runs the tool-calling agent over the frozen
run pool and returns the grounded answer, the tools it called (reasoning), the measurement facts
(evidence), and the performance-critic's verdict. The Session is built once at startup and reused.

Local only: binds 127.0.0.1. Auth to Azure OpenAI reuses the harness Entra token (run `az login`).
    python -m agent.agent_server        # then open http://127.0.0.1:8780/
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import diag_core as DC
import hpc_agent as HA

HOST, PORT = "127.0.0.1", 8780
_SESSION = None  # built lazily on first request


def _session():
    global _SESSION
    if _SESSION is None:
        _SESSION = DC.Session(verbose=True)
    return _SESSION


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "text" in ctype or "json" in ctype else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]  # ignore query string for routing
        if path in ("/", "/index.html"):
            path = os.path.join(_HERE, "ui", "index.html")
            if os.path.exists(path):
                self._send(200, open(path, "rb").read(), "text/html")
            else:
                self._send(404, "ui/index.html not found", "text/plain")
        elif self.path == "/programs":
            self._send(200, json.dumps({"programs": _session().list_programs()}))
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, "not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            q = json.loads(self.rfile.read(n) or b"{}").get("question", "").strip()
        except Exception as e:  # noqa: BLE001
            self._send(400, json.dumps({"error": "bad request: %s" % e}))
            return
        if not q:
            self._send(400, json.dumps({"error": "empty question"}))
            return
        try:
            out = HA.diagnose(q, sess=_session(), verify=True, trace=False)
            self._send(200, json.dumps(out, default=str))
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(e)}))


def main():
    _session()  # warm the pool before we accept requests
    print("[server] grounded HPC diagnostician on http://%s:%d/  (Ctrl-C to stop)" % (HOST, PORT))
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
