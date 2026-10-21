"""Loopback timing proxy for the REA capture — measurement only, not part of Argus.

Listens on 127.0.0.1:LISTEN and forwards every request verbatim to 127.0.0.1:UPSTREAM,
logging one JSON line per request: wall start, method, path(+query), status,
duration_ms (proxy receive -> last response byte sent), request/response byte counts and
content type. Header VALUES are never logged (only the response content-type).

    python3 -I timing_proxy.py 8025 8024 timing.jsonl
"""

import http.client
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN, UPSTREAM, LOG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
_lock = threading.Lock()
_seq = [0]
HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "upgrade", "te", "trailer"}


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):          # silence the default stderr access log
        pass

    def _forward(self):
        t0_wall, t0 = time.time(), time.perf_counter()
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else None
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "host"}
        up = http.client.HTTPConnection("127.0.0.1", UPSTREAM, timeout=1800)
        status, out, ctype, err = 502, b"", None, None
        try:
            up.request(self.command, self.path, body=body, headers=hdrs)
            r = up.getresponse()
            t_first = time.perf_counter()
            out = r.read()
            status, ctype = r.status, r.getheader("content-type")
            self.send_response(status)
            for k, v in r.getheaders():
                if k.lower() not in HOP and k.lower() != "content-length":
                    self.send_header(k, v)
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(out)
        except Exception as exc:        # upstream down / client gone — still log it
            err = type(exc).__name__
            t_first = time.perf_counter()
        finally:
            up.close()
        t1 = time.perf_counter()
        with _lock:
            _seq[0] += 1
            rec = {"seq": _seq[0], "start": round(t0_wall, 3), "method": self.command,
                   "path": self.path, "status": status,
                   "ttfb_ms": round((t_first - t0) * 1000, 1),
                   "duration_ms": round((t1 - t0) * 1000, 1),
                   "req_bytes": n, "resp_bytes": len(out), "content_type": ctype, "error": err}
            with open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")

    do_GET = do_POST = do_DELETE = do_PUT = do_HEAD = _forward


if __name__ == "__main__":
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("127.0.0.1", LISTEN), Proxy).serve_forever()
