"""Optional developer preview. Serves this one artifact on loopback only."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class Preview(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/ai-add.html"):
            self.send_error(404)
            return
        body = Path(__file__).with_name("dist").joinpath("ai-add.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("AI ADD preview: http://127.0.0.1:8765 (loopback; one read-only HTML route)", flush=True)
    HTTPServer(("127.0.0.1", 8765), Preview).serve_forever()
