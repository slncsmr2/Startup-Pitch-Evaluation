from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os

PORT = int(os.environ.get("PORT", "7860"))
ROOT = Path(__file__).resolve().parent

os.chdir(ROOT)

class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), NoCacheHandler)
    print(f"Serving {ROOT} on http://0.0.0.0:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
