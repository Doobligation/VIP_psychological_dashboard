from __future__ import annotations

import json
import mimetypes
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from live_dashboard_backend import LiveDashboardBridge

ROOT = Path(__file__).resolve().parent
PORT = 8000
LIVE_STREAM_URL = 'http://127.0.0.1:7000/latest'

bridge = LiveDashboardBridge(ROOT / 'dashboard_data.json', live_url=LIVE_STREAM_URL, poll_seconds=5)
bridge.start()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/dashboard-data':
            return self._send_json(bridge.get_dashboard_data())

        if path == '/api/live-status':
            data = bridge.get_dashboard_data()
            return self._send_json({
                'liveStatus': data.get('liveStatus', {}),
                'liveLatestSample': data.get('liveLatestSample'),
            })

        if path == '/':
            self.path = '/psychosocial_dashboard.html'
            return super().do_GET()

        return super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def _send_json(self, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    url = f'http://127.0.0.1:{PORT}/psychosocial_dashboard.html'
    print(f'Serving integrated dashboard at {url}')
    print(f'Expecting live stream at {LIVE_STREAM_URL}')
    try:
        webbrowser.open(url)
    except Exception:
        pass
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
