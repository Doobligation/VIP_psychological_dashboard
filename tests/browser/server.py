"""Browser checks use an isolated temporary database, never study recordings."""
from pathlib import Path
import sys
import tempfile
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve_dashboard import Handler, ThreadingHTTPServer
from serving.server import ClosedLoopService

with tempfile.TemporaryDirectory(prefix='polar-browser-') as directory:
    Handler.feedback_service = ClosedLoopService(Path(directory) / 'test.sqlite3')
    try:
        with ThreadingHTTPServer(('127.0.0.1', int(os.environ.get('POLAR_TEST_PORT', '8000'))), Handler) as server:
            server.serve_forever()
    finally:
        Handler.feedback_service.close()
