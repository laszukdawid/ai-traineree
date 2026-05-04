import contextlib
import http.server
import socketserver
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return


@contextlib.contextmanager
def static_server(root: Path, port: int):
    handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(root), **kwargs)
    socketserver.TCPServer.allow_reuse_address = True
    try:
        server = socketserver.TCPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        if exc.errno == 98:
            raise RuntimeError(f"Port {port} is already in use. Pick a different port with --port.") from exc
        raise
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def wait_for_http(url: str, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1) as response:
                return response.status, response.read(8192)
        except URLError:
            time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {url}")
