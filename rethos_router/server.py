"""Small serial HTTP interface; the process reuses one loaded embedding model."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging

from .common import RequestError, ReviewError, RouterError, strict_json

LOG = logging.getLogger(__name__)


def make_server(router, host='127.0.0.1', port=8787):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Do not log the request body or credentials.
            LOG.debug('HTTP request completed')

        def respond(self, status, body, content_type):
            raw = body.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            if self.path != '/route':
                self.respond(404, 'Not found', 'text/plain; charset=utf-8')
                return
            self.connection.settimeout(30)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if self.headers.get('Transfer-Encoding') or not 0 < length <= 1_048_576:
                    raise RequestError('A bounded Content-Length is required')
                content_type = self.headers.get('Content-Type', '').split(';', 1)[0].lower()
                if content_type != 'application/json':
                    raise RequestError('Content-Type must be application/json')
                payload = strict_json(self.rfile.read(length).decode('utf-8'))
                if not isinstance(payload, dict) or set(payload) != {'request'}:
                    raise RequestError('Body must contain only the request field')
                result = router.route(payload['request'])
                self.respond(200, json.dumps(result, ensure_ascii=False, allow_nan=False), 'application/json; charset=utf-8')
            except (RequestError, ValueError, UnicodeError):
                self.respond(400, 'Invalid routing request', 'text/plain; charset=utf-8')
            except ReviewError:
                self.respond(502, 'Skill review failed; no retry was made', 'text/plain; charset=utf-8')
            except (RouterError, OSError, RuntimeError):
                self.respond(500, 'Routing failed', 'text/plain; charset=utf-8')

    return HTTPServer((host, port), Handler)
