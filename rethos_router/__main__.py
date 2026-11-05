"""python -m rethos_router route | build-index | serve"""
import argparse
from contextlib import redirect_stdout
import json
import logging
from pathlib import Path
import sys

from .common import RouterError, load_config
from .router import Router, load_index
from .server import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Router JSON configuration; paths are relative to the project root')
    parser.add_argument('--verbose', action='store_true', help='Diagnostic messages go to stderr')
    commands = parser.add_subparsers(dest='command', required=True)
    route = commands.add_parser('route', help='Read text and write exactly one three-field JSON result')
    source = route.add_mutually_exclusive_group()
    source.add_argument('--request', help='User request text; omit to read stdin')
    source.add_argument('--request-file', type=Path, help='UTF-8 request text file')
    build = commands.add_parser('build-index', help='Build/reuse the local SkillRet vector index; no LLM call')
    build.add_argument('--rebuild', action='store_true')
    serve = commands.add_parser('serve', help='Serve POST /route using a persistent loaded router')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8787)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stdin, 'reconfigure'):
        sys.stdin.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s', stream=sys.stderr)
    try:
        # Transformers and third-party loading progress must never pollute route stdout.
        with redirect_stdout(sys.stderr):
            if args.command == 'build-index':
                index = load_index(load_config(args.config), rebuild=args.rebuild)
                result = {'indexed_skills': len(index.skills)}
            elif args.command == 'route':
                config = load_config(args.config)
                if args.request_file:
                    request = args.request_file.read_text(encoding='utf-8')
                elif args.request is not None:
                    request = args.request
                else:
                    request = sys.stdin.read(config['max_request_chars'] + 1)
                # Validate text before model/index initialization.
                if not request.strip() or len(request) > config['max_request_chars']:
                    raise RouterError('Request must be nonempty and within max_request_chars')
                result = Router.from_config(args.config).route(request)
            else:
                server = make_server(Router.from_config(args.config), args.host, args.port)
                logging.info('Listening on http://%s:%s/route', *server.server_address)
                try:
                    server.serve_forever()
                finally:
                    server.server_close()
                return 0
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (RouterError, OSError, ImportError, RuntimeError) as exc:
        print(f'Router error: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
