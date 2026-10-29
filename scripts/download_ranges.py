"""Resumable HTTP byte-range downloader for proxies buffering large responses."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import urllib.request

def download(url, path, size=None, workers=6):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    if size is None:
        with urllib.request.urlopen(urllib.request.Request(url + '?probe=' + str(time.time_ns()), headers={'Range': 'bytes=0-0'}), timeout=60) as response:
            size = int(response.headers['Content-Range'].split('/')[-1])
    checkpoint = path.with_suffix(path.suffix + '.ranges.json')
    done = set(json.loads(checkpoint.read_text())) if checkpoint.exists() else set()
    if not path.exists():
        with path.open('wb') as f:
            f.truncate(size)
        done = set()
    def fetch(start):
        end = min(size, start + chunk_size) - 1
        for attempt in range(6):
            try:
                sep = '&' if '?' in url else '?'
                req = urllib.request.Request(url + sep + 'chunk=' + str(start) + '&fresh=' + str(time.time_ns()), headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=90) as r:
                    assert r.status == 206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                    data = r.read()
                assert len(data) == end - start + 1
                with path.open('r+b') as f:
                    f.seek(start); f.write(data)
                return start
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
    starts = [s for s in range(0, size, chunk_size) if s not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, s) for s in starts]):
            start = future.result()
            done.add(start)
            checkpoint.write_text(json.dumps(sorted(done)))
            if len(done) % 16 == 0:
                print(f'{path.name}: {min(len(done)*chunk_size,size)/size:.0%}', flush=True)
    return path

if __name__ == '__main__':
    import sys
    download(sys.argv[1], sys.argv[2], workers=int(sys.argv[3]) if len(sys.argv) > 3 else 6)
