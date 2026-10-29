"""Download pinned model snapshots; validate every file against HF LFS/Git hashes."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
from download_ranges import download as download_ranges

ROOT = Path(__file__).resolve().parents[1]

def read_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)

def sha(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        if algorithm == 'sha1':
            h.update(f'blob {path.stat().st_size}\0'.encode())
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def download_file(job):
    repo, revision, entry, destination = job
    path = destination / entry['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get('lfs', {}).get('oid') or entry['oid']
    algorithm = 'sha256' if 'lfs' in entry else 'sha1'
    if not (path.exists() and path.stat().st_size == entry['size'] and sha(path, algorithm) == expected):
        for attempt in range(5):
            try:
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{entry['path']}?download=true&fresh={time.time_ns()}"
                temporary = path.with_suffix(path.suffix + '.part')
                if entry['size'] > 32 * 1024 * 1024:
                    download_ranges(url, temporary, entry['size'], workers=8)
                else:
                    with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                if temporary.stat().st_size != entry['size'] or sha(temporary, algorithm) != expected:
                    raise ValueError(f'Integrity check failed: {path}')
                temporary.replace(path)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    print(f'verified {repo}/{entry["path"]} ({entry["size"]:,} bytes)', flush=True)
    return {'path': entry['path'], 'size': entry['size'], 'sha256': sha(path), 'upstream_oid': expected}

def main():
    config = json.loads((ROOT / 'config/local_models.json').read_text())
    repos = list(dict.fromkeys(v[k] for v in config.values() for k in ('embedding', 'reranker')))
    manifest_path = ROOT / 'models/MANIFEST.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for repo in repos:
        meta = read_json(f'https://huggingface.co/api/models/{repo}')
        revision = manifest.get(repo, {}).get('revision', meta['sha'])
        entries = read_json(f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&limit=1000')
        selected = [e for e in entries if e['type'] == 'file' and (e['path'].endswith(('.json', '.safetensors', '.py', '.txt', '.model', '.md', '.jinja')) or e['path'] == 'merges.txt') and not e['path'].startswith(('onnx/', 'openvino/'))]
        directory = ROOT / 'models' / repo.replace('/', '--')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            files = list(pool.map(download_file, [(repo, revision, e, directory) for e in selected]))
        manifest[repo] = {'revision': revision, 'directory': str(directory.relative_to(ROOT)), 'license': meta.get('cardData', {}).get('license'), 'files': files}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'COMPLETE {repo} @ {revision}', flush=True)

if __name__ == '__main__':
    main()
