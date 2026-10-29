"""Check pinned repository commits and byte-identical downloaded weight snapshots."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    sources = json.loads((ROOT / 'config/deployment_sources.json').read_text(encoding='utf-8-sig'))
    for source in sources:
        actual = subprocess.check_output(['git', '-C', str(ROOT / 'external' / source['name']), 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == source['commit'], source['name']
    models = json.loads((ROOT / 'models/MANIFEST.json').read_text())
    config = json.loads((ROOT / 'config/local_models.json').read_text())
    expected = {v[k] for v in config.values() for k in ('embedding', 'reranker')}
    assert set(models) == expected, f'Missing models: {expected - set(models)}'
    total = 0
    for repo, model in models.items():
        for entry in model['files']:
            path = ROOT / model['directory'] / entry['path']
            assert path.stat().st_size == entry['size'], str(path)
            h = hashlib.sha256()
            with path.open('rb') as f:
                while chunk := f.read(8 * 1024 * 1024):
                    h.update(chunk)
            assert h.hexdigest() == entry['sha256'], str(path)
            total += entry['size']
        print(f'PASS {repo} @ {model["revision"]}')
    print(f'PASS {len(sources)} source repositories, {len(models)} model snapshots, {total:,} bytes')

if __name__ == '__main__':
    main()
