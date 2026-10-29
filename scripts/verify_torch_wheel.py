"""Validate the CUDA wheel against the SHA256 on PyTorch's official wheel index."""
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
wheel = ROOT / '.cache/wheels/torch-2.7.1+cu128-cp312-cp312-win_amd64.whl'
with urllib.request.urlopen('https://download.pytorch.org/whl/cu128/torch/', timeout=60) as response:
    index = response.read().decode()
links = [html.unescape(s) for s in re.findall(r'href="([^"]+)"', index)]
link = next(s for s in links if 'torch-2.7.1%2Bcu128-cp312-cp312-win_amd64.whl#sha256=' in s)
expected = link.split('#sha256=')[1]
source = {'url': link, 'sha256': expected}
if wheel.exists() and not wheel.with_suffix(wheel.suffix + '.ranges.json').exists():
    raise RuntimeError('Missing range checkpoint')
if wheel.exists():
    h = hashlib.sha256()
    with wheel.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    source['actual_sha256'] = h.hexdigest()
    source['verified'] = h.hexdigest() == expected
else:
    source['verified'] = False
(ROOT / 'config/torch-wheel-source.json').write_text(json.dumps(source, indent=2), encoding='utf-8')
print(json.dumps(source))
if wheel.exists():
    assert source['verified'], 'Wheel is incomplete or corrupt'
