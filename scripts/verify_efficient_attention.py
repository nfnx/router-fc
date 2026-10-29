"""Check the Windows SDPA optimization against default attention on fixed texts."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from evaluate_local_routers import ROOT, Reranker, configure_attention, sync

torch.set_num_threads(4)
rr = Reranker('Qwen/Qwen3-Reranker-0.6B', 'qwen', 4096)
query = 'Find a skill for systematically reviewing academic papers and evaluating evidence.'
docs = [('literature-review | Review academic evidence | ' + 'Search studies, compare evidence, assess methodology, and synthesize literature. ' * 600), ('image-editor | Edit image colors | ' + 'Resize photographs, change contrast, crop images, and export PNG files. ' * 600)]
sync(); t = time.perf_counter()
default = rr.score(query, docs)
sync(); default_s = time.perf_counter()-t
configure_attention()
sync(); t = time.perf_counter()
efficient = rr.score(query, docs)
sync(); efficient_s = time.perf_counter()-t
result = {'default_logits': default.tolist(), 'efficient_logits': efficient.tolist(), 'max_abs_difference': float(np.max(np.abs(default-efficient))), 'default_seconds': default_s, 'efficient_seconds': efficient_s, 'ordering_preserved': np.argsort(-default).tolist() == np.argsort(-efficient).tolist()}
assert result['ordering_preserved'] and result['max_abs_difference'] < .5
(ROOT / 'runs/efficient-attention-validation.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
