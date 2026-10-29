"""Measure local SDPA backends without loading or changing retrieval weights."""
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.set_num_threads(4)
q = torch.randn(1, 16, 4096, 128, device='cuda', dtype=torch.bfloat16)
for backend in [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]:
    for masked in [False, True]:
        try:
            mask = torch.ones(1, 1, 4096, 4096, device='cuda', dtype=torch.bool).tril() if masked else None
            with torch.inference_mode(), sdpa_kernel(backend):
                torch.cuda.synchronize(); t = time.perf_counter()
                for i in range(3):
                    out = torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=mask, is_causal=not masked)
                torch.cuda.synchronize()
            print(backend, 'masked=', masked, 'ms=', (time.perf_counter()-t)*1000/3, flush=True)
        except Exception as exc:
            print(backend, 'masked=', masked, type(exc).__name__, str(exc)[:200], flush=True)
