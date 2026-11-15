import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers>=3.0.0", "transformers>=4.51.0"])

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

assert torch.cuda.is_available(), "CUDA T4 GPU not available!"
print("Device:", torch.cuda.get_device_name(0), flush=True)

chunks = json.loads(Path("/content/all_h12_delta_chunks.json").read_text(encoding="utf-8"))
print(f"Loaded {len(chunks)} chunks to encode", flush=True)

t0 = time.perf_counter()
kw = "processor_kwargs" if "processor_kwargs" in inspect.signature(SentenceTransformer.__init__).parameters else "tokenizer_kwargs"
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    device="cuda",
    model_kwargs={"dtype": torch.float16},
    **{kw: {"padding_side": "left"}},
)
model.max_seq_length = 1024
print(f"Loaded Qwen3-Embedding-0.6B in {time.perf_counter()-t0:.2f}s", flush=True)

t1 = time.perf_counter()
vecs = model.encode(
    chunks,
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)
vecs_f16 = np.asarray(vecs, dtype=np.float16)
assert vecs_f16.shape == (len(chunks), 1024) and np.isfinite(vecs_f16).all()
np.savez_compressed("/content/all_h12_delta_embs.npz", chunks=np.array(chunks), embs=vecs_f16)
print(f"Encoded {len(chunks)} chunks -> {vecs_f16.shape} in {time.perf_counter()-t1:.2f}s", flush=True)
