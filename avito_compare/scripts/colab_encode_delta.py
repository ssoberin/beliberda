#!/usr/bin/env python3
"""Encode delta_chunks.json on Colab T4 GPU matching original Qwen3-Embedding-0.6B float16 settings."""
import json
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

with open("delta_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} delta chunks. CUDA available: {torch.cuda.is_available()}", flush=True)
t0 = time.perf_counter()
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cuda",
    model_kwargs={"dtype": torch.float16},
    processor_kwargs={"padding_side": "left"},
)
model.max_seq_length = 1024
print(f"Model loaded in {time.perf_counter() - t0:.2f}s", flush=True)

t1 = time.perf_counter()
embs = model.encode(
    chunks,
    batch_size=64,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
).astype(np.float16)
dt = time.perf_counter() - t1
print(f"Encoded {embs.shape} in {dt:.2f}s ({len(chunks)/dt:.1f} ch/s)", flush=True)

np.savez("delta_checkpoint.npz", done_n=np.int32(len(chunks)), embs=embs)
print("Saved delta_checkpoint.npz", flush=True)
