#!/usr/bin/env python3
"""Incrementally build chunk embeddings cache for fixed chunker (reusing old cache, encoding ONLY new chunks)."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed

PUBLIC_DIR = ROOT / "output/ngram-comparison/splits"
V3_ORIG_DIR = Path("/tmp/innopolis-dataset-v3-review/datasets/own_v3")
V3_CLEAN_DIR = ROOT / "datasets/own_v3_clean"
OFFICIAL_PQ = ROOT / "datasets/public/public-test-dataset.parquet"
OLD_CACHE_DIR = ROOT / "output/qwen3-category-aware/cache"
OUT_CACHE_DIR = ROOT / "output/cat-aware-v2/cache"


def main():
    OUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = OUT_CACHE_DIR / "chunks_embeddings.npz"
    out_meta = OUT_CACHE_DIR / "chunk_encoder_meta.json"
    ckpt_npz = OUT_CACHE_DIR / "delta_checkpoint.npz"

    # 1. Reconstruct exact old unique_chunks ordering to map old cache rows -> text
    splits_orig = cae.collect_all_splits(PUBLIC_DIR, V3_ORIG_DIR, OFFICIAL_PQ)
    old_uniq_chunks = cae._build_unique_chunk_lookup(splits_orig)
    old_npz = np.load(OLD_CACHE_DIR / "chunks_embeddings.npz")
    old_emb = old_npz["chunk_emb"]  # float16 (54694, 1024)
    query_emb = old_npz["query_emb"]  # float32 (18, 1024)
    assert len(old_uniq_chunks) == len(old_emb), (len(old_uniq_chunks), len(old_emb))
    old_chunk_to_emb = {ch: old_emb[i] for i, ch in enumerate(old_uniq_chunks)}

    # 2. Collect all splits using fixed chunker (using own_v3_clean as active v3 dataset, plus keep v3_orig_train mapping if needed)
    active_splits = {
        "pub_train": cae.load_public_rows(PUBLIC_DIR / "train.jsonl"),
        "pub_calibration": cae.load_public_rows(PUBLIC_DIR / "calibration.jsonl"),
        "pub_validation": cae.load_public_rows(PUBLIC_DIR / "validation.jsonl"),
        "v3_train": cae.load_v3_rows_full(V3_CLEAN_DIR / "train.jsonl"),
        "v3_validation": cae.load_v3_rows_full(V3_CLEAN_DIR / "validation.jsonl"),
        "v3_test": cae.load_v3_rows_full(V3_CLEAN_DIR / "test.jsonl"),
    }

    uniq_ads = []
    ad_to_uid = {}
    split_ad_uids = {}
    for sname, rows in active_splits.items():
        uids = []
        for r in rows:
            txt = r["description"]
            if txt not in ad_to_uid:
                ad_to_uid[txt] = len(uniq_ads)
                uniq_ads.append(txt)
            uids.append(ad_to_uid[txt])
        split_ad_uids[sname] = np.asarray(uids, dtype=np.int32)

    uniq_chunks = []
    chunk_to_cid = {}
    ad_chunk_offsets = [0]
    ad_chunk_indices = []
    ad_chunk_counts = []
    for txt in uniq_ads:
        cs = chunk_ad_fixed(txt)
        ad_chunk_counts.append(len(cs))
        for ch in cs:
            if ch not in chunk_to_cid:
                chunk_to_cid[ch] = len(uniq_chunks)
                uniq_chunks.append(ch)
            ad_chunk_indices.append(chunk_to_cid[ch])
        ad_chunk_offsets.append(len(ad_chunk_indices))

    delta_chunks = [ch for ch in uniq_chunks if ch not in old_chunk_to_emb]
    # Sort delta chunks by length ascending to minimize transformer padding overhead on CPU
    delta_sorted = sorted(delta_chunks, key=len)
    print(
        f"Active unique_ads={len(uniq_ads)} | unique_chunks={len(uniq_chunks)} | "
        f"reused={len(uniq_chunks) - len(delta_chunks)} | delta_to_encode={len(delta_chunks)}",
        flush=True,
    )

    delta_map = {}
    start_idx = 0
    if ckpt_npz.exists():
        ck = np.load(ckpt_npz, allow_pickle=False)
        done_n = int(ck["done_n"])
        done_embs = ck["embs"]
        if done_n <= len(delta_sorted):
            for idx in range(done_n):
                delta_map[delta_sorted[idx]] = done_embs[idx]
            start_idx = done_n
            print(f"Resumed delta checkpoint at {start_idx}/{len(delta_sorted)}", flush=True)

    encode_s = 0.0
    if start_idx < len(delta_sorted):
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))
        t0 = time.perf_counter()
        model = SentenceTransformer(
            cae.MODEL_NAME,
            device="cpu",
            model_kwargs={"dtype": torch.float32},
            processor_kwargs={"padding_side": "left"},
        )
        model.max_seq_length = 1024
        print(f"Loaded encoder on CPU in {time.perf_counter() - t0:.2f}s", flush=True)

        done_embs_list = [delta_map[delta_sorted[k]] for k in range(start_idx)]
        block_size = 512
        et0 = time.perf_counter()
        for pos in range(start_idx, len(delta_sorted), block_size):
            batch_texts = delta_sorted[pos : pos + block_size]
            bt0 = time.perf_counter()
            embs = model.encode(
                batch_texts,
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float16)
            for t, vec in zip(batch_texts, embs):
                delta_map[t] = vec
            done_embs_list.append(embs)
            done_now = min(pos + len(batch_texts), len(delta_sorted))
            all_done = np.vstack(done_embs_list) if len(done_embs_list) > 1 or done_now > 0 else np.zeros((0, 1024), dtype=np.float16)
            np.savez(ckpt_npz, done_n=np.int32(done_now), embs=all_done)
            dt_b = time.perf_counter() - bt0
            elapsed = time.perf_counter() - et0
            rem = (len(delta_sorted) - done_now) / max((done_now - start_idx) / elapsed, 1e-6)
            print(
                f"[{done_now}/{len(delta_sorted)}] block={len(batch_texts)} in {dt_b:.1f}s "
                f"({len(batch_texts)/dt_b:.1f} ch/s) | elapsed={elapsed:.1f}s | eta={rem/60:.1f}m",
                flush=True,
            )
        encode_s = time.perf_counter() - et0

    # Assemble final aligned chunk_emb matrix in uniq_chunks order
    final_emb = np.empty((len(uniq_chunks), 1024), dtype=np.float16)
    for cid, ch in enumerate(uniq_chunks):
        final_emb[cid] = old_chunk_to_emb[ch] if ch in old_chunk_to_emb else delta_map[ch]

    arrays = {
        "chunk_emb": final_emb,
        "query_emb": query_emb,
        "ad_chunk_offsets": np.asarray(ad_chunk_offsets, dtype=np.int32),
        "ad_chunk_indices": np.asarray(ad_chunk_indices, dtype=np.int32),
    }
    for sname, arr in split_ad_uids.items():
        arrays[f"uid_{sname}"] = arr
    np.savez_compressed(out_npz, **arrays)

    norms = np.linalg.norm(final_emb.astype(np.float32), axis=1)
    counts_sorted = sorted(ad_chunk_counts)
    meta = {
        "model_name": cae.MODEL_NAME,
        "prompt": cae.QUERY_PROMPT,
        "unique_ads": len(uniq_ads),
        "unique_chunks": len(uniq_chunks),
        "reused_from_old_cache": len(uniq_chunks) - len(delta_chunks),
        "newly_encoded_delta_chunks": len(delta_chunks),
        "delta_encode_seconds_cpu": encode_s,
        "chunks_per_ad_stats": {
            "min": counts_sorted[0],
            "median": counts_sorted[len(counts_sorted) // 2],
            "p90": counts_sorted[int(len(counts_sorted) * 0.9)],
            "max": counts_sorted[-1],
        },
        "chunk_norm_stats": {"min": float(norms.min()), "max": float(norms.max())},
        "v3_dataset": str(V3_CLEAN_DIR),
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved fixed chunk cache to {out_npz} (norm range {norms.min():.4f}..{norms.max():.4f})", flush=True)


if __name__ == "__main__":
    main()
