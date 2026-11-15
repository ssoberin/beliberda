"""Inference for frozen C_syntax using the unchanged research pipeline functions."""
import hashlib
import inspect
import json
import os
import logging
import time
from pathlib import Path
from threading import RLock
import numpy as np

LOG = logging.getLogger('backend.model')

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / 'models/c_syntax'
IDS = (102, 103, 104, 105, 106)
EXPECTED_FILES = {
    'catboost.cbm': '3b7dbea4b3fe6ba6c61b2499dd82911e904282217a7281d8cc1c22b19b399ca9',
    'sgd_heads.joblib': '49d95c79e2633d4ba2fc95593e04e8acc4ce0845549a7dcf7348f73af007c45f',
    'queries.npz': '988e0d8fe92a3d10299cadb4c0bd58cf1946f1fb10e597c02c6babc1587482b7',
}

class ClassifierUnavailable(RuntimeError):
    pass

def _sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

def encoder_snapshot(directory, model_id, revision):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    local_cache = str(Path(directory) / 'encoder_cache')
    for cache in (local_cache, None):
        try:
            path = snapshot_download(model_id, revision=revision, cache_dir=cache, local_files_only=True)
            LOG.info('encoder_cache_hit model=%s revision=%s cache=%s', model_id, revision, cache or 'huggingface-default')
            return path
        except LocalEntryNotFoundError:
            LOG.info('encoder_cache_miss model=%s cache=%s', model_id, cache or 'huggingface-default')
    LOG.info('encoder_download_start model=%s revision=%s cache=%s', model_id, revision, local_cache)
    path = snapshot_download(model_id, revision=revision, cache_dir=local_cache)
    LOG.info('encoder_download_complete model=%s revision=%s', model_id, revision)
    return path

class CSyntaxClassifier:
    def __init__(self, directory):
        directory = Path(directory)
        started = time.monotonic()
        LOG.info('model_load_start model=C_syntax directory=%s', directory)
        manifest_path = directory / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError('Отсутствует manifest.json модели.')
        self.manifest = json.loads(manifest_path.read_text())
        expected_files = EXPECTED_FILES if directory.resolve() == DEFAULT_MODEL_DIR.resolve() else self.manifest.get('files', {})
        if set(expected_files) != {'catboost.cbm', 'sgd_heads.joblib', 'queries.npz'}:
            raise ValueError('Некорректный список файлов в manifest.json модели.')
        for name, digest in expected_files.items():
            path = directory / name
            if not path.is_file() or _sha(path) != digest:
                LOG.error('model_hash_failed file=%s', name)
                raise ValueError(f'Хеш или файл {self.manifest.get("name", "C_syntax")} не соответствует зафиксированной модели: {name}')
            LOG.info('model_hash_verified file=%s sha256=%s', name, digest)
        if tuple(self.manifest['ids']) != IDS or self.manifest['thresholds'] != [.5] * 5:
            raise ValueError('Неверные категории или пороги C_syntax.')
        for name, digest in self.manifest['source_sha256'].items():
            if _sha(ROOT / name) != digest:
                raise ValueError(f'Код признаков отличается от обученной C_syntax: {name}')
        import joblib
        from catboost import CatBoostClassifier
        self.heads = joblib.load(directory / 'sgd_heads.joblib')['heads']
        if set(self.heads) != set(IDS):
            raise ValueError('Неверные SGD-головы C_syntax.')
        self.queries = np.load(directory / 'queries.npz', allow_pickle=False)['query_emb'].astype(np.float32)
        if self.queries.shape != (18, 1024) or not np.isfinite(self.queries).all():
            raise ValueError('Неверные retrieval-векторы C_syntax.')
        self.catboost = CatBoostClassifier()
        self.catboost.load_model(str(directory / 'catboost.cbm'))
        self.directory = directory
        self._encoder = self._analyzer = None
        self._lock = RLock()
        LOG.info('model_load_complete model=C_syntax heads=%s features=%d elapsed_ms=%d', IDS, len(self.catboost.feature_names_), int((time.monotonic()-started)*1000))

    def _encode(self, chunks):
        if self._encoder is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                torch.set_num_threads(min(4, os.cpu_count() or 1))
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model_id, revision = self.manifest['encoder'], self.manifest['encoder_revision']
                started = time.monotonic()
                path = encoder_snapshot(self.directory, model_id, revision)
                LOG.info('encoder_load_start model=%s device=%s path=%s', model_id, device, path)
                kw = 'processor_kwargs' if 'processor_kwargs' in inspect.signature(SentenceTransformer.__init__).parameters else 'tokenizer_kwargs'
                self._encoder = SentenceTransformer(path, device=device, local_files_only=True,
                    model_kwargs={'dtype': torch.float16 if device == 'cuda' else torch.float32},
                    **{kw: {'padding_side': 'left'}})
                self._encoder.max_seq_length = self.manifest['max_seq_length']
                LOG.info('encoder_load_complete model=%s device=%s max_seq_length=%d elapsed_ms=%d', model_id, device, self._encoder.max_seq_length, int((time.monotonic()-started)*1000))
            except Exception as exc:
                LOG.error('encoder_load_failed error_type=%s', type(exc).__name__)
                raise ClassifierUnavailable('Не удалось загрузить Qwen3-Embedding-0.6B для C_syntax.') from exc
        return np.asarray(self._encoder.encode(chunks, batch_size=4, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False), dtype=np.float16).astype(np.float32)

    def predict_proba(self, descriptions):
        from .scripts.fixed_chunker import chunk_ad_fixed
        from .scripts.qwen3_category_aware_experiment import ALL_IDS, build_category_features_for_uids
        from .scripts.syntax_features import SyntaxAnalyzer, extract_pair_features, SYNTAX_CAT_FEATURES
        import pandas as pd
        from catboost import Pool
        results = []
        with self._lock:
            for description in descriptions:
                started = time.monotonic()
                proc_desc = description
                if self.manifest.get('preprocess') == 'h12_normalize_and_strip_prompt':
                    from .scripts.text_preprocess import normalize_ad_text, strip_prompt_injections
                    proc_desc = strip_prompt_injections(normalize_ad_text(description))
                chunks = chunk_ad_fixed(proc_desc) or ['']
                LOG.info('model_prediction_start description_chars=%d chunks=%d', len(description), len(chunks))
                unique = list(dict.fromkeys(chunks))
                vectors = self._encode(unique)
                if vectors.shape != (len(unique), 1024) or not np.isfinite(vectors).all():
                    raise ClassifierUnavailable('Неверный ответ Qwen embedding encoder.')
                positions = {text: i for i, text in enumerate(unique)}
                indices = np.array([positions[text] for text in chunks])
                raw = vectors @ self.queries.T
                sims = np.column_stack([raw[:, j*3:(j+1)*3].max(1) for j in range(6)])
                LOG.info('model_embeddings_complete unique_chunks=%d dimensions=%d', len(unique), vectors.shape[1])
                features, _ = build_category_features_for_uids(np.array([0]),
                    np.array([0, len(chunks)]), indices, vectors, sims)
                if self._analyzer is None:
                    self._analyzer = SyntaxAnalyzer()
                records = []
                try:
                    for cid in IDS:
                        scores = sims[indices, ALL_IDS.index(cid)]
                        selected = np.argsort(-scores, kind='stable')[:3].tolist()
                        row = extract_pair_features(self._analyzer, chunks, selected,
                            [float(scores[p]) for p in selected], cid, len(proc_desc))
                        row['p_sgd'] = float(self.heads[cid].predict_proba(features[cid])[0, 1])
                        records.append(row)
                finally:
                    self._analyzer._sent_cache.clear()
                frame = pd.DataFrame(records)
                for col in SYNTAX_CAT_FEATURES:
                    frame[col] = frame[col].astype(str)
                results.append(self.catboost.predict_proba(Pool(
                    frame[self.catboost.feature_names_], cat_features=SYNTAX_CAT_FEATURES), thread_count=2)[:, 1])
                LOG.info('model_prediction_complete probabilities=%s elapsed_ms=%d',
                         dict(zip(IDS, map(float, results[-1]))), int((time.monotonic()-started)*1000))
        return np.asarray(results)

    def warmup(self):
        self.predict_proba(['Ремонт помещений.'])

def load_bundle(path=None):
    directory = Path(path) if path else DEFAULT_MODEL_DIR
    if not directory.exists():
        if path:
            return None
        raise FileNotFoundError(f'Не найден комплект C_syntax: {directory}')
    if not directory.is_dir():
        raise ValueError('Хеш/формат: ожидается доверенный каталог C_syntax, а не TF-IDF joblib.')
    return {'pipeline': CSyntaxClassifier(directory), 'ids': IDS, 'thresholds': [.5] * 5}
