"""Freeze input/config and prepare only missing Qwen chunks, no test scoring."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.eval_public_tests_syntax import reconstruct_v2_chunk_map
from avito_compare.scripts.qwen3_embedding_experiment import load_official_parquet
from avito_compare.scripts import run_syntax_experiment as se
P=Path('output/tail-services-v3-experiment');(P/'cache').mkdir(exist_ok=True,parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
basechunks,baseemb,queries=reconstruct_v2_chunk_map();lookup={t:baseemb[i] for i,t in enumerate(basechunks)}
splits={f'pub_{s}':cae.load_public_rows(Path('output/ngram-comparison/splits')/f'{s}.jsonl') for s in ['train','calibration','validation']}
splits.update({f'v3_{s}':cae.load_v3_rows_full(Path('datasets/own_v3_clean')/f'{s}.jsonl') for s in ['train','validation','test']})
splits['tail_train']=cae.load_v3_rows_full(Path('datasets/v3_tail_services_60/train_addition.jsonl'))
splits['tail_holdout20']=cae.load_v3_rows_full(Path('datasets/v3_tail_services_60/test_holdout.jsonl'))
# Only official208. public_synthetic_300 is never loaded/scored.
_,_,official=load_official_parquet(Path('datasets/public/public-test-dataset.parquet'))
splits['official_208']=official
needed={ch for rows in splits.values() for r in rows for ch in chunk_ad_fixed(r['description'])}
# Public-test delta is only an existing text->embedding cache. Select needed chunks;
# no synthetic300 rows/labels/features/evaluation enter this experiment.
delta_texts=json.loads(Path('output/syntax-aware-experiment/cache/public_tests_delta_chunks.json').read_text())
delta=np.load('output/syntax-aware-experiment/cache/public_tests_delta_embs.npz')
print('delta keys',delta.files,flush=True)
arr=delta['embs'] if 'embs' in delta else delta['chunk_emb']
for i,t in enumerate(delta_texts):
 if t in needed and t not in lookup:lookup[t]=arr[i]
new=sorted(needed-lookup.keys());(P/'cache/delta_chunks.json').write_text(json.dumps(new,ensure_ascii=False))
# Save only required embeddings and rows; no existing cache is overwritten.
import pickle
with (P/'cache/prepared.pkl').open('wb') as f:pickle.dump({'splits':splits,'lookup':lookup,'queries':queries},f)
config=json.loads(Path('output/syntax-aware-experiment/config.json').read_text())
config.update(experiment='tail-services-v3',baseline_train=6319,augmented_train=6379,threshold=.5,new_rows=60,encoder='Qwen/Qwen3-Embedding-0.6B',encoder_frozen=True,chunker='existing fixed_chunker.chunk_ad_fixed',retrieval_top_k=3,final_evaluation_once=True,calibration_use='CatBoost early stopping only',forbidden_dataset='public_synthetic_300',oof='Recompute all OOF with existing nested StratifiedGroupKFold protocol; never reuse old OOF values',dataset_sha256=sha(Path('datasets/v3_tail_services_60/train_addition.jsonl')),baseline_model_sha256=sha(Path('output/syntax-aware-experiment/models/catboost_C_syntax.cbm')),baseline_heads_sha256=sha(Path('output/syntax-aware-experiment/models/sgd_full_heads.joblib')),encoder_missing_chunks=len(new))
config['new_holdout_sha256']=sha(Path('datasets/v3_tail_services_60/test_holdout.jsonl'))
config['encoder_execution']={'device':'cpu','weights_dtype':'float32','cached_embeddings_dtype':'float16','max_seq_length':1024,'reason':'Colab CLI logging path read-only even after escalation; existing supported CPU encoder fallback'}
config['source_sha256']={str(p):sha(p) for p in [Path('datasets/own_v3_clean/train.jsonl'),Path('datasets/own_v3_clean/validation.jsonl'),Path('datasets/own_v3_clean/test.jsonl'),*[Path('output/ngram-comparison/splits')/f'{s}.jsonl' for s in ['train','calibration','validation']],Path('avito_compare/scripts/fixed_chunker.py'),Path('avito_compare/scripts/syntax_features.py'),Path('avito_compare/scripts/run_syntax_experiment.py')]}
(P/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
print('Prepared',len(new),'missing chunks; frozen config. Existing caches unchanged.',flush=True)
