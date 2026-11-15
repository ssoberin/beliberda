"""Inference-only probes: original frozen C_syntax, existing chunker/retrieval/features.

Core embeddings are reused. Only unseen chunks are encoded locally, offline, on CPU.
These extra probes are NOT added to train/test and are not used for model selection.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='4'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier,Pool

sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts import eval_public_tests_syntax as eps
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.syntax_features import SyntaxAnalyzer,extract_pair_features,SYNTAX_CAT_FEATURES

OUT=Path('output/syntax-edge-case-audit')
ART=Path('/home/user/hackaton/innopolis_2026')
SNAPSHOT=Path('/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,obj):(OUT/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
    assert not (OUT/'probe_predictions.json').exists(),'Do not repeat probes after inspecting results'
    source=Path('datasets/own_v3_clean/test.jsonl')
    contexts=Path('datasets/context_pairs_100/test_20.jsonl')
    model_file=ART/'output/syntax-aware-experiment/models/catboost_C_syntax.cbm'
    heads_file=ART/'output/syntax-aware-experiment/models/sgd_full_heads.joblib'
    hashes={str(p):sha(p) for p in [source,contexts,model_file,heads_file,Path('avito_compare/scripts/syntax_features.py')]}
    raw=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    rows=[dict(id='partial_ellipsis',description='Плитку на пол не укладываем, только на стены.',mcId=101,expert_expected=[103],role='new_manual_probe_not_official_gold'),
          dict(id='partial_explicit',description='Мы не укладываем напольную плитку, но укладываем плитку на стены.',mcId=101,expert_expected=[103],role='new_manual_probe_not_official_gold'),
          dict(id='floor_refusal_only',description='Плитку на пол не укладываем.',mcId=101,expert_expected=[],role='new_manual_probe_not_official_gold'),
          dict(id='different_categories_one_sentence',description='Мы укладываем плитку, но не устанавливаем раковины.',mcId=101,expert_expected=[103],role='new_manual_probe_not_official_gold'),
          dict(id='different_categories_separate_sentences',description='Мы укладываем плитку. Раковины не устанавливаем.',mcId=101,expert_expected=[103],role='new_manual_probe_not_official_gold')]
    for r in [json.loads(s) for s in contexts.read_text().splitlines() if s.strip()]:
        rows.append(dict(id=r['item_id'],description=r['description'],mcId=r['mcId'],official_dataset_gold=sorted(e['mcId'] for e in r['found_microcategories'] if e['mcId'] in cae.SCOPE_102_106),role='existing_separate_context_test20',topic=r['topic']))
    for i in [96,42]:
        r=raw[i];rows.append(dict(id=f'known_v3_{i}',description=r['description'],mcId=r['mcId'],official_dataset_gold=sorted(e['mcId'] for e in r['found_microcategories'] if e['mcId'] in cae.SCOPE_102_106),role='existing_main_test_reproduction',row_idx=i))
    write('probe_inputs_frozen.json',dict(rows=rows,sha256=hashes,encoder_snapshot=str(SNAPSHOT),threshold=.5,no_training=True))
    eps.PUBLIC_DIR=ART/'output/ngram-comparison/splits'
    eps.V3_CLEAN_DIR=Path('datasets/own_v3_clean').resolve()
    eps.V2_CACHE_NPZ=ART/'output/cat-aware-v2/cache/chunks_embeddings.npz'
    old_chunks,old_emb,query=eps.reconstruct_v2_chunk_map()
    lookup={text:i for i,text in enumerate(old_chunks)}
    ad_chunks=[chunk_ad_fixed(r['description']) for r in rows]
    chunks=list(dict.fromkeys(c for parts in ad_chunks for c in parts))
    unseen=[c for c in chunks if c not in lookup]
    print('Frozen probe inputs',len(rows),'chunks',len(chunks),'new',len(unseen),flush=True)
    encoded={};encoding_seconds=0.
    if unseen:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        kw='processor_kwargs' if 'processor_kwargs' in inspect.signature(SentenceTransformer.__init__).parameters else 'tokenizer_kwargs'
        start=time.perf_counter()
        encoder=SentenceTransformer(str(SNAPSHOT),device='cpu',local_files_only=True,model_kwargs={'dtype':torch.float32},**{kw:{'padding_side':'left'}})
        encoder.max_seq_length=8192
        values=encoder.encode(unseen,batch_size=4,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float16).astype(np.float32)
        encoding_seconds=time.perf_counter()-start
        encoded=dict(zip(unseen,values));del encoder;gc.collect()
        np.savez_compressed(OUT/'new_chunk_embeddings.npz',chunk_emb=values)
        write('new_chunk_texts.json',unseen)
        print('Offline CPU embedding seconds',round(encoding_seconds,3),flush=True)
    embs=np.vstack([old_emb[lookup[c]].astype(np.float32) if c in lookup else encoded[c] for c in chunks])
    del old_emb;gc.collect()
    local={c:i for i,c in enumerate(chunks)};indices=[];offsets=[0]
    for parts in ad_chunks:indices.extend(local[c] for c in parts);offsets.append(len(indices))
    raw_sims=embs@query.T
    sims=np.column_stack([raw_sims[:,j*3:(j+1)*3].max(1) for j in range(6)])
    feats,_=cae.build_category_features_for_uids(np.arange(len(rows)),np.asarray(offsets),np.asarray(indices),embs,sims)
    heads=joblib.load(heads_file)['heads']
    p_sgd=np.column_stack([heads[c].predict_proba(feats[c])[:,1] for c in cae.SCOPE_102_106])
    analyzer=SyntaxAnalyzer();records=[]
    for i,r in enumerate(rows):
        start,end=offsets[i:i+2];ad_indices=np.asarray(indices[start:end]);ad_sims=sims[ad_indices]
        for j,c in enumerate(cae.SCOPE_102_106):
            col=cae.ALL_IDS.index(c);positions=np.argsort(-ad_sims[:,col],kind='stable')[:min(3,len(ad_indices))].tolist()
            f=extract_pair_features(analyzer,ad_chunks[i],positions,[float(ad_sims[p,col]) for p in positions],c,len(r['description']))
            f.update(p_sgd=float(p_sgd[i,j]),probe_id=r['id'],row_idx=i)
            records.append(f)
    df=pd.DataFrame(records);model=CatBoostClassifier();model.load_model(str(model_file));cols=model.feature_names_
    for c in SYNTAX_CAT_FEATURES:df[c]=df[c].astype(str)
    probabilities=model.predict_proba(Pool(df[cols],cat_features=SYNTAX_CAT_FEATURES))[:,1].reshape(len(rows),5)
    result=[]
    for i,r in enumerate(rows):
        selected=[c for j,c in enumerate(cae.SCOPE_102_106) if c!=r['mcId'] and probabilities[i,j]>=.5]
        result.append({**r,'prediction':selected,'probabilities':{str(c):float(probabilities[i,j]) for j,c in enumerate(cae.SCOPE_102_106)},'p_sgd':{str(c):float(p_sgd[i,j]) for j,c in enumerate(cae.SCOPE_102_106)}})
    saved=pd.read_csv(ART/'output/catboost-optimization/final_predictions.csv');saved=saved[(saved.model=='C_syntax_baseline__fixed')&(saved.dataset=='v3_test')]
    reproduction={}
    for r in result:
        if r['role']!='existing_main_test_reproduction':continue
        old=saved[saved.row_idx==r['row_idx']]
        diff=max(abs(r['probabilities'][str(int(x.cat_id))]-float(x.probability)) for x in old.itertuples())
        old_prediction=sorted(old[old.prediction==1].cat_id.astype(int).tolist())
        assert old_prediction==r['prediction']
        assert diff<1e-5,(r['id'],diff)
        reproduction[r['id']]=dict(max_abs_probability_difference=diff,prediction_identical=True)
    assert all(sha(path)==h for path,h in hashes.items())
    write('probe_predictions.json',dict(rows=result,reproduction=reproduction,encoding_seconds=encoding_seconds,threshold=.5,sha256=hashes,no_training=True,encoder_dtype='CPU float32, chunk vectors stored float16 as original cache'))
    df.to_csv(OUT/'probe_features.csv',index=False)
    for r in result:
        if r['role']!='existing_separate_context_test20' or r['topic']=='partial_refusal':print(r['id'],'prediction',r['prediction'],'expected',r.get('official_dataset_gold',r.get('expert_expected')),'score103',round(r['probabilities']['103'],6),flush=True)
    print('Reproduction',reproduction,flush=True)

if __name__=='__main__':main()
