"""Existing supported CPU execution of frozen Qwen; encode only new chunks."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import hashlib,inspect,json,time
from pathlib import Path
import numpy as np,torch
from sentence_transformers import SentenceTransformer
P=Path('output/tail-services-v3-experiment')
snapshot=Path('/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')
texts=json.loads((P/'cache/delta_chunks.json').read_text());assert snapshot.exists()
assert not (P/'cache/delta_checkpoint.npz').exists(),'Refuse overwrite of frozen new embedding cache'
torch.set_num_threads(4);torch.set_num_interop_threads(1)
kw='processor_kwargs' if 'processor_kwargs' in inspect.signature(SentenceTransformer.__init__).parameters else 'tokenizer_kwargs'
start=time.perf_counter();model=SentenceTransformer(str(snapshot),device='cpu',local_files_only=True,model_kwargs={'dtype':torch.float32},**{kw:{'padding_side':'left'}})
model.max_seq_length=1024
print('Encoding only missing chunks:',len(texts),flush=True)
emb=model.encode(texts,batch_size=4,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype(np.float16)
assert emb.shape==(len(texts),1024) and np.isfinite(emb).all()
np.savez(P/'cache/delta_checkpoint.npz',done_n=np.int32(len(texts)),embs=emb)
diag={'device':'cpu','execution_reason':'Colab CLI cannot open its read-only default log even after escalation; no Colab VM created','model_snapshot':str(snapshot),'weights_dtype':'float32','stored_embedding_dtype':'float16','max_seq_length':1024,'batch_size':4,'chunk_count':len(texts),'seconds':time.perf_counter()-start,'chunk_texts_sha256':hashlib.sha256((P/'cache/delta_chunks.json').read_bytes()).hexdigest(),'embedding_sha256':hashlib.sha256((P/'cache/delta_checkpoint.npz').read_bytes()).hexdigest()}
(P/'encoding.json').write_text(json.dumps(diag,indent=2)+'\n');print(json.dumps(diag),flush=True)
