import json,time,numpy as np,torch
from sentence_transformers import SentenceTransformer
assert 'T4' in torch.cuda.get_device_name(0)
texts=json.load(open('/content/hn200_delta.json'));print('New chunks',len(texts),flush=True)
model=SentenceTransformer('Qwen/Qwen3-Embedding-0.6B',device='cuda',model_kwargs={'dtype':torch.float16},processor_kwargs={'padding_side':'left'})
model.max_seq_length=1024
start=time.monotonic();emb=model.encode(texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype(np.float16)
np.savez('/content/hn200_delta.npz',done_n=np.int32(len(texts)),embs=emb)
print('Encoded',emb.shape,'seconds',time.monotonic()-start,'GPU',torch.cuda.get_device_name(0),flush=True)
