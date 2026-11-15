"""Text-only duplicate audit; heldout gold is not read or used."""
import collections,hashlib,json,re
from pathlib import Path
import duckdb,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
P=Path('output/hard-negatives-v3-experiment');D=Path('datasets/v3_hard_negatives_200')
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def norm(s):return re.sub(r'\W+',' ',s.casefold()).strip()
r=read(D/'train_addition.jsonl');a=json.loads((D/'annotation_review.json').read_text())
paths={**{f'pub_{s}':Path('output/ngram-comparison/splits')/f'{s}.jsonl' for s in ['train','calibration','validation']},**{f'v3_{s}':Path('datasets/own_v3_clean')/f'{s}.jsonl' for s in ['train','validation','test']}}
previous={k:[x['description'] for x in read(p)] for k,p in paths.items()}
previous['official_208']=[x[0] for x in duckdb.connect().execute("select description from read_parquet('datasets/public/public-test-dataset.parquet')").fetchall()]
texts=[x['description'] for x in r];assert len(r)==200 and len(set(map(norm,texts)))==200
for name,old in previous.items():assert not set(map(norm,texts))&set(map(norm,old)),name
vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=1,dtype=np.float32)
X=vec.fit_transform(texts+[t for v in previous.values() for t in v]);sim=cosine_similarity(X[:200],X[:200]);np.fill_diagonal(sim,0)
pairs=[{'a':r[i]['item_id'],'b':r[j]['item_id'],'cosine':float(sim[i,j])} for i,j in zip(*np.where(np.triu(sim,k=1)>=.65))]
checks={};start=200
for name,old in previous.items():
 z=cosine_similarity(X[:200],X[start:start+len(old)]);inds=z.argmax(1);mx=z.max(1)
 checks[name]={'exact_overlap':0,'max_similarity':float(mx.max()),'pairs_ge_0_65':[{'new':r[i]['item_id'],'old_row':int(inds[i]),'cosine':float(mx[i])} for i in range(200) if mx[i]>=.65]};start+=len(old)
para=collections.Counter(p for t in texts for p in t.split('\n\n'));most=para.most_common(5)
res={'rows':200,'topic_counts':dict(collections.Counter(x['topic'] for x in a)),'length_counts':dict(collections.Counter(str(x['length_bin']) for x in a)),'within_max_cosine':float(sim.max()),'within_pairs_ge_0_65':pairs,'cross':checks,'repeated_support_paragraphs_top5':most,'repeated_support_warning':'Supporting transaction paragraphs are shared. Full descriptions and service briefs are unique; this is compositional authorship, not 200 independent hand-written full narratives.','train_sha256':hashlib.sha256((D/'train_addition.jsonl').read_bytes()).hexdigest()}
(P/'dataset_audit.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in res.items() if k not in ['within_pairs_ge_0_65','repeated_support_paragraphs_top5']},ensure_ascii=False,indent=2));print('candidate pairs',len(pairs),'top support repetition',most[0][1])
