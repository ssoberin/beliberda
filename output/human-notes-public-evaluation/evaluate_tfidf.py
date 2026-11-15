"""Same human37, separately evaluate the frozen pure TF-IDF+SGD baseline."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,runpy
from pathlib import Path
import joblib,numpy as np
P=Path('output/human-notes-public-evaluation')
def main():
 assert not (P/'tfidf_only_metrics.json').exists(),'Preserve the previously frozen comparison'
 rows=json.loads((P/'parsed_annotations.json').read_text());resolved=[r for r in rows if r['annotation_status']=='human_resolved']
 path=Path('output/syntax-aware-experiment/models/tfidf_baseline.joblib');model=joblib.load(path)
 probs=model.predict_proba([r['canonical_description'] for r in rows]);ix=np.array([r['item_number']-1 for r in resolved]);gold=np.array([[int(c in r['human_gold']) for c in range(102,107)] for r in resolved])
 helper=runpy.run_path(str(P/'evaluate.py'),run_name='helpers')['metrics'];score=helper(gold,(probs[ix]>=.5).astype(np.int8))
 result={'model':'tfidf_sgd_only','training':'Same public3398 + V3clean2921; no extra60','thresholds':[.5]*5,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'human_resolved37':score}
 (P/'tfidf_only_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (P/'tfidf_only_predictions.json').write_text(json.dumps([{'item_number':r['item_number'],'human_gold':r['human_gold'],'probabilities':probs[i].tolist(),'prediction':[c for j,c in enumerate(range(102,107)) if probs[i,j]>=.5]} for i,r in enumerate(rows)],ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
