"""Count feature filtering with fitted vectorizers; no new model fitting."""
import collections,json
from pathlib import Path
import joblib
P=Path('output/v3-revision3-study')
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
old=read(P/'before/splits/train.jsonl');after=read(P/'after/splits/train.jsonl')
new=[r for r in after if str(r['item_id']).startswith('own-v3-r3-')]
a=joblib.load(P/'after/model/tfidf_baseline.joblib');b=joblib.load(P/'before/model/tfidf_baseline.joblib')
vec=dict(a['pipeline'].named_steps['vectorizer'].transformer_list)['word']
oldvec=dict(b['pipeline'].named_steps['vectorizer'].transformer_list)['word']
analyze=vec.build_analyzer()
seen=set()
for r in old:seen.update(analyze(r['description'][:6000]))
features=set()
for r in new:features.update(analyze(r['description'][:6000]))
novel=features-seen;df=collections.Counter()
for r in after:df.update(set(analyze(r['description'][:6000]))&novel)
eligible={f for f in novel if df[f]>=2}
kept=eligible&vec.vocabulary_.keys();excluded=eligible-vec.vocabulary_.keys()
result={'definition':'Word features (unigrams/bigrams/trigrams) present in replacement250 but absent from entire BEFORE train. Eligibility means AFTER document frequency>=2. Count of kept features uses actual saved AFTER vocabulary. No fitting. Novel lexical features are not proven useful.',
'new_features_observed':len(novel),'excluded_by_min_df_2':len(novel-eligible),'eligible_new_features':len(eligible),'retained_new_features':len(kept),'eligible_but_excluded_by_max_features_18000':len(excluded),
'word_vocabulary_before':len(oldvec.vocabulary_),'word_vocabulary_after':len(vec.vocabulary_),'vocabulary_entries_replaced':len(set(vec.vocabulary_)-oldvec.vocabulary_.keys()),
'examples_excluded':sorted(excluded,key=lambda s:(-df[s],s))[:40],'examples_retained':sorted(kept,key=lambda s:(-df[s],s))[:40]}
(P/'feature_cap_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
