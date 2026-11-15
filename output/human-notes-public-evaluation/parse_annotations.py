"""Extract explicit provided human labels, never infer missing or choose alternatives."""
import hashlib,json,re,sys
from pathlib import Path
import duckdb
sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
P=Path('output/human-notes-public-evaluation')
SOURCE=Path('/home/user/Downloads/Telegram Desktop/public_test_139_unique_with_notes.md')
PROJECT_SOURCE=Path('datasets/public_review_139/public_test_139_unique_with_notes.md')
def norm(t):return ' '.join(t.casefold().split())
def main():
 source=SOURCE if SOURCE.exists() else PROJECT_SOURCE
 s=source.read_text();matches=list(re.finditer(r'^## Объявление (\d+) — строка (\d+)\s*$',s,re.M))
 raw=duckdb.connect().execute('select title,description,categories from read_parquet(?)',['datasets/public/public-test-dataset.parquet']).fetchall()
 rows=[]
 for i,m in enumerate(matches):
  block=s[m.start():matches[i+1].start() if i+1<len(matches) else len(s)]
  indices=list(map(int,re.search(r'\*\*Индексы в Public test:\*\* ([\d, ]+)',block).group(1).split(',')))
  title=re.search(r'\*\*Заголовок \(`title`\):\*\*\s*\n+````text\n(.*?)\n````',block,re.S).group(1)
  desc=re.search(r'\*\*Полное описание \(`description`\):\*\*\s*\n+````text\n(.*?)\n````',block,re.S).group(1)
  stripped=re.sub(r'````[^\n]*\n.*?\n````','',block,flags=re.S);hits=[];notes=[]
  for line in stripped.splitlines():
   hit=re.search(r'human_labels[`*_ \t]*:[`*_ \t]*(\[[^\]\r\n]*\])',line)
   if not hit:continue
   labels=json.loads(hit.group(1));assert set(labels)<=set(range(102,107));options=[labels]
   if re.search(r'\bили\b',line[hit.end():],re.I):options += [json.loads(a) for a in re.findall(r'\[[\d,\s]*\]',line[hit.end():])]
   hits+=options;notes.append(line)
  choices=sorted({tuple(sorted(set(a))) for a in hits})
  assert all(norm(desc)==norm(raw[j][1]) for j in indices),(int(m.group(1)),indices)
  rows.append({'item_number':int(m.group(1)),'official_rows':indices,'canonical_row':indices[0],'title':title,'description':desc,'canonical_description':raw[indices[0]][1],'source_mcId':101,'human_label_options':[list(x) for x in choices],'human_gold':list(choices[0]) if len(choices)==1 else None,'annotation_status':'human_resolved' if len(choices)==1 else 'human_ambiguous' if choices else 'no_human_labels','human_label_lines':notes,'notes_outside_input':stripped,'exact_description_matches_original':desc==raw[indices[0]][1],'same_chunker_output':chunk_ad_fixed(desc)==chunk_ad_fixed(raw[indices[0]][1])})
 assert len(rows)==139 and sorted(j for r in rows for j in r['official_rows'])==list(range(208))
 P.mkdir(exist_ok=True)
 (P/'parsed_annotations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 manifest={'source_file':str(SOURCE),'project_copy':str(PROJECT_SOURCE),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'items':len(rows),'resolved':sum(r['annotation_status']=='human_resolved' for r in rows),'ambiguous':[r['item_number'] for r in rows if r['annotation_status']=='human_ambiguous'],'without_human':sum(r['annotation_status']=='no_human_labels' for r in rows),'exact_text_matches':sum(r['exact_description_matches_original'] for r in rows),'chunker_matches':sum(r['same_chunker_output'] for r in rows),'frozen_models_planned':['C_syntax_baseline__fixed','C_syntax_plus60_fixed_05','adaptive_0_3__per_category','adaptive_1_0__fixed'],'model_selection':'Primary pre-existing calibration-selected adaptive_0_3; other models fixed references; never select or tune using new human labels','scoring_inputs':'Canonical original descriptions matched by explicit source row indices. Markdown is a transcript; raw character-length differences are not silently treated as newly inferred inputs.'}
 (P/'annotation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(manifest,ensure_ascii=False,indent=2))
 for num in [19,23,31]:
  r=rows[num-1];print('ITEM',num,'source rows',r['official_rows'],'human',r['human_label_options']);print(r['description'])
if __name__=='__main__':main()
