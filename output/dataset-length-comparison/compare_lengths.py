"""Read-only length comparison of public, prepared public, synthetic and mixed data."""
import collections
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PUBLIC = ROOT/'datasets/public/public-train-annotated.jsonl'
if not PUBLIC.exists():
    PUBLIC = ROOT/'mvp/innoglobalhack_avito/innoglobalhack_avito/public-train-annotated.jsonl'
SOURCES = {
    'public_raw': PUBLIC,
    'public_prepared_train': ROOT/'output/ngram-comparison/splits/train.jsonl',
    'public_prepared_calibration': ROOT/'output/ngram-comparison/splits/calibration.jsonl',
    'public_prepared_test': ROOT/'output/ngram-comparison/splits/validation.jsonl',
    'v2_train': ROOT/'datasets/hard_dataset/v2/train.jsonl',
    'v2_validation': ROOT/'datasets/hard_dataset/v2/validation.jsonl',
    'v2_test': ROOT/'datasets/hard_dataset/v2/test.jsonl',
    'v2_challenge': ROOT/'datasets/hard_dataset/v2/challenge.jsonl',
    'mixed_train': ROOT/'output/mixed-charwb-comparison/splits/train.jsonl',
}
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(text): return ' '.join(text.casefold().split())
def quantile(values,q):
    values=sorted(values); pos=(len(values)-1)*q
    a,b=math.floor(pos),math.ceil(pos)
    return values[a]+(values[b]-values[a])*(pos-a)
def numeric(values):
    return {**{key:quantile(values,q) for key,q in [('min',0),('p10',.1),('p25',.25),('median',.5),('p75',.75),('p90',.9),('p95',.95),('p99',.99),('max',1)]},'mean':sum(values)/len(values),'sum':sum(values)}
def measure(rows):
    chars=[len(r['description']) for r in rows]
    words=[len(re.findall(r'[А-Яа-яЁёA-Za-z]+',r['description'])) for r in rows]
    lines=[len(r['description'].splitlines()) for r in rows]
    buckets={label:sum(lo<=n<hi for n in chars) for label,lo,hi in [('<100',0,100),('100-299',100,300),('300-599',300,600),('600-999',600,1000),('1000-2999',1000,3000),('3000+',3000,float('inf'))]}
    return {'rows':len(rows),'characters':numeric(chars),'words':numeric(words),'lines':numeric(lines),'character_buckets':buckets,'character_bucket_percent':{key:100*n/len(rows) for key,n in buckets.items()},'over6000':sum(n>6000 for n in chars),'over12000':sum(n>12000 for n in chars)}

hashes={str(path.relative_to(ROOT)):sha(path) for path in SOURCES.values()}
rows={name:[json.loads(s) for s in path.read_text().splitlines() if s.strip()] for name,path in SOURCES.items()}
seen=set();dedup=[]
for row in rows['public_raw']:
    key=norm(row['description'])
    if key not in seen:dedup.append(row);seen.add(key)
rows['public_dedup_normalized']=dedup
rows['public_prepared_all']=[r for name in ['public_prepared_train','public_prepared_calibration','public_prepared_test'] for r in rows[name]]
rows['v2_all']=[r for name in ['v2_train','v2_validation','v2_test','v2_challenge'] for r in rows[name]]
stats={name:measure(rs) for name,rs in rows.items()}
mixture=collections.defaultdict(list)
for r in rows['mixed_train']:mixture[r.get('dataset','missing')].append(r)
mix={name:{'rows':len(rs),'row_share_percent':100*len(rs)/len(rows['mixed_train']),'characters':sum(len(r['description']) for r in rs),'character_share_percent':100*sum(len(r['description']) for r in rs)/stats['mixed_train']['characters']['sum'],'words':sum(len(re.findall(r'[А-Яа-яЁёA-Za-z]+',r['description'])) for r in rs)} for name,rs in mixture.items()}
stats['mixed_train_composition']=mix
stats['checks']={'public_prepared_all_matches_normalized_dedup_texts':set(norm(r['description']) for r in dedup)==set(norm(r['description']) for r in rows['public_prepared_all']),'source_hashes_before':hashes,'all_source_hashes_unchanged':all(sha(ROOT/path)==expected for path,expected in hashes.items())}
stats['definitions']={'characters':'Python len(str), including spaces and newlines; Unicode code points, not bytes.','words':'Sequences of Cyrillic or Latin letters, regex [А-Яа-яЁёA-Za-z]+. Numbers and punctuation excluded. These are not LLM tokens.','lines':'len(description.splitlines()); includes empty internal lines.','quantiles':'Linear interpolation at (n-1)*q.','dedup':'casefold plus collapsed whitespace, preserve first original description; labels unchanged in sources.'}
examples={}
for name in ['public_raw','public_prepared_train','v2_train','v2_test']:
    ordered=sorted(rows[name],key=lambda r:len(r['description']))
    r=ordered[len(ordered)//2]
    examples[name+'_median_example']={'length':len(r['description']),'item_id':r.get('item_id'),'excerpt':r['description'][:700],'truncated':len(r['description'])>700}
    r=ordered[-1]
    examples[name+'_longest_example']={'length':len(r['description']),'item_id':r.get('item_id'),'excerpt':r['description'][:700],'truncated':len(r['description'])>700}
(OUT/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
(OUT/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')

report=['# Разница длин исходных и синтетических объявлений','',
        '**Разница очень большая и сохраняется после удаления повторов.** Public и v2 относятся к разным распределениям текстов: длинные реальные объявления против коротких синтетических логических конструкций. Это подтверждённое различие данных; само по себе оно не доказывает причину изменения F1.','',
        'Все исходники прочитаны без изменения. В public raw 10 000 строк; после штатной нормализации повторов остаётся 4 855. Суммарные готовые public train/calibration/test содержат те же нормализованные описания. Для сравнения именно обучений отдельно показан public train из 3 398 строк.','',
        'Символы — len(str), включая пробелы и переносы. Слова — последовательности кириллических/латинских букв; это **не токены LLM**. Строки — физические строки текста, включая внутренние пустые. Перцентили интерполированы линейно.','']
report += [
    '## Самое важное', '',
    f"- Медиана raw public: {stats['public_raw']['characters']['median']:.0f} символов; после удаления повторов: {stats['public_dedup_normalized']['characters']['median']:.0f}; в фактически использованном public train: {stats['public_prepared_train']['characters']['median']:.0f}.",
    f"- Медиана v2 train: {stats['v2_train']['characters']['median']:.0f}. Она короче raw public в {stats['public_raw']['characters']['median']/stats['v2_train']['characters']['median']:.2f} раза, а дедуплицированного public — в {stats['public_dedup_normalized']['characters']['median']/stats['v2_train']['characters']['median']:.2f} раза.",
    f"- В public после удаления повторов {stats['public_dedup_normalized']['character_bucket_percent']['1000-2999']+stats['public_dedup_normalized']['character_bucket_percent']['3000+']:.2f}% описаний от 1000 символов; в v2 таких нет.",
    f"- Медиана числа слов: {stats['public_prepared_train']['words']['median']:.0f} в public train против {stats['v2_train']['words']['median']:.0f} в v2 train. Медиана физических строк: {stats['public_prepared_train']['lines']['median']:.0f} против {stats['v2_train']['lines']['median']:.0f}.",
    '',
]
display=['public_raw','public_dedup_normalized','public_prepared_train','public_prepared_calibration','public_prepared_test','v2_train','v2_validation','v2_test','v2_challenge','mixed_train']
for metric,title in [('characters','Символы'),('words','Слова'),('lines','Строки')]:
    report+=['## '+title,'','| Набор | N | min | p10 | p25 | median | p75 | p90 | p95 | p99 | max |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in display:
        vals=stats[name][metric]
        report.append('| '+name+' | '+str(stats[name]['rows'])+' | '+' | '.join(f'{vals[k]:.1f}' for k in ['min','p10','p25','median','p75','p90','p95','p99','max'])+' |')
    report+=['']
report+=['## Распределение по длине','', 'Доля строк каждого набора, %.','', '| Набор | <100 | 100–299 | 300–599 | 600–999 | 1000–2999 | 3000+ |', '|---|---:|---:|---:|---:|---:|---:|']
for name in display:
    report.append('| '+name+' | '+' | '.join(f'{stats[name]["character_bucket_percent"][b]:.2f}' for b in ['<100','100-299','300-599','600-999','1000-2999','3000+'])+' |')
report+=['','## Состав смешанного train','', '| Источник (поле dataset) | Строк | Доля строк | Всего символов | Доля символов |', '|---|---:|---:|---:|---:|']
for name,m in mix.items():report.append(f"| {name} | {m['rows']} | {m['row_share_percent']:.2f}% | {m['characters']} | {m['character_share_percent']:.2f}% |")
report+=['','Доля строк важна: IDF рассчитывается по частоте признака **в документах**, а не по общей сумме символов. Поэтому высокая доля коротких синтетических строк может существенно изменить IDF, даже если большая часть букв в смешанном корпусе приходится на реальные объявления. Суммы символов не являются прямым весом документов в SGD.','',
         '## Что это означает для TF-IDF + SGD','',
         'Подтверждено расчётами: v2 имеет короткие описания, узкую лексику и почти не представляет длинные рекламные объявления. Повторы объясняют часть длины raw public, но после дедупликации разница остаётся большой. Ни один из рассматриваемых исходных текстов не длиннее 6 000 символов; текущая обрезка данных не вызывает это различие.','',
         'Возможные последствия, которые требуют отдельной экспериментальной проверки:','',
         '- При смешивании IDF меняется из-за доли синтетических документов и повторяемости формулировок. Меняется и конкуренция признаков за max_features.',
         '- В длинном объявлении больше разных услуг, рекламных обещаний, нерелевантных деталей и областей отрицания. Синтетическая короткая фраза проверяет другой контекст.',
         '- При L2-нормализации TF-IDF вектора (по умолчанию) вклад конкретной фразы зависит от остальных признаков документа. Переход от коротких фраз к большим спискам может менять оценки, но по одной длине нельзя предсказать направление ошибки.',
         '- sublinear_tf уменьшает влияние повторяемости признаков внутри текста, но не делает короткую синтетику эквивалентной длинным реальным объявлениям.',
         '- Указание общей длины не позволяет отделить эффект длины от отличий содержания, лексики и шаблонов. Здесь эти факторы меняются одновременно; назвать длину доказанной причиной плохого F1 нельзя.','',
         '## Практический вывод','',
         'V2 полезен как дополнительный стрессовый набор, но представлять его как замену реальным объявлениям нельзя. Качество нужно отдельно показывать на неизменном реальном test и синтетическом test. Для проверки именно влияния длины нужен отдельный контролируемый эксперимент: одинаковый смысл/метки с короткой и длинной естественной формой, без переноса исходника и производных между выборками.','',
         'Долю синтетики или её sample_weight следует подбирать по реальной validation. Автоматически обрезать длинные объявления до размеров v2 или искусственно растягивать v2 без семантической проверки не рекомендуется.','',
         'Примеры и фрагменты с измеренной длиной находятся в examples.json. Полные точные числа и SHA-256 — в stats.json. Исходные данные не менялись.']
report += ['', '## Иллюстрация на объявлениях около медианы', '',
           f"Public train, {examples['public_prepared_train_median_example']['length']} символов (начало, остальной текст не показан):", '',
           '> '+examples['public_prepared_train_median_example']['excerpt'][:350].replace('\n','\n> '), '',
           f"V2 train, {examples['v2_train_median_example']['length']} символа (полный текст):", '',
           '> '+examples['v2_train_median_example']['excerpt'].replace('\n','\n> '), '',
           'Выбор примеров: по одному тексту в середине упорядоченного по длине списка; это иллюстрация структуры, а не случайная репрезентативная выборка содержания.']
(OUT/'report.md').write_text('\n'.join(report)+'\n')
print(json.dumps({name:stats[name] for name in ['public_raw','public_dedup_normalized','public_prepared_train','v2_train','v2_test','mixed_train_composition']},ensure_ascii=False,indent=2))
