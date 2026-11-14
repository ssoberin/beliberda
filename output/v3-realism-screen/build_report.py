"""Summarize review-only candidates; no dataset filtering or model evaluation."""
import collections,hashlib,json
from pathlib import Path
P=Path('output/v3-realism-screen');s=json.loads((P/'summary.json').read_text());c=json.loads((P/'candidates.json').read_text())
keys=['repeated_price','repeated_team','repeated_faq','repeated_offer_correction']
priority=[r for r in c['train'] if max(r['text_counts'][k] for k in keys)>=5]
assert len(priority)==79
before={r['item_id'] for r in c['train'] if r['tier']=='rewrite_candidate'}
after={r['item_id'] for r in c['revision3_train'] if r['tier']=='rewrite_candidate'}
assert before==after and len(before)==472
for path,digest in s['sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
card=collections.Counter(len(set(r['found'])&set(range(102,107))) for r in priority)
plan={'status':'REVIEW_ONLY_NO_DATA_FILTERING','priority_train_count':79,'priority_rule':'At least 5 repeats of same actor/FAQ/offer-correction/price framing within one text; this is an expert realism heuristic, not a label-error finding.','all_rewrite_candidates_train':472,'same472_retained_in_revision3':True,'priority_scope_target_cardinality':dict(sorted(card.items())),'priority_ids':[r['item_id'] for r in priority],'suggested_action':'Review and rewrite strongest repetition while preserving services/labels. If excluded from main training, keep as stress material and report effects on class/cardinality/length distributions. No test removal for higher scores.','caveat':'Not a complete human semantic audit. Valid negations/actor distinctions must not be removed merely because the classifier errs.'}
(P/'priority79.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
(P/'priority79_examples.json').write_text(json.dumps(priority,ensure_ascii=False,indent=2)+'\n')
lines=['# V3: что выглядит искусственным и что проверять перед удалением','',
'**Измерено:** в исходном train3000 найдено472текста с ≥3повторениями одной сложной статусной рамки. Из них79содержат5–6повторений и являются первым списком на смысловой просмотр/переписывание. **Экспертная оценка:** такая многократная конструкция выглядит учебной. Это не объективная частота на Авито и не доказательство неправильных меток.','',
'Проверены сами тексты всех4344строк исходногоv3, новые250и итоговый экспериментальныйtrain3000. Автоматическая проверка охватывает весь корпус; смысловой просмотр — характерные примеры каждого флага и код их сборки, не независимая ручная проверка каждой строки. Предсказания и ошибки модели не использовались. Файлы источников, пороги и модель не менялись; SHA-256проверены повторно.','',
'## Кандидаты','', '| Часть | Строк | Сильный флаг: переписать/проверить | Только старые SEO-теги | Без этих сильных флагов |','|---|---:|---:|---:|---:|']
for name in ['train','validation','test','challenge','revision3_new250','revision3_train']:
 r=s['sets'][name];n=r['rows'];tiers=r['tiers'];f=tiers.get('rewrite_candidate',0)
 lines.append(f"| {name} | {n} | {f} ({100*f/n:.2f}%) | {tiers.get('review_stale_tag',0)} | {tiers.get('no_high_priority_flag',0)} |")
lines+=['','Категории взаимоисключающие: SEO-теги могут также встречаться среди сильных кандидатов. Challenge144—24описания, не144независимыхтекста. Отсутствие флага не подтверждает реалистичность; в частности, новые250построены как50контрастныхсюжетов с5версиями и остаются учебным дополнением.','',
'## Что именно повторяется в train','', '| Конструкция, ≥3вхождения в тексте | Текстов | Пример |','|---|---:|---|',
'| Отказ от бесплатности с подтверждением платного заказа | 133 | «Бесплатно … не делаем; по согласованной смете такие заказы принимаем» |',
'| Руководитель сам не делает, собственная бригада делает | 112 | «Лично руководитель не выполняет …, но наша бригада делает это по нашему договору» |',
'| Прежний отказ / исправление старого предложения | 111 | «В старом объявлении писали … Исправление: эту услугу выполняем» |',
'| Повтор одного вопроса и утвердительного ответа | 116 | «Спрашивают: “Вы выполняете …?” Да, принимаем такие заказы» |','',
'Один такой оборот может быть правдоподобным и полезным. Флаг вызывает многократное однотипное повторение, а не наличие отрицания само по себе. Отдельно371train-текст содержит явные старыеSEO-теги;292изнихнеимеютдругихсильныхфлагов. Это необычный сценарий, но единичный случай не равен мусору. Вtest43текста прямо объясняют логику: «отказ относится к бесплатной работе, а не к самой услуге». Их лучше рассматривать как стрессовую формулировку, а не как обычную рекламу.','',
'## Противоречия и разметка','',
'В проверках не найдено нарушений found=offered−source, повреждённых evidence-span или категории одновременно со статусом offered и not_offered. Для одинакового нормализованного текста при одинаковомsource конфликтующихgold-групп не найдено. Это формальная согласованность, **не доказательство полной смысловой корректности**: generator мог последовательно присвоить ошибочный смысл. В прочитанных примерах не установлены явные перевёрнутые метки; независимая человеческая проверка всё ещё нужна. Разные операции внутри одной категории могут иметь разные статусы: это допустимый частичный отказ.','',
'## Самые явные примеры на переписывание','']
for reason in ['repeated_team_ge3','repeated_offer_correction_ge3','repeated_price_ge3','repeated_faq_ge3']:
 r=max((r for r in c['train'] if reason in r['reasons']),key=lambda r:max(r['text_counts'][k] for k in keys))
 lines += [f"- `{r['item_id']}`: {max(r['text_counts'][k] for k in keys)}однотипныхповторений; offered={r['offered']}. Причина: {reason}."]
lines+=['','Полные тексты и решения эвристик:priority79_examples.json. IDпервойочереди:priority79.json. Все кандидаты по всемчастям:candidates.json.','',
'## Рекомендация','',
'Начать с79самыхповторяющихсяtrain-текстов: просмотреть и переписать как естественные предложения с теми же услугами, оставив не больше1–2контекстныхусложнений. Не удалять472автоматически: они сосредоточены на многометочных ответах, и массовое удаление упроститраспределение. Для79приоритетныхтекстовкардинальность102–106: '+str(dict(sorted(card.items())))+'.','',
'Наши предыдущие250замен **сохранили все472исходныхсильныхкандидата**: мы выбрали простые строки, а не многократно искусственные рамки. Это объясняет, почему та замена не являлась очисткой от таких сценариев; это не доказательство причины паденияF1.','',
'Если исключать эти случаи из основногоtrain, сохранять их отдельно какстрессовыйматериал; удаление не делает их подходящими для независимогоtest, поскольку модели уже видели их при обучении. После исправления train сравнить на прежних неизменённых проверках. Отдельный более естественныйtest можно составить как новую версию с опубликованными критериями, но не выдавать облегчённыйF1заулучшение модели.','',
'Ничего не удалено, новые fit не запускались.']
(P/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(plan,ensure_ascii=False,indent=2))
