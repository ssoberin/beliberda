"""Generate a new authored corpus, without reading any public/legacy records."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
import re

from lexicon import CONTEXT, NAMES, OUTSIDE, SERVICES, TYPO, service_bank, detail_bank
from workflows import workflow_bank, SHORT_SERVICES, SHORT_FRAMES

ROOT = Path(__file__).resolve().parent
IDS = tuple(NAMES)
CARDINALITY = {0: 12, 1: 26, 2: 28, 3: 22, 4: 9, 5: 3}
LENGTHS = {'short': (80, 249, 20), 'medium': (250, 699, 35),
           'long': (700, 1499, 25), 'extended': (1500, 3500, 20)}
WORDS = re.compile(r'[а-яёa-z]+', re.I)
PARTICLES = re.compile(r'\b(?:не|ни|нет|только|кроме|без|зато|теперь|раньше)\b', re.I)

# Each split has its own surface constructions as well as service vocabulary.
FRAMES = {
 'train': {
  'direct': ['Выполняем {s}.', 'Предлагаем {s}.', 'Берём заказы на {s}.'],
  'condition': ['Выполняем {s}, если обеспечен доступ к рабочему участку.', 'За {s} берёмся по предварительной записи.'],
  'price': ['Бесплатно {s} не делаем; по согласованной смете такие заказы принимаем.'],
  'past': ['Раньше не брались за {s}, теперь такую работу выполняем.', 'В старом объявлении писали, что не выполняем {s}. Исправление: эту услугу выполняем.'],
  'faq': ['Спрашивают: «Вы выполняете {s}?» Да, принимаем такие заказы.'],
  'team': ['Лично руководитель не выполняет {s}, но наша бригада делает это по нашему договору.'],
  'not_only': ['Не только {a}, но и {b} можно заказать у нас.'],
  'denied': ['За {s} не берёмся.', 'Не предлагаем {s}.'],
  'other': ['Заказчик заказывает {s} у другой бригады; мы эту услугу не оказываем.'],
  'former': ['Раньше предлагали {s}, сейчас такие заказы не принимаем.'],
  'quote': ['Клиент спросил про {s}. Ответили, что эту работу не выполняем.'],
  'seo': ['В старых тегах осталось: {s}. Это не наше действующее предложение, такую работу не берём.'],
 },
 'validation': {
  'direct': ['У нас заказывают {s}.', 'Можем выполнить {s}.'],
  'condition': ['При свободном подходе к участку готовы выполнить {s}.', 'По согласованной дате принимаем {s}.'],
  'price': ['Не обещаем выполнить {s} бесплатно, но по смете эту услугу оказываем.'],
  'past': ['Сейчас в перечень добавили {s}; раньше такой услуги у нас не было.', 'Отказ выполнять {s} в прежнем тексте устарел. Сейчас это действующая услуга.'],
  'faq': ['Вопрос из чата: «Можно заказать {s}?» Ответ: да, мы это выполняем.'],
  'team': ['Бригадир лично не выезжает выполнять {s}; за эту работу отвечают наши штатные сотрудники по нашему договору.'],
  'not_only': ['Наш перечень не ограничен одной услугой: выполняем {a} и {b}.'],
  'denied': ['Заявки на {s} отклоняем: этим не занимаемся.', 'Заказывать у нас {s} не нужно, такой услуги нет.'],
  'other': ['Для заявки на {s} нужен отдельный договор с чужими мастерами, мы этим не занимаемся.'],
  'former': ['Из перечня убрали {s}: это прежняя услуга, больше её не оказываем.'],
  'quote': ['Клиент хотел заказать {s}. Мы отказались, эта задача за рамками наших услуг.'],
  'seo': ['Поисковый запрос «заказать {s}» устарел: соответствующих заказов мы не принимаем.'],
 },
 'test': {
  'direct': ['Принимаем в работу {s}.', 'Готовы взять на себя {s}.'],
  'condition': ['Записаться на {s} можно после согласования доступа в помещение.', 'Принимаем {s}, но дату начала нужно подтвердить заранее.'],
  'price': ['За оплату выполняем {s}; отказ относится к бесплатной работе, а не к самой услуге.'],
  'past': ['Прежде не брались выполнять {s}, а сейчас такие заказы принимаем.', 'Сообщение о том, что мы не делаем {s}, относится к прошлому. Теперь такие заявки принимаем.'],
  'faq': ['В переписке уточняют, делаем ли мы {s}. Отвечаем утвердительно, услуга доступна.'],
  'team': ['Хотя владелец сам не выполняет {s}, его штатные мастера делают это по договору нашей бригады.'],
  'not_only': ['Помимо того что выполняем {a}, также предлагаем {b}.'],
  'denied': ['Выполнять {s} не готовы, таких услуг у нас нет.', 'Заявки на {s} сейчас не принимаем.'],
  'other': ['На {s} приедет исполнитель, которого заказчик нанял самостоятельно; мы эту работу не предлагаем.'],
  'former': ['Предложение на {s} больше не действует, сейчас мы за это не берёмся.'],
  'quote': ['Нас попросили выполнить {s}, но получили отказ: такой услуги у нас нет.'],
  'seo': ['Слова «{s}» попали в поисковые теги случайно; выполнять это не предлагаем.'],
 },
}

def normalized(text):
    return ' '.join(text.casefold().split())

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def labels(values):
    return [{'mcId': c, 'label': NAMES[c]} for c in sorted(values)]

def corrupt(text, mode, rng):
    if mode == 'clean':
        return text
    if mode == 'punctuation':
        # Keep sentences, negations and scope. Only remove some commas.
        changed = re.sub(',', lambda m: '' if rng.random() < .45 else m[0], text)
        return changed if changed != text else text.replace(' ', '  ', 1)
    matches = list(re.finditer(r'[а-яё]+', text, re.I))
    safe = [m for m in matches if m[0].lower() in TYPO]
    chosen = rng.sample(safe, min(len(safe), 2))
    for m in sorted(chosen, key=lambda x: x.start(), reverse=True):
        replacement = TYPO[m[0].lower()]
        if m[0][0].isupper(): replacement = replacement.capitalize()
        text = text[:m.start()] + replacement + text[m.end():]
    return text

def render(split, source, targets, length_bin, noise, rng, require_negative=False):
    bank, frames = service_bank(split), FRAMES[split]
    offered = set(targets)
    if rng.random() < .7 and not (require_negative and len(targets) == 5):
        offered.add(source)
    scenario = rng.choices(['direct', 'condition', 'price', 'past', 'faq', 'team', 'not_only'],
                           [25, 20, 10, 10, 10, 10, 15])[0]
    if scenario == 'not_only' and len(offered) < 2:
        scenario = 'direct'
    blocks = []
    def add(category, status, kind, quote):
        blocks.append({'mcId': category, 'status': status, 'kind': kind, 'quote': quote})

    if length_bin == 'short':
        compact, short = SHORT_SERVICES[split], SHORT_FRAMES[split]
        ordered = rng.sample(sorted(offered), len(offered))
        if ordered:
            if scenario == 'not_only':
                a, b, *rest = ordered
                quote = short['not_only'].format(a=compact[a], b=compact[b])
                for c in (a, b): add(c, 'offered', scenario, quote)
                if rest:
                    quote = short['direct'].format(items=', '.join(compact[c] for c in rest))
                    for c in rest: add(c, 'offered', 'direct', quote)
            else:
                quote = short[scenario].format(items=', '.join(compact[c] for c in ordered))
                for c in ordered: add(c, 'offered', scenario, quote)
        absent = sorted(set(IDS) - offered)
        if absent and (require_negative or rng.random() < .65):
            c = rng.choice(absent)
            kind = rng.choice(['denied', 'other', 'former', 'quote'])
            phrase = compact[c]
            add(c, 'not_offered', kind, short[kind].format(s=phrase, S=phrase[0].upper() + phrase[1:]))
    else:
        paired = set()
        if scenario == 'not_only':
            a, b = rng.sample(sorted(offered), 2)
            quote = rng.choice(frames['not_only']).format(a=rng.choice(bank[a]), b=rng.choice(bank[b]))
            for c in (a, b): add(c, 'offered', 'not_only', quote)
            paired = {a, b}
        for c in sorted(offered - paired):
            phrase = rng.choice(bank[c])
            quote = rng.choice(frames['direct' if scenario == 'not_only' else scenario]).format(s=phrase)
            add(c, 'offered', scenario, quote)
            if length_bin in ('long', 'extended') and rng.random() < .45:
                other = rng.choice([s for s in bank[c] if s != phrase])
                add(c, 'offered', 'additional_operation', rng.choice(frames['direct']).format(s=other))
        absent = sorted(set(IDS) - offered)
        if absent and rng.random() < .65:
            for c in rng.sample(absent, min(len(absent), rng.choice([1, 1, 2]))):
                kind = rng.choice(['denied', 'other', 'former', 'quote', 'seo'])
                add(c, 'not_offered', kind, rng.choice(frames[kind]).format(s=rng.choice(bank[c])))
    if not offered:
        # Extended outside-scope cases elaborate furniture assembly, not a
        # random other job which would contradict the furniture narrative.
        quote = 'Собираем мебель из готовых деталей.' if length_bin in ('long', 'extended') else rng.choice(OUTSIDE)
        add(None, 'outside_scope', 'outside_scope', quote)

    style = rng.choice(['paragraphs', 'bullets', 'compact'])
    separator = '\n\n' if style == 'paragraphs' else '\n' if style == 'bullets' else ' '
    lo, hi, _ = LENGTHS[length_bin]
    desired = rng.randint(lo, hi)
    if length_bin in ('long', 'extended'):
        technical = []
        for c in sorted(offered) if offered else [0]:
            paragraphs = workflow_bank(split)[c]
            if c:
                paragraphs = detail_bank(split)[c] + paragraphs
            technical.extend((c, paragraph) for paragraph in paragraphs)
        rng.shuffle(technical)
        # Service content must supply at least 70% of the intended length.
        # Do not repeat paragraphs just to reach the quota.
        for c, paragraph in technical:
            if sum(len(b['quote']) for b in blocks) >= desired * .75:
                break
            if sum(len(b['quote']) + 2 for b in blocks) + len(paragraph) <= hi - 20:
                add(c if c else None, 'offered' if c else 'outside_scope',
                    'technical_detail' if c else 'outside_workflow', paragraph)
    rng.shuffle(blocks)
    quotes = list(dict.fromkeys(b['quote'] for b in blocks))
    intro = rng.choice({
        'train': ['Небольшая бригада. Вот за что берёмся.', 'Услуги для вашего дома.', 'Принимаем заявки на ближайшие недели.', 'Здравствуйте! Уточняем наш перечень работ.'],
        'validation': ['Работаем по согласованному перечню.', 'Добрый день. Расскажем о доступных услугах.', 'Оставить заявку можно в переписке.'],
        'test': ['Частная команда мастеров на связи.', 'Ниже перечислено, что можно заказать.', 'Перед обращением ознакомьтесь с условиями.'],
    }[split])
    if length_bin == 'short':
        intro = ''
    parts = ([intro] if intro else []) + [('• ' if style == 'bullets' else '') + quote for quote in quotes]
    contexts = []
    context = rng.sample(CONTEXT[split], len(CONTEXT[split]))
    while len(separator.join(parts)) < desired and context:
        paragraph = context.pop()
        size = len(separator.join(parts + [paragraph]))
        context_size = sum(map(len, contexts)) + len(paragraph)
        cap_ok = length_bin not in ('long', 'extended') or context_size / size <= .35
        if size <= hi and cap_ok:
            parts.insert(rng.randint(0, len(parts)), paragraph)
            contexts.append(paragraph)
    text = separator.join(parts)
    if not lo <= len(text) <= hi:
        return None
    dirty = corrupt(text, noise, rng)
    if noise != 'clean' and dirty == text:
        return None
    if [s.lower() for s in PARTICLES.findall(text)] != [s.lower() for s in PARTICLES.findall(dirty)]:
        raise AssertionError('Noise changed a logical operator')

    def locate(quote):
        pattern = ''.join(re.escape(ch) if ch != ',' else ',?' for ch in quote)
        if noise == 'spelling':
            for word, replacement in TYPO.items():
                pattern = re.sub(re.escape(word), lambda _: f'(?:{word}|{replacement})', pattern, flags=re.I)
        # Punctuation noise may add one space; keep all actual offsets.
        if noise == 'punctuation': pattern = pattern.replace(' ', ' +')
        return re.search(pattern, dirty, re.I)
    evidence = []
    for block in blocks:
        match = locate(block['quote'])
        if not match: return None
        evidence.append({**block, 'quote': match[0], 'start': match.start(), 'end': match.end()})
    context_spans = []
    for quote in contexts:
        match = locate(quote)
        if not match: return None
        context_spans.append({'quote': match[0], 'start': match.start(), 'end': match.end()})
    context_chars = sum(e['end'] - e['start'] for e in context_spans)
    if length_bin in ('long', 'extended') and context_chars / len(dirty) > .35:
        return None
    if not lo <= len(dirty) <= hi: return None
    return {'title': rng.choice(['Услуги частной бригады', 'Мастера для вашего дома', 'Работы по согласованной смете']),
            'description': dirty, 'mcId': source, 'found_microcategories': labels(targets),
            'offered_microcategories': sorted(offered), 'evidence': evidence,
            'context_spans': context_spans, 'generic_context_chars': context_chars,
            'length_bin': length_bin, 'noise': noise, 'layout': style,
            'construction': scenario, 'provenance': 'authored_synthetic_v3',
            'annotation_status': 'scenario_defined_not_independently_reviewed'}

def build(seed=2026):
    rng = random.Random(seed)
    splits, seen = {}, set()
    for split, count in [('train', 3000), ('validation', 600), ('test', 600)]:
        tasks = []
        category_counts = Counter({c: 0 for c in IDS})
        for k, percent in CARDINALITY.items():
            sources = list(IDS) * (count * percent // 100 // 6)
            rng.shuffle(sources)
            for source in sources:
                options = list(combinations([c for c in IDS if c != source], k))
                rng.shuffle(options)
                targets = min(options, key=lambda cats: sum(category_counts[c] for c in cats))
                category_counts.update(targets)
                tasks.append((source, targets))
        rng.shuffle(tasks)
        lengths = [name for name, (_, _, p) in LENGTHS.items() for _ in range(count * p // 100)]
        noises = ['clean'] * (count * 90 // 100) + ['spelling'] * (count * 7 // 100) + ['punctuation'] * (count * 3 // 100)
        rng.shuffle(lengths)
        rng.shuffle(noises)
        rows = []
        short_counts = Counter()
        for i, ((source, targets), length_bin, noise) in enumerate(zip(tasks, lengths, noises), 1):
            # Alternate scope traps within every short cardinality stratum,
            # including five labels, rather than dropping traps to fit length.
            require_negative = length_bin == 'short' and short_counts[len(targets)] % 2 == 0
            for _ in range(2000):
                row = render(split, source, targets, length_bin, noise, rng, require_negative)
                if row is None: continue
                key = normalized(row['description'])
                if key in seen: continue
                seen.add(key)
                row.update(item_id=f'own-v3-{split}-{i:04d}',
                           group_id=f'own-v3-{split}-{digest(key)[:16]}')
                rows.append(row)
                if length_bin == 'short': short_counts[len(targets)] += 1
                break
            else:
                raise RuntimeError(f'Cannot generate {split}: {source}, {targets}, {length_bin}, {noise}')
        splits[split] = rows
    return splits

def validate(splits):
    ids, texts, groups = set(), set(), {}
    for split, rows in splits.items():
        for row in rows:
            assert row['item_id'] not in ids
            ids.add(row['item_id'])
            key = normalized(row['description'])
            assert key not in texts, 'Repeated description'
            texts.add(key)
            assert groups.setdefault(row['group_id'], split) == split
            offered = set(row['offered_microcategories'])
            assert offered <= set(IDS)
            assert row['mcId'] in IDS
            assert row['found_microcategories'] == labels(offered - {row['mcId']})
            ev_offered = {e['mcId'] for e in row['evidence'] if e['status'] == 'offered'}
            assert ev_offered == offered
            for e in row['evidence']:
                assert row['description'][e['start']:e['end']] == e['quote']
                if e['status'] == 'not_offered': assert e['mcId'] not in offered
                if e['kind'] == 'technical_detail':
                    # Global service refusals belong in explicit status blocks,
                    # never in independently sampled technical explanations.
                    assert not re.search(r'не (?:выполня\w*|предлага\w*|включа\w*|вход\w*|заявля\w*)', e['quote'], re.I)
            lo, hi, _ = LENGTHS[row['length_bin']]
            assert lo <= len(row['description']) <= hi
            for e in row['context_spans']:
                assert row['description'][e['start']:e['end']] == e['quote']
            assert row['generic_context_chars'] == sum(e['end'] - e['start'] for e in row['context_spans'])
            if row['length_bin'] in ('long', 'extended'):
                assert row['generic_context_chars'] / len(row['description']) <= .35
        count = len(rows)
        for k, p in CARDINALITY.items():
            assert sum(len(r['found_microcategories']) == k for r in rows) == count * p // 100
        for c in IDS:
            assert sum(r['mcId'] == c for r in rows) == count // 6
        for name, (_, _, p) in LENGTHS.items():
            assert sum(r['length_bin'] == name for r in rows) == count * p // 100
        assert sum(r['noise'] != 'clean' for r in rows) == count // 10

def quantile(values, q):
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    a = math.floor(pos)
    return round(xs[a] + (xs[math.ceil(pos)] - xs[a]) * (pos - a), 1)

def summarize(rows):
    texts = [r['description'] for r in rows]
    words = [w.casefold() for text in texts for w in WORDS.findall(text)]
    sample = random.Random(2026).sample(words, min(8000, len(words)))
    counts = lambda xs: dict(sorted(Counter(xs).items()))
    context_share = {}
    for name in ('long', 'extended'):
        values = [100 * r['generic_context_chars'] / len(r['description']) for r in rows if r['length_bin'] == name]
        context_share[name] = {'median_pct': quantile(values, .5), 'max_pct': quantile(values, 1)}
    short_diagnostics = {}
    for k in range(6):
        subset = [r for r in rows if r['length_bin'] == 'short' and len(r['found_microcategories']) == k]
        short_diagnostics[k] = {'rows': len(subset),
                                'non_direct': sum(r['construction'] != 'direct' for r in subset),
                                'with_negative_mentions': sum(any(e['status'] == 'not_offered' for e in r['evidence']) for r in subset)}
    return {'rows': len(rows), 'unique_descriptions': len({normalized(t) for t in texts}),
            'generic_context_share': context_share,
            'short_diagnostics': short_diagnostics,
            'characters': {name: quantile([len(t) for t in texts], q) for name, q in [('min', 0), ('median', .5), ('p95', .95), ('max', 1)]},
            'length_bins': counts(r['length_bin'] for r in rows),
            'source_counts': counts(r['mcId'] for r in rows),
            'target_counts': counts(m['mcId'] for r in rows for m in r['found_microcategories']),
            'target_cardinality': counts(len(r['found_microcategories']) for r in rows),
            'noise': counts(r['noise'] for r in rows),
            'construction': counts(r['construction'] for r in rows),
            'evidence_kinds': counts(e['kind'] for r in rows for e in r['evidence']),
            'word_forms': len(set(words)), 'words': len(words),
            'random_8000_words_unique_pct': round(100 * len(set(sample)) / len(sample), 2)}

def write(output, splits, seed):
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, rows in splits.items():
        data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        (output / f'{split}.jsonl').write_text(data, encoding='utf-8')
        hashes[f'{split}.jsonl'] = digest(data)
    stats = {'version': 'own_v3', 'revision': 2, 'seed': seed, 'inputs': ['lexicon.py', 'generate.py', 'challenge.py', 'workflows.py'],
             'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('lexicon.py', 'generate.py', 'challenge.py', 'workflows.py')},
             'splits': {s: summarize(rows) for s, rows in splits.items()}, 'sha256': hashes,
             'limitations': ['Entirely synthetic; does not establish real-world representativeness.',
                             'Scenario-defined labels have no independent human adjudication.',
                             'Split-disjoint service phrases, frames and context banks; shared meanings remain.',
                             'Exact deduplication does not prove absence of all semantic near-duplicates.',
                             'No public records, old datasets, or model predictions used.']}
    (output / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# Проверка собственного корпуса v3', '', 'Синтетика по авторским сценариям; метрики ниже описывают данные, а не точность модели.', '',
             '| Выборка | Строк | Медиана длины | p95 | Максимум | Словоформ |', '|---|---:|---:|---:|---:|---:|']
    for s, m in stats['splits'].items():
        c = m['characters']
        lines.append(f"| {s} | {m['rows']} | {c['median']} | {c['p95']} | {c['max']} | {m['word_forms']} |")
    lines += ['', '## Общие условия заказа: доля символов', '',
              '| Выборка | Длинные: медиана, % | Развёрнутые: медиана, % | Максимум, % |', '|---|---:|---:|---:|']
    for s, m in stats['splits'].items():
        c = m['generic_context_share']
        lines.append(f"| {s} | {c['long']['median_pct']} | {c['extended']['median_pct']} | {max(c['long']['max_pct'], c['extended']['max_pct'])} |")
    lines += ['', '## Короткие примеры', '',
              '| Выборка | Целевых категорий | Записей | Непрямых конструкций | С отрицательными упоминаниями |', '|---|---:|---:|---:|---:|']
    for s, m in stats['splits'].items():
        for k, d in m['short_diagnostics'].items():
            lines.append(f"| {s} | {k} | {d['rows']} | {d['non_direct']} | {d['with_negative_mentions']} |")
    lines += ['', 'Полные распределения, SHA-256 и ограничения: `stats.json`.', '',
              'Проверены: точные квоты категорий/длин/шума, исключение источника, доказательные фразы и их позиции, отсутствие повторных описаний между всеми выборками.',
              'Разметка задаётся сценариями, а не поиском ключевых слов. Корректные ссылки на фразы не заменяют независимую проверку смысла.',
              'Длинные тексты включают условия заказа и несколько операций одной услуги. Общие абзацы остаются шаблонными; увеличение длины не равно увеличению независимого смыслового разнообразия.',
              'В ревизии 2 доля общих условий ограничена 35% каждого длинного/развёрнутого текста. Остальное составляют описания услуг, хода работ, условий их предложения и отрицательных упоминаний.',
              'Лексика услуг, конструкции и контекстные абзацы разделены между train/validation/test. Не объединяйте выборки перед случайным разделением.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    from challenge import build_challenge
    challenge = build_challenge()
    challenge_data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in challenge)
    (output / 'challenge.jsonl').write_text(challenge_data, encoding='utf-8')
    stats['challenge'] = {'rows': len(challenge), 'independent_descriptions': len(challenge) // 6,
                          'purpose': 'Scope and source-category probes; not training or prevalence estimation.'}
    stats['sha256']['challenge.jsonl'] = digest(challenge_data)
    (output / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return stats

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    splits = build(args.seed)
    validate(splits)
    stats = write(args.output_dir, splits, args.seed)
    print(json.dumps(stats['splits'], ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
