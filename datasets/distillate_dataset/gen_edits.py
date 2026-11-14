"""Контролируемые правки реальных объявлений из train (уверенно размеченных).
Типы: NEG (отрицание), OTHER (чужие услуги), OBF (обфускация латиницей),
      SEO (услуга только в SEO-хвосте), SRC (смена исходной mcId).
Запуск: python3 -I gen_edits.py TRAIN.jsonl CONF_IDS.json OUT.jsonl [per_type] [seed]
"""
import json, re, sys, random

train, conf_ids, out = sys.argv[1:4]
PER = int(sys.argv[4]) if len(sys.argv) > 4 else 100
rnd = random.Random(int(sys.argv[5]) if len(sys.argv) > 5 else 2026)

NAMES = {101: "Ремонт под ключ", 102: "Штукатурные работы", 103: "Плиточные работы", 104: "Сантехника",
         105: "Электрика", 106: "Поклейка обоев и малярные работы"}
KW = {102: r'штукатур|по маякам|механизированн',
      103: r'плитк|плиточ|керамогранит|мозаик|кафел|затирк',
      104: r'сантехн|водоснабж|канализац|унитаз|раковин|смесител|полотенцесуш|радиатор|сифон|протечк|инсталляц',
      105: r'электр|проводк|розет|выключател|щит|кабел|светильник|люстр|узо',
      106: r'обои|обоев|поклей|малярн|покрас|окрас|шпакл|стеклохолст'}
KW = {k: re.compile(v, re.I) for k, v in KW.items()}
SEO_RE = re.compile(r'возможно,? вы искали|теги:|ключевые слова', re.I)

ids = set(json.load(open(conf_ids)))
rows = [json.loads(l) for l in open(train, encoding='utf-8')]
rows = [r for r in rows if r['item_id'] in ids and not SEO_RE.search(r['description'])]
rnd.shuffle(rows)

def labs(r): return sorted(m['mcId'] for m in r['found_microcategories'])
def L(ids): return [{"mcId": i, "label": NAMES[i]} for i in sorted(set(ids))]
def mentions(r, c): return bool(KW[c].search(r['title'] + ' ' + r['description']))

# Фразы по категориям для вставки
WORD = {102: ["штукатурные работы", "штукатурку стен", "штукатурку"],
        103: ["плиточные работы", "укладку плитки", "плитку"],
        104: ["сантехнику", "сантехнические работы", "разводку труб и установку сантехники"],
        105: ["электрику", "электромонтажные работы", "замену проводки и розетки"],
        106: ["поклейку обоев", "малярные работы", "покраску стен и поклейку обоев"]}
WORD_NOM = {102: "штукатурка", 103: "плитка", 104: "сантехника", 105: "электрика", 106: "обои и покраска"}

NEG_T = [
    "{W} не делаем.", "{W} не выполняем — только то, что перечислено выше.",
    "Обратите внимание: {w} не берём.", "{W} не предлагаем, обращайтесь к профильным мастерам.",
    "Не занимаемся: {w}.", "Важно: {w} в наши услуги не входит.", "{W} мы не делаем принципиально.",
    "Пожалуйста, не пишите по поводу {wg}: этим не занимаемся.",
]
OTHER_T = [
    "{W} выполняет подрядчик заказчика.", "{W} делает другая бригада, можем дать контакт.",
    "По {wd} можем порекомендовать проверенного мастера, сами этим не занимаемся.",
    "{W} в нашем доме делали соседи — так что знаем, как важно выбрать хорошего мастера.",
    "Если нужна {wn}, наш партнёр сделает это отдельно, мы только координируем.",
    "{W} заказчик обычно делает своими силами до нашего прихода.",
    "Раньше брали {w}, но сейчас этим не занимаемся.",
    "{W} нам часто предлагают, но мы работаем только по своему профилю.",
]
SEO_T = ["Возможно, вы искали: {kws}.", "Теги: {kws}.", "Ключевые слова: {kws}."]
SEO_KW = {102: ["штукатурка стен", "штукатур", "штукатурные работы"], 103: ["укладка плитки", "плиточник", "плитка в ванной"],
          104: ["сантехник", "сантехнические работы", "замена труб"], 105: ["электрик", "электромонтаж", "замена проводки"],
          106: ["поклейка обоев", "маляр", "покраска стен"]}
SEO_FILL = ["ремонт квартир", "ремонт под ключ", "отделка квартиры", "бригада", "мастер", "ремонт дома", "строительные работы"]

def fmt(t, c):
    w = rnd.choice(WORD[c])
    return t.format(w=w, W=w[0].upper() + w[1:], wg=w.replace("ку", "ки").replace("ие работы", "их работ"),
                    wd=WORD_NOM[c].replace("ка", "ке").replace("обои и покраска", "обоям и покраске"),
                    wn=WORD_NOM[c])

def insert(desc, sent, where):
    paras = [p for p in desc.split("\n") if p.strip()]
    if where == "end" or len(paras) < 3: return desc.rstrip() + "\n\n" + sent
    if where == "start": return sent + "\n\n" + desc
    i = rnd.randint(1, len(paras) - 2); paras.insert(i, sent); return "\n\n".join(paras)

# Обфускация: кириллица -> латинские омоглифы
HOMO = {'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x', 'А': 'A', 'В': 'B', 'Е': 'E',
        'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X', 'і': 'i'}
def obfuscate(text, rate):
    return "".join(HOMO[ch] if ch in HOMO and rnd.random() < rate else ch for ch in text)

ads = []
def add(t, r, desc, ids, mcid, conf, note, extra=None):
    a = {"item_id": f"edit-{t}-{len(ads)+1:04d}", "source_item_id": r['item_id'], "title": r['title'],
         "description": desc, "found_microcategories": L(ids), "mcId": mcid, "type": t,
         "confidence": conf, "note": note}
    if extra: a.update(extra)
    ads.append(a)

made = {k: 0 for k in ["NEG", "OTHER", "OBF", "SEO", "SRC"]}

def do_NEG(r, lab, absent):
    k = rnd.choice([1, 1, 2]); cs = rnd.sample(absent, min(k, len(absent)))
    sent = " ".join(fmt(rnd.choice(NEG_T), c) for c in cs)
    add("NEG", r, insert(r['description'], sent, rnd.choice(["end", "mid", "start"])), lab, 101, "high",
        f"Добавлено отрицание про {cs}; метка исходника не меняется", {"injected": cs})

def do_OTHER(r, lab, absent):
    cs = rnd.sample(absent, 1)
    add("OTHER", r, insert(r['description'], fmt(rnd.choice(OTHER_T), cs[0]), rnd.choice(["end", "mid"])), lab, 101, "high",
        f"Упомянута чужая/бывшая услуга {cs}; метка исходника не меняется", {"injected": cs})

def do_OBF(r, lab, absent):
    rate = rnd.choice([0.2, 0.4, 0.7, 1.0])
    add("OBF", r, obfuscate(r['description'], rate), lab, 101, "high",
        f"Кириллица заменена латинскими омоглифами с долей {rate}; метка не меняется", {"obf_rate": rate})

def do_SEO(r, lab, absent):
    cs = rnd.sample(absent, min(rnd.choice([1, 2]), len(absent)))
    kws = [rnd.choice(SEO_KW[c]) for c in cs] + rnd.sample(SEO_FILL, 3); rnd.shuffle(kws)
    add("SEO", r, r['description'].rstrip() + "\n\n" + rnd.choice(SEO_T).format(kws=", ".join(kws)), lab, 101, "high",
        f"Категории {cs} только в SEO-хвосте; в train такое не размечалось в 39/47 случаев", {"injected": cs})

def do_SRC(r, lab, absent):
    src = rnd.choice(lab); exp = [c for c in lab if c != src] + [101]
    add("SRC", r, r['description'], exp, src, "medium",
        f"mcId={src}; ожидаем исходные метки без {src} плюс 101 (объявление было размещено в «Ремонт под ключ»); включение 101 — допущение",
        {"expected_without_101": [c for c in lab if c != src]})

DO = {"NEG": do_NEG, "OTHER": do_OTHER, "OBF": do_OBF, "SEO": do_SEO, "SRC": do_SRC}
for r in rows:
    lab = labs(r); absent = [c for c in NAMES if c != 101 and c not in lab and not mentions(r, c)]
    elig = [t for t in DO if made[t] < PER and (absent if t in ("NEG", "OTHER", "SEO") else True) and (lab if t == "SRC" else True)]
    if not elig:
        if all(v >= PER for v in made.values()): break
        continue
    # для баланса меток: 5-категорийные чаще нужны OBF/SRC, но берём случайно
    t = rnd.choice(elig); DO[t](r, lab, absent); made[t] += 1

for a in ads:
    assert a['mcId'] not in {m['mcId'] for m in a['found_microcategories']}
with open(out, "w", encoding="utf-8") as f:
    for a in ads: f.write(json.dumps(a, ensure_ascii=False) + "\n")
print(made, len(ads))
