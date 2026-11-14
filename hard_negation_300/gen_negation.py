"""Ловушки с отрицанием: настоящие отрицания, ложные («не только», «не дорого») и смешанные.
Метка считается из множеств offered / denied. Запуск: python3 -I gen_negation.py OUT.jsonl [N] [seed]
"""
import json, re, sys, random

N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
rnd = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 2026)
NAMES = {102: "Штукатурные работы", 103: "Плиточные работы", 104: "Сантехника",
         105: "Электрика", 106: "Поклейка обоев и малярные работы"}
C = list(NAMES)

# падежи: nom, acc, gen, ins, dat, adj-plural для «X работы», профессия
W = {102: dict(nom="штукатурка", acc="штукатурку", gen="штукатурки", ins="штукатуркой", dat="штукатурке", pro="штукатура", prof="штукатур"),
     103: dict(nom="плитка", acc="плитку", gen="плитки", ins="плиткой", dat="плитке", pro="плиточника", prof="плиточник"),
     104: dict(nom="сантехника", acc="сантехнику", gen="сантехники", ins="сантехникой", dat="сантехнике", pro="сантехника", prof="сантехник"),
     105: dict(nom="электрика", acc="электрику", gen="электрики", ins="электрикой", dat="электрике", pro="электрика", prof="электрик"),
     106: dict(nom="малярка", acc="малярку", gen="малярки", ins="маляркой", dat="малярке", pro="маляра", prof="маляр")}
# конкретные услуги (1 л. ед. ч.)
SV = {102: ["штукатурю стены по маякам", "выравниваю стены штукатуркой", "штукатурю откосы и углы", "делаю механизированную штукатурку"],
      103: ["кладу плитку", "кладу плитку и керамогранит", "облицовываю ванную плиткой", "делаю фартук на кухне", "кладу мозаику"],
      104: ["меняю трубы", "ставлю унитазы и раковины", "меняю смесители", "подключаю стиральные машины", "делаю разводку воды"],
      105: ["меняю проводку", "ставлю розетки и выключатели", "собираю щиты", "вешаю люстры", "переношу электроточки"],
      106: ["клею обои", "красю стены и потолки", "шпаклюю под покраску", "клею обои и красю", "снимаю старые обои и клею новые"]}
# частные работы для «смешанных» случаев
PART = {105: ["розетки перенести и люстру повесить могу", "подключу светильник и розетку, если проводка уже разведена"],
        104: ["смеситель поменять могу", "подключу стиралку к готовым выводам"],
        103: ["затирку швов обновлю", "пару плиток заменить могу"],
        102: ["мелкие дыры заштукатурю", "откос подправить могу"],
        106: ["подкрасить стену могу", "одну комнату обоями оклею"]}
PART_NEG = {105: "новую проводку не тяну", 104: "разводку труб не делаю", 103: "целую ванную плиткой не кладу",
            102: "стены целиком не штукатурю", 106: "большие объёмы малярки не беру"}

HEAD = ["Делаю ремонт квартир.", "Ремонт квартир.", "Ремонт под ключ.", "Мастер по ремонту.", "Ремонт ванной и кухни.",
        "Частный мастер.", "Бригада отделочников.", "Отделка квартир.", "", "Здравствуйте! Делаю ремонт.", "Ремонт квартир, опыт {e} лет."]
TAIL = ["Опыт {e} лет.", "Звоните.", "Цена договорная.", "Работаю по договору.", "", "Гарантия на работы.", "Пишите в чат.", "Выезд бесплатно."]

def one(c): return rnd.choice(SV[c])
def listing(cs):  # перечисление услуг
    style = rnd.random()
    if style < .4: return ", ".join(one(c) for c in cs)
    if style < .7: return ", ".join(W[c]["nom"] for c in cs)
    return ". ".join(cap(one(c)) for c in cs)
def names(cs, case): return " и ".join(W[c][case] for c in cs) if len(cs) <= 2 else ", ".join(W[c][case] for c in cs[:-1]) + " и " + W[cs[-1]][case]
def cap(s): return s[:1].upper() + s[1:]

# --- шаблоны настоящих отрицаний: (текст, нужно ли перечисление offered отдельно) ---
NEG_T = [
    "{off}, {neg_acc} не делаю.",
    "{off}. {Neg_acc} не делаю.",
    "{off}. {Neg_acc} — нет, не моё, вызывайте профильных.",
    "{off}. Всё кроме {neg_gen} — {neg_nom} за заказчиком.",
    "Только {off_nom}, за {neg_acc} не берусь даже за доплату.",
    "{off}. Без {neg_gen} — чисто своё направление.",
    "{off}. Ни {neg_ins_ni} не занимаюсь — не имею допуска.",
    "Нужна {neg_nom}? Это не ко мне. Нужна {off_nom} — звоните.",
    "{off}. {Neg_nom} в стоимость не входит и нами не выполняется.",
    "{off}. Раньше делал и {neg_acc}, но больше не беру.",
    "{off}. За {neg_acc} не берусь, {neg_acc2}.",
    "{off}. {Neg_acc} не делаем, не умеем и не хотим учиться. Зато {off_one} отлично.",
    "{off_prof}. Если бы ещё и {neg_acc} делал, был бы универсал, но нет: только {off_nom}.",
    "{off}. {Neg_acc} не предлагаю, обращайтесь к {neg_pro_dat}.",
    "{off}. По {neg_dat} не работаю.",
    "{off}, остальное ({neg_nom}) не делаю.",
    "{off}. {Neg_nom} — не ко мне.",
    "{off}. Из того, что НЕ делаю: {neg_nom}.",
    "{off}. {Neg_acc} не беру, можно даже не спрашивать.",
    "{off}. К сожалению, {neg_acc} не делаю — нет инструмента.",
    "{off}; {neg_acc} исключаю.",
    "{off}. Без {neg_gen}!",
    "{off_prof} — не {neg_prof}: {neg_acc} не делаю.",
    "{off}. {Neg_nom} не входит, это к другим мастерам.",
]
# --- ложные отрицания: услуга предлагается, несмотря на «не» ---
POS_T = [
    "Не только {a_acc}, но и {b_acc} — всё своими руками.",
    "{A_nom} не дорого и не долго. Делаю не хуже фирм, но дешевле.",
    "{A_acc} не делаю? Делаю! И {b_acc} тоже.",
    "Нет такого, чего я не делаю: {all_nom}.",
    "Не нужно искать двух мастеров — {a_one} и {b_one} сам.",
    "Не бойтесь доверить мне {a_acc}: опыт {e} лет.",
    "{A_nom} — не проблема, {b_nom} тоже.",
    "Не поверите, но {a_acc} и {b_acc} делаю один, без бригады.",
    "Никогда не отказываюсь от {a_gen} и {b_gen}.",
    "Не затягиваю сроки: {a_one}, {b_one}.",
    "Делаю {a_acc}, не переплачивайте фирмам.",
    "{A_nom}, {b_nom} — не беру предоплату, оплата по факту.",
    "Не умею делать плохо: {a_one}, {b_one}.",
    "Не первый год {a_one} и {b_one}.",
    "{A_nom} без переделок, {b_nom} без задержек.",
]
# --- смешанные: общее отрицание категории, но частная работа предлагается ---
MIX_T = [
    "{off}. {Neg_acc} не делаю, но {part}.",
    "{off}. {Neg_acc} делаю только по мелочи: {part}. {part_neg}.",
    "{off}. С {neg_ins} сложно: {part_neg_l}, но {part}.",
]

ads = []; seen = set()
def add(desc, cats, pol, conf, note, extra):
    key = re.sub(r"\W+", " ", desc.lower())
    if key in seen or len(desc) < 20: return False
    seen.add(key)
    ads.append({"item_id": f"neg-{len(ads)+1:04d}", "title": rnd.choice(["Ремонт квартир", "Мастер по ремонту", "Отделка",
                "Ремонт ванной", cap(names(cats, 'nom')) if cats else "Ремонт под ключ"]), "description": desc,
                "found_microcategories": [{"mcId": c, "label": NAMES[c]} for c in sorted(cats)], "mcId": 101,
                "type": "NEG2", "confidence": conf, "polarity": pol, "note": note, **extra})
    return True

def wrap(body, e):
    h = rnd.choice(HEAD).format(e=e); t = rnd.choice(TAIL).format(e=e)
    return " ".join(x for x in (h, body, t) if x).replace("С ш", "Со ш").replace("с ш", "со ш")

quota = {"negated": int(N * .6), "positive": int(N * .25)}
quota["mixed"] = N - sum(quota.values())
made = {k: 0 for k in quota}
tries = 0
while any(made[k] < quota[k] for k in quota) and tries < N * 50:
    tries += 1
    e = rnd.randint(3, 25)
    pol = rnd.choice([k for k in quota if made[k] < quota[k]])
    if pol == "negated":
        k_off = rnd.choice([1, 1, 2, 2, 3]); k_neg = rnd.choice([1, 1, 1, 2, 2, 3])
        cs = rnd.sample(C, min(5, k_off + k_neg)); off, neg = cs[:k_off], cs[k_off:]
        if not neg: continue
        t = rnd.choice(NEG_T)
        n1 = neg[0]
        body = t.format(off=cap(listing(off)), off_nom=names(off, "nom"), off_one=one(off[0]), off_prof=cap(W[off[0]]["prof"]),
                        neg_acc=names(neg, "acc"), Neg_acc=cap(names(neg, "acc")), neg_gen=names(neg, "gen"), neg_nom=names(neg, "nom"),
                        Neg_nom=cap(names(neg, "nom")), neg_dat=names(neg, "dat"), neg_ins=names(neg, "ins"),
                        neg_ins_ni=", ни ".join(W[c]["ins"] for c in neg), neg_pro_dat=W[n1]["pro"].replace("а", "у", 1) if n1 != 106 else "маляру",
                        neg_prof=W[n1]["prof"], neg_acc2=(W[neg[1]]["acc"] + " не трогаю") if len(neg) > 1 else "это к другим")
        ok = add(wrap(body, e), off, "negated", "high", f"отрицание {neg}, предложено {off}", {"offered": off, "denied": neg})
    elif pol == "positive":
        a, b = rnd.sample(C, 2); t = rnd.choice(POS_T)
        allc = rnd.sample(C, rnd.randint(3, 5))
        cats = allc if "{all_nom}" in t else ([a, b] if "{b_" in t else [a])
        body = t.format(a_acc=W[a]["acc"], A_acc=cap(W[a]["acc"]), b_acc=W[b]["acc"], a_nom=W[a]["nom"], A_nom=cap(W[a]["nom"]), b_nom=W[b]["nom"],
                        a_gen=W[a]["gen"], b_gen=W[b]["gen"], a_one=one(a), b_one=one(b), all_nom=", ".join(W[c]["nom"] for c in allc), e=e)
        ok = add(wrap(body, e), cats, "positive", "high", "ЛОЖНОЕ отрицание: «не» есть, услуги предлагаются", {"offered": cats, "denied": []})
    else:
        cs = rnd.sample(C, rnd.choice([2, 3])); off, n1 = cs[:-1], cs[-1]
        t = rnd.choice(MIX_T)
        body = t.format(off=cap(listing(off)), Neg_acc=cap(W[n1]["acc"]), neg_ins=W[n1]["ins"], part=rnd.choice(PART[n1]), part_neg=cap(PART_NEG[n1]), part_neg_l=PART_NEG[n1])
        ok = add(wrap(body, e), off + [n1], "mixed", "medium",
                 f"СМЕШАННЫЙ: общее отрицание {n1}, но частная работа предлагается -> {n1} ставим; спорно", {"offered": off, "denied": [], "partial": [n1]})
    if ok: made[pol] += 1

with open(sys.argv[1], "w", encoding="utf-8") as f:
    for a in ads: f.write(json.dumps(a, ensure_ascii=False) + "\n")
print(made, len(ads))
