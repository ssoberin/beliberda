"""Explicit truth-valued scenarios. Russian negative concord is NOT parity."""
from itertools import permutations
import re

# split, family, positive A statement, negative A statement, semantic nesting depth
NESTED = [
    ("train", "denial_of_denial", "Неверно, что мы не выполняем {a}.", "Верно, что мы не выполняем {a}.", 2),
    ("train", "quote_truth", "Фраза «мы не выполняем {a}» не соответствует действительности.", "Фраза «мы не выполняем {a}» соответствует действительности.", 2),
    ("train", "no_refusal", "Мы не отказываемся выполнять {a} и принимаем такие заказы.", "Мы отказываемся выполнять {a} и не принимаем такие заказы.", 2),
    ("train", "concord", "Мы никому не отказываем в услуге: выполняем {a}.", "Мы ни для кого, ни при каких обстоятельствах не выполняем {a}.", 2),
    ("train", "triple_claim", "Неверно, что мы не подтверждаем предложение выполнять {a}: предложение подтверждаем.", "Неверно, что мы подтверждаем предложение выполнять {a}: такую работу не берём.", 3),
    ("train", "refutation_refuted", "Опровержение фразы «мы не выполняем {a}» верно.", "Опровержение фразы «мы не выполняем {a}» неверно.", 3),
    ("train", "not_excluded", "Не исключаем из своих услуг {a}: заказы принимаем.", "Не исключено, что когда-нибудь предложим {a}, но сейчас такие заказы не принимаем.", 2),
    ("train", "not_only_nested", "Неверно, что мы не только не выполняем {a}, но и не принимаем на неё заказы. Эту работу выполняем.", "Верно, что мы не только не выполняем {a}, но и не принимаем на неё заказы.", 3),
    ("train", "not_all_negative", "Не все работы исключены из нашего списка: {a} выполняем.", "Не все работы включены в наш список: {a} не выполняем.", 2),
    ("train", "four_layer_facts", "Неверно, что опровержение утверждения «мы не выполняем {a}» не соответствует фактам.", "Неверно, что опровержение утверждения «мы не выполняем {a}» соответствует фактам.", 4),
    ("validation", "false_report", "Сообщение о том, что мы больше не выполняем {a}, ошибочно.", "Сообщение о том, что мы больше не выполняем {a}, не ошибочно.", 3),
    ("validation", "not_denied", "Не отрицаем, что выполняем {a}: это наша действующая услуга.", "Не отрицаем, что не выполняем {a}: это не наша услуга.", 3),
    ("validation", "not_unavailable", "Нельзя считать недоступной нашу услугу: {a} выполняем по записи.", "Нельзя считать доступной нашу услугу: {a} больше не выполняем.", 3),
    ("test", "four_layers", "Неверно, что опровержение фразы «мы не выполняем {a}» неверно.", "Неверно, что опровержение фразы «мы не выполняем {a}» верно.", 4),
    ("test", "double_report", "Не соответствует правде сообщение, будто неверно, что мы выполняем {a}.", "Не соответствует правде сообщение, будто неверно, что мы не выполняем {a}.", 4),
    ("test", "concord_never", "Никто не может сказать, что мы не предлагаем {a}: мы её выполняем.", "Никто не может сказать, что мы предлагаем {a}: мы её не выполняем.", 3),
]

# Transforms change spelling/inflection, never delete or invert logical operators.
SPELLING = {
    "выполняем": "выпалняем", "предлагаем": "предлогаем", "штукатурку": "штукотурку",
    "механизированную": "механезированую", "керамогранита": "керамогронита",
    "мозаики": "мозайки", "раковин": "ракавен", "унитазов": "унетазов",
    "водоснабжения": "водоснабженья", "канализации": "канолизации",
    "смесителей": "смисителей", "электропроводки": "електропроводки",
    "розеток": "разеток", "выключателей": "выключатилей", "электрощита": "електрощита",
    "поклейку": "паклейку", "покраску": "пакраску", "шпаклёвку": "шпаклевку",
    "комплексный": "камплексный", "квартиры": "квортиры", "ремонта": "римонта",
    "ремонт": "римонт", "заказы": "заказзы", "подтверждаем": "потверждаем",
}
GRAMMAR = {
    "выполняем": "выполняет", "принимаем": "принимаит", "предлагаем": "предлагает",
    "подтверждаем": "подтверждает", "отказываемся": "отказываюсь", "выполнять": "выполнят",
    "штукатурку": "штукатурка", "укладку": "укладка", "установку": "установка",
    "разводку": "разводка", "замену": "замена", "сборку": "сборка",
    "поклейку": "поклейка", "покраску": "покраска", "шпаклёвку": "шпаклёвка",
    "облицовку": "облицовка", "выровняем": "выровним",
}
PROTECTED = re.compile(r"\b(?:не|ни|нет|только|кроме|без|неверно|верно|никто|никому|никогда)\b", re.I)


def corrupt(text, mode):
    table = SPELLING if mode == "spelling" else GRAMMAR
    def replace(match):
        word = match.group()
        result = table.get(word.lower(), word)
        return result[:1].upper() + result[1:] if word[0].isupper() else result
    changed = re.sub(r"[А-Яа-яЁё]+", replace, text)
    if mode == "grammar_spacing":
        changed = changed.replace(",", "").replace(": ", ":").lower()
        changed = changed.replace(". ", ".  ")
    assert [x.lower() for x in PROTECTED.findall(changed)] == [x.lower() for x in PROTECTED.findall(text)]
    return changed


# No inference from mentions: each entry has an explicit scope/status per service.
LIST_STYLES = [
    ("train", "status_offer", "Условия заказа по каждой позиции:", "Предлагаем {s}.", "Не предлагаем {s}."),
    ("train", "status_accept", "Наш действующий перечень:", "Заказы на {s} принимаем.", "Заказы на {s} не принимаем, эту работу не выполняем."),
    ("train", "status_scope", "Список услуг с ограничениями:", "Выполняем {s}, но не бесплатно.", "Не выполняем {s} даже за доплату."),
    ("validation", "status_delegation", "Кто отвечает за работы:", "Выполняем {s} своими силами.", "Заказчик заказывает {s} у другой бригады; мы эту работу не предлагаем."),
    ("test", "status_confirmation", "Ответы на вопросы об услугах:", "Выполняем ли {s}? Да, выполняем.", "Выполняем ли {s}? Нет, не выполняем."),
    ("test", "status_negated_denial", "Уточнение информации о работах:", "Неверно, что мы не выполняем {s}.", "Верно, что мы не выполняем {s}."),
]

# Deliberately malformed particles are authored separately: blindly replacing
# НЕ with НИ (or deleting it) would make label preservation unsafe.
COLLOQUIAL = [
    ("train", 101, "я плитку неделаю а вот разетки меняю", [105], "я разетки неделаю а вот плитку ложу", [103]),
    ("train", 101, "не тока абои клею но и сантехнику делаю трубы меняю", [104, 106], "тока абои клею сантехнику неделаю трубы не меняю", [106]),
    ("train", 101, "штукатурку небирём зато шпаклюем под пакраску", [106], "шпаклёвку небирём зато штукатурим по маякам", [102]),
    ("train", 101, "я ни делаю електрику отказываюсь от таких заказов тока крашу стены", [106], "я делаю електрику меняю провада а красить стены отказываюсь", [105]),
    ("train", 101, "не не трубы я меняю а плитку не ложу", [104], "не не плитку я ложу а трубы не меняю", [103]),
    ("train", 101, "мы ни разу не отказывали по укладке кафеля берём такие заказы а обои неклеим", [103], "мы отказываем по укладке кафеля неберём такие заказы а обои клеим", [106]),
    ("train", 105, "квартиру целиком пад ключ делаем от начала до здачи а проводку сами не меняем", [101], "квартиру целиком пад ключ неделаем тока проводку меняем", [105]),
    ("train", 101, "всё не так написали мы НЕ отказываемся штукатурить стены но красить не будем", [102], "всё не так написали мы отказываемся штукатурить стены но красить будем", [106]),
    ("validation", 101, "сантехнику неберу кроме замены смесителя это могу остальное нет", [104], "сантехнику неберу даже смеситель менять не буду тока шкафы собираю", []),
    ("validation", 101, "электрику нивкоем случае не делаем но абои поклеим", [106], "абои нивкоем случае не клеим но электрику делаем проводку меняем", [105]),
    ("test", 101, "ниправда што я не ложу плитку ложу а сантехника не ко мне", [103], "правда што я не ложу плитку неберусь а сантехнику делаю трубы меняю", [104]),
    ("test", 101, "нехачу сказать што не крашу крашу стены а разетки неставлю", [106], "хачу сказать што не крашу стены неберусь а разетки ставлю", [105]),
]


def expanded_rows(names, services, make_row):
    result = {k: [] for k in ("train", "validation", "test")}

    def emit(split, family, group, texts, offered, source, note, depth, topic):
        for variant, (text, cats) in enumerate(zip(texts, offered)):
            for mode in ("clean", "spelling", "grammar_spacing"):
                desc = text if mode == "clean" else corrupt(text, mode)
                row = make_row(f"{group}-{variant}-{mode}", group, family, desc, cats, source, note, variant)
                row.update(augmentation=mode, clean_item_id=f"{group}-{variant}-clean",
                           negation_depth=depth, topic=topic, semantic_version="v2")
                result[split].append(row)

    for split, family, positive, negative, depth in NESTED:
        for ca, cb in permutations(names, 2):
            for lex in range(3):
                a, b = services[ca][lex], services[cb][(lex + 1) % 3]
                group = f"v2-{family}-{ca}-{cb}-{lex}"
                source = sorted(set(names) - {ca})[(ca + cb + lex) % 5]
                tail = f" Отдельно выполняем {b}."
                texts = [positive.format(a=a) + tail, negative.format(a=a) + tail]
                emit(split, family, group, texts, [{ca, cb}, {cb}], source,
                     "Первый вариант утверждает A, второй отрицает A. B предлагается в обоих. Вложенное отрицание высказывания отличается от русского отрицательного согласования.", depth, "nested_negation")

    for split, family, intro, positive, negative in LIST_STYLES:
        for mask in range(64):
            # Use only one orientation of each pair (flip the highest category).
            # 32 masks cover all 64 subsets without duplicate reverse pairs.
            if mask & 32:
                continue
            for lex in range(3):
                cats = {c for i, c in enumerate(names) if mask & (1 << i)}
                other = cats | {106}
                texts = []
                for offered in (cats, other):
                    lines = [(positive if c in offered else negative).format(s=services[c][lex]) for c in names]
                    texts.append(intro + "\n" + "\n".join(lines))
                source = 101 + (mask + lex) % 5
                emit(split, family, f"v2-{family}-{mask:02d}-{lex}", texts,
                     [cats, other], source, "Каждая из шести услуг имеет отдельный явный статус. Отрицание одной позиции не распространяется на другие.",
                     2 if family == "status_negated_denial" else 1, "multicategory")
    for i, (split, source, positive, pos_ids, negative, neg_ids) in enumerate(COLLOQUIAL):
        family = f"colloquial-{i:02d}"
        for variant, (text, offered) in enumerate(((positive, pos_ids), (negative, neg_ids))):
            row = make_row(f"v2-{family}-{variant}", f"v2-{family}", family, text,
                           offered, source, "Разговорная фраза и ошибочное написание частицы размечены целиком по смыслу; это не случайное удаление НЕ.", variant)
            row.update(topic="colloquial_particles", augmentation="authored_errors", negation_depth=2, semantic_version="v2")
            result[split].append(row)
    return result
