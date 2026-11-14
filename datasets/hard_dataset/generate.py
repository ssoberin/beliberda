"""Deterministic semantic contrast data. Standard library only; no external APIs."""
import argparse
from collections import Counter
from itertools import permutations
import json
from pathlib import Path
import random
from challenge_cases import CASES

NAMES = {
    101: "Ремонт под ключ", 102: "Штукатурные работы", 103: "Плиточные работы",
    104: "Сантехника", 105: "Электрика", 106: "Поклейка обоев и малярные работы",
}
# Accusative objects: suitable after «выполняем», «предлагаем», «заказывают».
SERVICES = {
    101: ["комплексный ремонт квартиры под ключ", "полный цикл ремонта дома со сдачей готового результата", "комплексный ремонт санузла под ключ"],
    102: ["штукатурку стен по маякам", "механизированную штукатурку стен", "выравнивание стен штукатурными составами"],
    103: ["укладку керамогранита", "облицовку стен плиткой", "укладку мозаики"],
    104: ["установку раковин и унитазов", "разводку водоснабжения и канализации", "замену смесителей и сифонов"],
    105: ["замену электропроводки", "установку розеток и выключателей", "сборку электрощита с автоматами и УЗО"],
    106: ["поклейку обоев", "покраску стен и потолков", "шпаклёвку стен под покраску"],
}
# Each family contains two separately authored meanings. A/B are semantic roles,
# not keywords inferred from the generated text. Scope changes the gold labels.
# (id, explanation, [(text, offered roles), ...])
FAMILIES = [
    ("not_only", "«Не только A, но и B» утверждает обе услуги; «только A» ограничивает перечень.", [
        ("Выполняем только {a}. {B} не выполняем.", "A"),
        ("Выполняем не только {a}, но и {b}.", "AB")]),
    ("neither", "Общее отрицание распространяется на оба однородных дополнения.", [
        ("Не выполняем ни {a}, ни {b}. Занимаемся только сборкой мебели.", ""),
        ("Выполняем и {a}, и {b}. Сборкой мебели не занимаемся.", "AB")]),
    ("instead", "Противопоставление меняет выбранную услугу.", [
        ("Предлагаем не {a}, а {b}.", "B"),
        ("Предлагаем {a}, а не {b}.", "A")]),
    ("except", "Исключение удаляет только названную услугу из явного списка.", [
        ("В списке обсуждались две услуги: {a} и {b}. Выполняем обе, кроме первой.", "B"),
        ("В списке обсуждались две услуги: {a} и {b}. Выполняем обе, кроме второй.", "A")]),
    ("only_together", "Запрет отдельного заказа не отрицает услугу в составе пакета.", [
        ("Выполняем {a}; {b} делаем только вместе с первой услугой, отдельно не берём.", "AB"),
        ("Выполняем {a}; {b} не делаем ни отдельно, ни вместе с первой услугой.", "A")]),
    ("not_yet", "Намерение на будущее не является текущим предложением.", [
        ("Сейчас выполняем {a}. Планируем добавить {b}, но пока заказы на эту работу не принимаем.", "A"),
        ("Сейчас выполняем {a}. Уже добавили {b} и принимаем заказы на эту работу.", "AB")]),
    ("no_longer", "Прошлая деятельность отличается от действующего предложения.", [
        ("Раньше выполняли {a}, теперь эту услугу не оказываем. Сейчас предлагаем {b}.", "B"),
        ("Раньше выполняли {a} и продолжаем принимать такие заказы. Сейчас также предлагаем {b}.", "AB")]),
    ("third_party", "Рекомендация другого исполнителя не становится своей услугой.", [
        ("Выполняем {a}. На {b} можем дать телефон другой бригады; сами эту работу не берём.", "A"),
        ("Выполняем {a}. На {b} приезжает наш мастер; эту работу берём на себя по нашему договору.", "AB")]),
    ("customer", "Следует различать действия заказчика и исполнителя.", [
        ("Выполняем {a}. Заказчик делает {b} своими силами, мы эту работу не предлагаем.", "A"),
        ("Выполняем {a}. Заказчик не делает {b} своими силами: эту работу предлагаем и выполняем мы.", "AB")]),
    ("quoted_request", "Пожелание в цитате не доказывает согласие исполнителя.", [
        ("Клиент написал: «Нужно выполнить {a} и {b}». Мы согласились только на первую работу; вторую не выполняем.", "A"),
        ("Клиент написал: «Нужно выполнить {a} и {b}». Мы согласились на обе работы и предлагаем их новым клиентам.", "AB")]),
    ("question_answer", "Вопрос о работе интерпретируется вместе с ответом.", [
        ("Вы выполняете {a}? Да. А {b}? Нет, такую услугу не оказываем.", "A"),
        ("Вы выполняете {a}? Да. А {b}? Тоже да, принимаем заказы.", "AB")]),
    ("double_negative", "«Не отказываемся» утверждает готовность, а «не берёмся» её отрицает.", [
        ("Выполняем {a}. От заказов на {b} не отказываемся: эту работу тоже делаем.", "AB"),
        ("Выполняем {a}. За {b} не берёмся: эту работу не делаем.", "A")]),
    ("not_required", "Отсутствие обязанности заказывать B не отменяет доступность B.", [
        ("Выполняем {a}. Заказывать {b} необязательно, но по желанию клиента мы эту услугу оказываем.", "AB"),
        ("Выполняем {a}. Заказывать у нас {b} невозможно: такой услуги у нас нет.", "A")]),
    ("condition", "Условие выполнения действующей услуги отличается от неподтверждённой возможности.", [
        ("Выполняем {a}. Если понадобится, выполним и {b}: эта услуга есть в нашем прайсе.", "AB"),
        ("Выполняем {a}. Если бы был специалист, могли бы выполнять {b}, но сейчас такой услуги нет.", "A")]),
    ("location_scope", "Отрицание географии не равно отрицанию самой услуги.", [
        ("Выполняем {a}. Предлагаем также {b}, но только в городе; за городом эту работу не выполняем.", "AB"),
        ("Выполняем {a}. Не предлагаем {b} ни в городе, ни за городом.", "A")]),
    ("time_scope", "Запрет на конкретное время не отменяет услугу в другое время.", [
        ("Выполняем {a}. По выходным не выполняем {b}, а по будням принимаем и выполняем такие заказы.", "AB"),
        ("Выполняем {a}. Не выполняем {b} ни по выходным, ни по будням.", "A")]),
    ("price_scope", "Отрицание бесплатности не является отказом от услуги.", [
        ("Выполняем {a}. Бесплатно {b} не выполняем; за оплату по смете эту работу делаем.", "AB"),
        ("Выполняем {a}. Ни бесплатно, ни за оплату {b} не выполняем.", "A")]),
    ("correction", "Явное исправление старого списка имеет приоритет.", [
        ("В старой версии объявления предлагали {a} и {b}. Исправление: первую услугу убрали, сейчас выполняем только вторую.", "B"),
        ("В старой версии объявления предлагали {a} и {b}. Исправление: вторую услугу убрали, сейчас выполняем только первую.", "A")]),
    ("anaphora", "Указание «это» относится к предыдущему упоминанию.", [
        ("Часто спрашивают про {a}. Это мы не делаем. Зато выполняем {b}.", "B"),
        ("Часто спрашивают про {a}. Это мы делаем. А вот {b} не выполняем.", "A")]),
    ("exception_to_denial", "«Кроме» может задавать исключение из отрицательного списка.", [
        ("Из двух работ — {a} и {b} — не выполняем ничего, кроме первой.", "A"),
        ("Из двух работ — {a} и {b} — не выполняем ничего, кроме второй.", "B")]),
    ("without_addon", "Ограничение комплектации отделяет услугу от дополнения.", [
        ("Можно заказать {a} без второй услуги. Вторую услугу — {b} — мы также выполняем по желанию клиента.", "AB"),
        ("Можно заказать {a} без второй услуги. Вторую услугу — {b} — мы вообще не выполняем.", "A")]),
    ("not_confuse", "Упоминание для сравнения не является предложением.", [
        ("Выполняем {a}. Не путайте это с другой услугой: {b} мы не предлагаем.", "A"),
        ("Выполняем {a}. Не путайте это с другой услугой: {b} мы тоже предлагаем, но считаем отдельной позицией.", "AB")]),
    ("seo", "Перечень поисковых запросов без предложения работ не даёт меток.", [
        ("Предлагаем {a}. Поисковые запросы: {b}. Последняя фраза — только тег, эту работу не выполняем.", "A"),
        ("Предлагаем {a}. Поисковые запросы: {b}. Последняя фраза — не только тег, эту работу тоже выполняем.", "AB")]),
    ("ne_tolko_list", "Ограничительная частица и её отрицание меняют объём предложения.", [
        ("Наши услуги ограничиваются одной позицией: выполняем лишь {a}. На {b} заказы не принимаем.", "A"),
        ("Наши услуги не ограничиваются одной позицией: выполняем как {a}, так и {b}.", "AB")]),
    ("both_refused", "Отрицание сказуемого относится ко всему перечислению.", [
        ("Оказываем две услуги: выполняем {a} и {b}.", "AB"),
        ("Не оказываем эти две услуги: не выполняем {a} и {b}. Только собираем мебель.", "")]),
    ("rumour", "Отрицание чужого утверждения об отказе восстанавливает предложение.", [
        ("Говорят, что мы не выполняем {a}. Это неверно: выполняем. А {b} не предлагаем.", "A"),
        ("Говорят, что мы не выполняем {a}. Это верно: не выполняем. А {b} предлагаем.", "B")]),
    ("unless", "Условное ограничение не означает полного отказа.", [
        ("Выполняем {a}. За {b} не берёмся, если нет доступа в помещение; при наличии доступа работу выполняем.", "AB"),
        ("Выполняем {a}. За {b} не берёмся даже при наличии доступа в помещение.", "A")]),
    ("list_status", "Статус позиций важнее наличия их названий в списке.", [
        ("Список работ:\n— {a}: принимаем заказы;\n— {b}: не принимаем заказы, услугу сняли.", "A"),
        ("Список работ:\n— {a}: не принимаем заказы, услугу сняли;\n— {b}: принимаем заказы.", "B")]),
    ("long_distance", "Область отрицания может определяться после длинного перечисления.", [
        ("Обсуждали {a} и {b}. Сначала уточняем объём, затем согласуем доступ и порядок работ. Это порядок переговоров, а не подтверждение всех услуг. Из названных работ берём только первую, вторую не предлагаем.", "A"),
        ("Обсуждали {a} и {b}. Сначала уточняем объём, затем согласуем доступ и порядок работ. Это порядок переговоров, а не подтверждение всех услуг. Из названных работ берём только вторую, первую не предлагаем.", "B")]),
    ("self_vs_team", "Отрицание личного участия не исключает услугу собственной бригады.", [
        ("Лично я не выполняю {a}, но её выполняет моя бригада по нашему договору. Также берём {b}.", "AB"),
        ("Лично я не выполняю {a}, и моя бригада её тоже не выполняет. Берём только {b}.", "B")]),
]

VALIDATION = {"exception_to_denial", "not_required", "location_scope", "quoted_request"}
TEST = {"rumour", "list_status", "unless", "long_distance"}


def labels(ids):
    return [{"mcId": i, "label": NAMES[i]} for i in sorted(ids)]


def make_row(item_id, group, family, description, offered, source, note, variant):
    return {
        "item_id": item_id, "title": "Услуги мастера", "description": description,
        "mcId": source, "found_microcategories": labels(set(offered) - {source}),
        "offered_microcategories": sorted(offered), "type": family,
        "group_id": group, "variant": variant, "confidence": "high",
        "provenance": "synthetic_rule_authored", "note": note,
    }


def generate(seed):
    rng = random.Random(seed)
    splits = {name: [] for name in ("train", "validation", "test")}
    for family, note, variants in FAMILIES:
        split = "validation" if family in VALIDATION else "test" if family in TEST else "train"
        for index, (ca, cb) in enumerate(permutations(NAMES, 2)):
            a, b = SERVICES[ca][index % 3], SERVICES[cb][(index // 3) % 3]
            role_ids = {"A": ca, "B": cb}
            sets = [{role_ids[x] for x in roles} for _, roles in variants]
            # Keep changed labels observable after excluding the source category.
            changed = sets[0] ^ sets[1]
            sources = sorted(set(NAMES) - changed)
            source = 101 if 101 in sources and index % 2 == 0 else rng.choice(sources)
            group = f"hard-{family}-{ca}-{cb}"
            for v, ((template, _), offered) in enumerate(zip(variants, sets)):
                desc = template.format(a=a, b=b, B=b[0].upper() + b[1:])
                splits[split].append(make_row(f"{group}-{v}", group, family, desc, offered, source, note, v))
    for rows in splits.values():
        rng.shuffle(rows)
    splits["challenge"] = []
    for i, (source, first, first_ids, second, second_ids, note) in enumerate(CASES, 1):
        group = f"challenge-{i:03d}"
        for v, (text, offered) in enumerate(((first, first_ids), (second, second_ids))):
            splits["challenge"].append(make_row(
                f"{group}-{v}", group, group, text, offered, source, note, v))
    return splits


def validate(splits):
    seen_ids, seen_texts, seen_groups, seen_families = set(), set(), {}, {}
    stats = {}
    for split, rows in splits.items():
        pairs = {}
        for r in rows:
            assert r["item_id"] not in seen_ids, r["item_id"]
            seen_ids.add(r["item_id"])
            # Description-only deduplication matches the service's actual input.
            key = " ".join(r["description"].casefold().split())
            assert key not in seen_texts, r["item_id"]
            seen_texts.add(key)
            assert 1 <= len(r["description"]) <= 12000
            assert r["mcId"] in NAMES
            assert set(r["offered_microcategories"]) <= NAMES.keys()
            assert r["found_microcategories"] == labels(set(r["offered_microcategories"]) - {r["mcId"]})
            for value, owners in ((r["group_id"], seen_groups), (r["type"], seen_families)):
                assert owners.setdefault(value, split) == split, f"Split leakage: {value}"
            pairs.setdefault(r["group_id"], []).append(r)
        for pair in pairs.values():
            assert len(pair) == 2
            assert pair[0]["mcId"] == pair[1]["mcId"]
            assert pair[0]["found_microcategories"] != pair[1]["found_microcategories"]
        stats[split] = {
            "rows": len(rows), "contrast_pairs": len(pairs),
            "empty_targets": sum(not r["found_microcategories"] for r in rows),
            "types": dict(sorted(Counter(r["type"] for r in rows).items())),
            "source_counts": dict(sorted(Counter(r["mcId"] for r in rows).items())),
            "target_counts": dict(sorted(Counter(m["mcId"] for r in rows for m in r["found_microcategories"]).items())),
        }
        assert set(stats[split]["target_counts"]) == set(NAMES)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    splits = generate(args.seed)
    stats = validate(splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"seed": args.seed, "total": sum(len(r) for r in splits.values()), "splits": stats}
    (args.output_dir / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total": report["total"], **{k: v["rows"] for k, v in stats.items()}}))


if __name__ == "__main__":
    main()
