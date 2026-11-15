import re
import unittest
from collections import Counter
from pathlib import Path

from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.qwen3_category_aware_experiment import (
    ALL_IDS,
    chunk_ad,
    load_public_rows,
    load_v3_rows_full,
)


def _word_counter(text: str) -> Counter:
    return Counter(re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower()))


class CategoryAwareChunkerTests(unittest.TestCase):
    def test_preserves_question_and_short_answer_together(self):
        text = "Делаем ли мы сантехнику?\nДа.\n\nВыполняем ли электромонтаж?\nНет, электрику не берём."
        for fn in (chunk_ad, chunk_ad_fixed):
            chunks = fn(text)
            self.assertEqual(len(chunks), 2)
            self.assertIn("Да.", chunks[0])
            self.assertIn("Нет, электрику не берём.", chunks[1])

    def test_keeps_heading_attached_across_blank_line(self):
        text = "Что НЕ делаем:\n\n— Штукатурка стен\n— Поклейка обоев"
        for fn in (chunk_ad, chunk_ad_fixed):
            chunks = fn(text)
            self.assertEqual(len(chunks), 2)
            self.assertTrue(all("Что НЕ делаем:" in ch for ch in chunks))

    def test_keeps_opposing_bullets_separated(self):
        text = "• Больше не берём заявки на штукатурное выравнивание.\n• Владелец сам не выполняет полный ремонт дома, но его штатные мастера делают."
        for fn in (chunk_ad, chunk_ad_fixed):
            chunks = fn(text)
            self.assertEqual(len(chunks), 2)
            self.assertIn("штукатурное", chunks[0])
            self.assertNotIn("штатные мастера", chunks[0])

    def test_does_not_truncate_long_ads(self):
        parts = [f"Абзац {i}: выполняем работы по ремонту объекта номер {i} с проверкой качества." for i in range(120)]
        full = "\n\n".join(parts)
        self.assertGreater(len(full), 7000)
        for fn in (chunk_ad, chunk_ad_fixed):
            chunks = fn(full)
            self.assertTrue(any("номер 119" in ch for ch in chunks))

    def test_public_never_fabricates_101_and_v3_has_101(self):
        pub = load_public_rows(Path("output/ngram-comparison/splits/validation.jsonl"))
        self.assertTrue(all(not r["has_101_label"] and 101 not in r["labels"] for r in pub[:20]))
        v3_path = Path("datasets/own_v3_clean/validation.jsonl")
        if v3_path.exists():
            v3 = load_v3_rows_full(v3_path)
            self.assertTrue(all(r["has_101_label"] for r in v3[:20]))
            self.assertTrue(any(101 in r["evaluation_labels"] for r in v3))

    def test_fixed_preserves_pub_val_654_short_lines_and_hashtags(self):
        pub = load_public_rows(Path("output/ngram-comparison/splits/validation.jsonl"))
        text = pub[654]["description"]
        chunks = chunk_ad_fixed(text)
        joined = "\n".join(chunks)
        for expected in ("Шпаклёвка", "Обои", "Краска", "Сантехника", "Электрика", "Плитка", "#ремонт", "#шпаклёвкастен"):
            self.assertIn(expected, joined)
        self.assertGreaterEqual(len(chunks), 10)

    def test_fixed_preserves_consecutive_headings(self):
        text = "НАШИ УСЛУГИ:\nВнутренняя отделка:\n— Штукатурка стен\n— Шпаклёвка и покраска"
        chunks = chunk_ad_fixed(text)
        joined = "\n".join(chunks)
        self.assertIn("НАШИ УСЛУГИ:", joined)
        self.assertIn("Внутренняя отделка:", joined)
        self.assertIn("Штукатурка стен", joined)
        self.assertIn("Шпаклёвка и покраска", joined)

    def test_fixed_preserves_extended_bullets_and_unbulleted_lists(self):
        text = (
            "В отделку входит:\n"
            "➤ Выравнивание стен\n"
            "◾ Укладка плитки\n"
            "– Монтаж розеток\n"
            "🔹БезПробела сантехника\n\n"
            "Дополнительно:\n"
            "Штукатурка стен\n"
            "Поклейка обоев\n"
            "Установка сантехники"
        )
        chunks = chunk_ad_fixed(text)
        joined = "\n".join(chunks)
        for item in ("Выравнивание стен", "Укладка плитки", "Монтаж розеток", "сантехника", "Штукатурка стен", "Поклейка обоев", "Установка сантехники"):
            self.assertIn(item, joined)

    def test_fixed_splits_long_emoji_no_space_line_without_loss(self):
        pub = load_public_rows(Path("output/ngram-comparison/splits/validation.jsonl"))
        for idx in (63, 238, 423, 551):
            text = pub[idx]["description"]
            chunks = chunk_ad_fixed(text)
            self.assertTrue(all(len(c) <= 425 for c in chunks), f"pub_val#{idx} had oversized chunk")
            sc = _word_counter(text)
            jc = _word_counter("\n".join(chunks))
            missing = [w for w, c in sc.items() if jc[w] < c]
            self.assertEqual(missing, [], f"pub_val#{idx} lost words: {missing}")

    def test_fixed_zero_word_loss_across_active_splits(self):
        files = [
            ("pub", Path("output/ngram-comparison/splits/train.jsonl")),
            ("pub", Path("output/ngram-comparison/splits/calibration.jsonl")),
            ("pub", Path("output/ngram-comparison/splits/validation.jsonl")),
            ("v3", Path("datasets/own_v3_clean/train.jsonl")),
            ("v3", Path("datasets/own_v3_clean/validation.jsonl")),
            ("v3", Path("datasets/own_v3_clean/test.jsonl")),
        ]
        for kind, path in files:
            rows = load_public_rows(path) if kind == "pub" else load_v3_rows_full(path)
            for idx, r in enumerate(rows):
                text = r["description"]
                chunks = chunk_ad_fixed(text)
                sc = _word_counter(text)
                jc = _word_counter("\n".join(chunks))
                missing = [w for w, c in sc.items() if jc[w] < c]
                self.assertEqual(missing, [], f"{path.name}[{idx}] lost words: {missing}")


if __name__ == "__main__":
    unittest.main()
