import unittest

from avito_compare.scripts.qwen3_category_aware_experiment import ALL_IDS, chunk_ad, load_public_rows, load_v3_rows_full


class CategoryAwareChunkerTests(unittest.TestCase):
    def test_preserves_question_and_short_answer_together(self):
        text = "Делаем ли мы сантехнику?\nДа.\n\nВыполняем ли электромонтаж?\nНет, электрику не берём."
        chunks = chunk_ad(text)
        self.assertEqual(len(chunks), 2)
        self.assertIn("Да.", chunks[0])
        self.assertIn("Нет, электрику не берём.", chunks[1])

    def test_keeps_heading_attached_across_blank_line(self):
        text = "Что НЕ делаем:\n\n— Штукатурка стен\n— Поклейка обоев"
        chunks = chunk_ad(text)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all("Что НЕ делаем:" in ch for ch in chunks))

    def test_keeps_opposing_bullets_separated(self):
        text = "• Больше не берём заявки на штукатурное выравнивание.\n• Владелец сам не выполняет полный ремонт дома, но его штатные мастера делают."
        chunks = chunk_ad(text)
        self.assertEqual(len(chunks), 2)
        self.assertIn("штукатурное", chunks[0])
        self.assertNotIn("штатные мастера", chunks[0])

    def test_does_not_truncate_long_ads(self):
        parts = [f"Абзац {i}: выполняем работы по ремонту объекта номер {i} с проверкой качества." for i in range(120)]
        full = "\n\n".join(parts)
        self.assertGreater(len(full), 7000)
        chunks = chunk_ad(full)
        self.assertTrue(any("номер 119" in ch for ch in chunks))

    def test_public_never_fabricates_101_and_v3_has_101(self):
        from pathlib import Path
        pub = load_public_rows(Path("output/ngram-comparison/splits/validation.jsonl"))
        self.assertTrue(all(not r["has_101_label"] and 101 not in r["labels"] for r in pub[:20]))
        v3_path = Path("/tmp/innopolis-dataset-v3-review/datasets/own_v3/validation.jsonl")
        if v3_path.exists():
            v3 = load_v3_rows_full(v3_path)
            self.assertTrue(all(r["has_101_label"] for r in v3[:20]))
            self.assertTrue(any(101 in r["evaluation_labels"] for r in v3))


if __name__ == "__main__":
    unittest.main()
