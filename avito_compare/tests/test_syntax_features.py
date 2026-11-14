#!/usr/bin/env python3
"""Unit tests verifying UD dependency-graph syntax feature extraction on Russian renovation edge cases."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avito_compare.scripts.syntax_features import SyntaxAnalyzer, extract_pair_features


class TestSyntaxFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sa = SyntaxAnalyzer()

    def test_direct_negation(self) -> None:
        txt = "Электрику не выполняем."
        f = extract_pair_features(self.sa, [txt], [0], [0.7], 105, len(txt))
        self.assertEqual(f["best_chunk_polarity"], "negated")
        self.assertEqual(f["syn_has_neg_on_action"], 1)
        self.assertEqual(f["syn_affirmative_offer"], 0)
        self.assertEqual(f["dominant_subject"], "seller_team")
        self.assertGreater(f["syn_min_dist_service_action"], 0)
        self.assertGreater(f["syn_min_dist_service_neg"], 0)

    def test_not_only_but_also_both_positive(self) -> None:
        txt = "Мы не только укладываем плитку, но и устанавливаем сантехнику."
        f_tile = extract_pair_features(self.sa, [txt], [0], [0.7], 103, len(txt))
        f_plumb = extract_pair_features(self.sa, [txt], [0], [0.7], 104, len(txt))
        for f in (f_tile, f_plumb):
            self.assertEqual(f["best_chunk_polarity"], "offered")
            self.assertEqual(f["syn_has_neg_on_action"], 0)
            self.assertEqual(f["syn_has_not_only"], 1)
            self.assertEqual(f["syn_affirmative_offer"], 1)

    def test_double_negation_cancelled_refusal(self) -> None:
        txt = "Больше не отказываемся от электромонтажа."
        f = extract_pair_features(self.sa, [txt], [0], [0.7], 105, len(txt))
        self.assertEqual(f["best_chunk_polarity"], "double_neg_positive")
        self.assertEqual(f["syn_double_neg_cancel"], 1)
        self.assertEqual(f["syn_has_neg_on_action"], 0)
        self.assertEqual(f["syn_affirmative_offer"], 1)
        self.assertEqual(f["temporal_state"], "now_only")

    def test_third_party_contractor_and_seller_team(self) -> None:
        txt_3p = "Электрику выполняет подрядчик заказчика."
        f_3p = extract_pair_features(self.sa, [txt_3p], [0], [0.7], 105, len(txt_3p))
        self.assertEqual(f_3p["best_chunk_polarity"], "third_party")
        self.assertEqual(f_3p["dominant_subject"], "third_party")
        self.assertEqual(f_3p["syn_third_party_actor"], 1)

        txt_sel = "Наши мастера выполняют электромонтаж."
        f_sel = extract_pair_features(self.sa, [txt_sel], [0], [0.7], 105, len(txt_sel))
        self.assertEqual(f_sel["best_chunk_polarity"], "offered")
        self.assertEqual(f_sel["dominant_subject"], "seller_team")
        self.assertEqual(f_sel["syn_seller_actor"], 1)

    def test_temporal_past_neg_to_now_pos(self) -> None:
        txt = "Раньше электрику не делали, теперь выполняем."
        f = extract_pair_features(self.sa, [txt], [0], [0.7], 105, len(txt))
        self.assertEqual(f["best_chunk_polarity"], "offered")
        self.assertEqual(f["temporal_state"], "past_neg_to_now_pos")
        self.assertEqual(f["syn_temporal_resumed"], 1)
        self.assertEqual(f["syn_has_conflicting_claims"], 1)
        self.assertEqual(f["syn_affirmative_offer"], 1)

    def test_global_turnkey_and_partial_refusal_unknown_service(self) -> None:
        txt = "Только ремонт под ключ, отдельные работы не берём."
        f = extract_pair_features(self.sa, [txt], [0], [0.3], 105, len(txt))
        self.assertEqual(f["best_chunk_polarity"], "unknown")
        self.assertEqual(f["dominant_subject"], "unknown")
        self.assertEqual(f["syn_global_turnkey_only"], 1)
        self.assertEqual(f["syn_global_partial_refusal"], 1)
        self.assertEqual(f["syn_min_dist_service_action"], -1)

    def test_non_service_negation_guard(self) -> None:
        txt = "Укладываем плитку, монтируем электрику. Не берём предоплату за весь ремонт."
        f = extract_pair_features(self.sa, [txt], [0], [0.7], 105, len(txt))
        self.assertEqual(f["best_chunk_polarity"], "offered")
        self.assertEqual(f["syn_has_neg_on_action"], 0)
        self.assertEqual(f["syn_global_partial_refusal"], 0)


if __name__ == "__main__":
    unittest.main()
