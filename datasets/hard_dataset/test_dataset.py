"""Checks for label/scope regressions and leakage; run with unittest discovery."""
import copy
import unittest

from build_v2 import build, validate
from expanded_cases import corrupt, PROTECTED


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = build()
        cls.rows = {r["item_id"]: r for rows in cls.splits.values() for r in rows}

    def test_entire_corpus(self):
        stats = validate(self.splits)
        self.assertEqual(sum(s["rows"] for s in stats.values()), 13960)
        for split in ("train", "validation", "test"):
            self.assertEqual(set(stats[split]["target_cardinality"]), set(range(6)))

    def test_nested_truth_and_concord(self):
        # Explicit expectations independent from the label-generation helper.
        for name in ("denial_of_denial", "concord", "refutation_refuted", "four_layers", "double_report", "four_layer_facts"):
            for mode in ("clean", "spelling", "grammar_spacing"):
                for variant, offered in ((0, [102, 103]), (1, [103])):
                    row = self.rows[f"v2-{name}-102-103-0-{variant}-{mode}"]
                    self.assertEqual(row["offered_microcategories"], offered)

    def test_partial_service_and_source(self):
        expected = {"challenge-001-0": [105], "challenge-001-1": [],
                    "challenge-002-0": [106], "challenge-002-1": [102],
                    "challenge-005-1": [101, 103], "challenge-017-0": [],
                    "challenge-018-1": [101]}
        for item_id, targets in expected.items():
            self.assertEqual([m["mcId"] for m in self.rows[item_id]["found_microcategories"]], targets)

    def test_no_uncontrolled_particle_corruption(self):
        text = "Мы не выполняем поклейку обоев, но не только устанавливаем розетки. Ни трубы, ни плитку не берём."
        for mode in ("spelling", "grammar_spacing"):
            result = corrupt(text, mode)
            self.assertNotEqual(result, text)
            self.assertEqual([x.lower() for x in PROTECTED.findall(result)], [x.lower() for x in PROTECTED.findall(text)])

    def test_validator_rejects_leakage(self):
        damaged = {key: list(value) for key, value in self.splits.items()}
        row = copy.deepcopy(next(r for r in damaged["train"] if "clean_item_id" not in r))
        row["item_id"] += "-leak"
        row["description"] += " Уточнение."
        damaged["test"].append(row)
        with self.assertRaises(AssertionError):
            validate(damaged)

    def test_validator_rejects_inverted_noisy_label(self):
        damaged = {key: list(value) for key, value in self.splits.items()}
        index = next(i for i, r in enumerate(damaged["train"]) if r.get("augmentation") == "spelling")
        row = copy.deepcopy(damaged["train"][index])
        row["offered_microcategories"] = []
        row["found_microcategories"] = []
        damaged["train"][index] = row
        with self.assertRaises(AssertionError):
            validate(damaged)


if __name__ == "__main__":
    unittest.main()
