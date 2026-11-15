import unittest

from segmentation_benchmark import fragments, units


class SegmentationTests(unittest.TestCase):
    def test_question_answer_not_detached(self):
        parts = units('Выполняем плиточные работы? Нет. Клеим обои.')
        self.assertEqual(parts, ['Выполняем плиточные работы? Нет.', 'Клеим обои.'])

    def test_answer_on_next_line(self):
        parts = units('Делаем электрику?\nНет.\nУстанавливаем раковины.')
        self.assertEqual(parts, ['Делаем электрику? Нет.', 'Устанавливаем раковины.'])

    def test_elliptic_denial_not_detached(self):
        self.assertEqual(units('Плиточные работы.\nНе делаем.\nСантехнику выполняем.'),
                         ['Плиточные работы. Не делаем.', 'Сантехнику выполняем.'])

    def test_bullet_negation_and_windows(self):
        parts = fragments('Услуги:\n• Электрику не делаем.\n• Укладываем плитку.')
        self.assertIn('Электрику не делаем.', parts['units'])
        self.assertIn('Электрику не делаем. Укладываем плитку.', parts['window2'])
        self.assertEqual(len(parts['units']), len(parts['window3']))

    def test_no_semantic_claim_from_split(self):
        self.assertEqual(units('Плитку кладёт другой мастер, мы не делаем.'),
                         ['Плитку кладёт другой мастер, мы не делаем.'])


if __name__ == '__main__':
    unittest.main()
