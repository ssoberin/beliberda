import unittest

from verifier import verify_response


class VerificationTests(unittest.TestCase):
    def test_positive_requires_exact_quote(self):
        text = 'Монтируем трубы и ставим раковины.'
        valid = {'offered': True, 'reason': 'Предлагаем сами', 'evidence': 'Монтируем трубы'}
        self.assertTrue(verify_response(valid, text)['accepted'])
        self.assertEqual(verify_response({**valid, 'evidence': 'ставим плитку'}, text)['rejection'],
                         'evidence_not_exact_substring')

    def test_negative_and_malformed(self):
        src = 'Электрику не делаем.'
        self.assertEqual(verify_response({'offered': False, 'reason': 'Отказ',
                                          'evidence': 'не делаем'}, src)['rejection'], 'model_rejected')
        self.assertEqual(verify_response({'offered': 'false', 'reason': 'Отказ',
                                          'evidence': ''}, src)['rejection'], 'invalid_offered_type')
        self.assertEqual(verify_response({'offered': True, 'reason': 'Да',
                                          'evidence': ''}, src)['rejection'], 'missing_positive_evidence')


if __name__ == '__main__':
    unittest.main()
