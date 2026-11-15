"""Smoke-test running frontend -> TF-IDF -> Qwen, with synthetic input only."""
import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DESCRIPTION = 'Укладываем плитку и керамогранит на стены и пол в Казани. Стоимость согласуем после осмотра.'


def request(base, route, payload=None):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = Request(base.rstrip('/') + route, data=body, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=125) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:15173')
    args = parser.parse_args()
    try:
        status = request(args.base_url, '/api/status')
        assert status.get('mode') == 'backend' and status.get('categoryGeneration') and status.get('ready'), status
        source = {'title': 'Плиточные работы', 'description': DESCRIPTION, 'mcId': 101}
        categories = request(args.base_url, '/api/categories', source)['categories']
        selected = next((c for c in categories if c['id'] == 103), None)
        assert selected, f'TF-IDF did not suggest tilework: {categories}'
        result = request(args.base_url, '/api/generate', {**source, 'categories': [selected]})
        assert len(result['drafts']) == 1 and result['drafts'][0]['categoryId'] == 103, result
        assert result['drafts'][0]['title'] and result['drafts'][0]['description'], result
        print(json.dumps({'status': 'ok', 'suggested_ids': [c['id'] for c in categories],
                          'generated_id': result['drafts'][0]['categoryId']}, ensure_ascii=False))
    except (AssertionError, HTTPError, URLError, KeyError, ValueError) as exc:
        parser.exit(1, f'Pipeline smoke failed: {exc}\n')


if __name__ == '__main__':
    main()
