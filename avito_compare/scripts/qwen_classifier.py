"""Zero-shot multi-label service classification with locally hosted vLLM."""
import json
import time
import urllib.error
import urllib.request

IDS = (101, 102, 103, 104, 105, 106)
CATEGORY_GUIDE = '''101 Ремонт под ключ: комплексный ремонт помещения с несколькими этапами; одно слово «под ключ» недостаточно.
102 Штукатурные работы: оштукатуривание стен, откосов, механизированная штукатурка, выравнивание штукатуркой.
103 Плиточные работы: укладка плитки, керамогранита, мозаики, фартука, затирка плиточных швов.
104 Сантехника: трубы, водоснабжение, канализация, раковины, унитазы, смесители, водяное отопление, подключение стиральных машин к воде.
105 Электрика: проводка, розетки, выключатели, светильники, электрощит, электрический тёплый пол.
106 Поклейка обоев и малярные работы: обои, покраска, шпаклёвка, грунтовка и шлифовка для малярного цикла.'''
SYSTEM = f'''Ты точный многоразметочный классификатор объявлений об услугах.\n{CATEGORY_GUIDE}\nПравила:\n1) Выбирай категорию, только если исполнитель действительно предлагает соответствующие работы.\n2) Не выбирай категорию за отрицания («не делаем»), сравнения, чужих мастеров, пожелания клиента, планы без предложения услуги.\n3) Общая фраза «ремонт под ключ» не означает автоматическое предложение штукатурки, плитки, сантехники, электрики и малярки.\n4) Различай штукатурку (102) и шпаклёвку под покраску (106).\n5) Категорию исходного объявления исключай.\n6) Для каждого найденного mcId укажи точную цитату из текста, подтверждающую работу. Если не можешь привести точную цитату, не добавляй категорию.\n7) Верни только объект с полем matches, без объяснений. Никаких выдуманных сведений.'''
SCHEMA = {
    'type': 'object', 'properties': {'matches': {
        'type': 'array', 'items': {'type': 'object', 'properties': {
            'mcId': {'type': 'integer', 'enum': list(IDS)},
            'evidence': {'type': 'string'}
        }, 'required': ['mcId', 'evidence'], 'additionalProperties': False}
    }}, 'required': ['matches'], 'additionalProperties': False
}


def classify(description, source_mc_id=101, base_url='http://127.0.0.1:8000', model='avito-qwen', timeout=180, retries=2):
    if source_mc_id not in IDS:
        raise ValueError('mcId must be from 101 to 106')
    if not isinstance(description, str) or not description.strip():
        raise ValueError('description required')
    # Align input limits with the TF-IDF baseline.
    description = description[:6000]
    payload = {
        'model': model, 'temperature': 0, 'max_tokens': 640,
        'messages': [{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': f'Исходная микрокатегория: {source_mc_id}\nТекст объявления:\n{description}'}],
        'structured_outputs': {'json': SCHEMA}
    }
    req = urllib.request.Request(base_url.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.load(resp)
            choice = obj['choices'][0]
            if choice.get('finish_reason') == 'length':
                raise RuntimeError('Qwen output truncated (increase max_tokens)')
            parsed = json.loads(choice['message']['content'])
            matches = parsed['matches']
            if not isinstance(matches, list):
                raise ValueError('matches is not array')
            selected = set()
            rejected = []
            for item in matches:
                mid = item.get('mcId')
                evidence = item.get('evidence')
                if type(mid) is not int or mid not in IDS or not isinstance(evidence, str):
                    rejected.append({'reason': 'invalid_fields', 'item': item})
                elif mid == source_mc_id:
                    rejected.append({'reason': 'source_category', 'item': item})
                elif not evidence.strip() or evidence not in description:
                    rejected.append({'reason': 'unverified_evidence', 'item': item})
                else:
                    selected.add(mid)
            return {'mcIds': sorted(selected), 'rejected': rejected,
                    'usage': obj.get('usage', {})}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise RuntimeError(f'Qwen request failed: {exc}') from exc
            time.sleep(1.5 * (attempt + 1))


def as_drafts(ids):
    return {'drafts': [{'mcId': mid, 'title': None, 'text': None} for mid in sorted(set(ids))]}
