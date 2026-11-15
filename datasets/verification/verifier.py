"""Local Qwen binary verifier. No application imports and no model mutation."""
import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter

import httpx

VERSION = 'verification-v2'
VLLM = 'http://127.0.0.1:8000'
MODEL = 'avito-qwen'
LABELS = {102: 'Штукатурные работы', 103: 'Плиточные работы', 104: 'Сантехника',
          105: 'Электрика', 106: 'Поклейка обоев и малярные работы'}
GUIDE = {
    102: 'Штукатурка стен, откосов, раствором/по маякам; не путай со шпаклёвкой под окраску.',
    103: 'Укладка плитки, керамогранита, облицовка и затирка плиточных швов; не ламинат.',
    104: 'Монтаж труб, водоснабжения, раковин, смесителей и другой сантехники.',
    105: 'Проводка, розетки, выключатели, щит, светильники и прочий электромонтаж.',
    106: 'Обои, окраска, шпаклёвка под окраску и малярная отделка.',
}
SCHEMA = {'type': 'object', 'properties': {
    'offered': {'type': 'boolean'}, 'reason': {'type': 'string'},
    'evidence': {'type': 'string'}},
    'required': ['offered', 'reason', 'evidence'], 'additionalProperties': False}
SYSTEM = (
    'Ты проверяешь ОДНУ микрокатегорию услуг в объявлении. Верни JSON: offered, reason, evidence. '
    'offered=true, только если сам исполнитель явно предлагает хотя бы одну работу именно этой категории. '
    'Достаточно ОДНОЙ конкретной операции: укладываем плитку или керамогранит — 103; '
    'устанавливаем раковину или смеситель — 104; не требуй полного набора работ категории. '
    'Отрицание конкретной операции не запрещает категорию, если другую работу той же категории исполнитель предлагает. '
    'Не считай предложением работу заказчика, другого мастера, прошлый опыт, просьбу, гипотезу, пример, '
    'цитату клиента, SEO-теги, перечень материалов или общий «ремонт под ключ» без конкретной услуги. '
    'Если статус неясен, offered=false. Кандидатный фрагмент может быть ложной подсказкой: '
    'проверь его вместе с соседними фразами и всем объявлением. '
    'При offered=true evidence — дословная непрерывная цитата исходного объявления с подтверждением. '
    'При offered=false evidence может быть пустым или дословной цитатой с отказом. '
    'reason — короткое объяснение НА РУССКОМ без выдуманных фактов; верни только JSON.'
)


def key(description, mid, fragment, before, after):
    payload = [VERSION, MODEL, description, mid, fragment, before, after]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def verify_response(obj, description):
    if not isinstance(obj, dict) or type(obj.get('offered')) is not bool:
        return {'accepted': False, 'rejection': 'invalid_offered_type'}
    reason, evidence = obj.get('reason'), obj.get('evidence')
    if not isinstance(reason, str) or not reason.strip() or len(reason)>500:
        return {'accepted': False, 'rejection': 'invalid_reason'}
    if not isinstance(evidence, str) or len(evidence)>500:
        return {'accepted': False, 'rejection': 'invalid_evidence_type_or_length'}
    if obj['offered']:
        if not evidence.strip():
            return {'accepted': False, 'rejection': 'missing_positive_evidence'}
        if evidence not in description:
            return {'accepted': False, 'rejection': 'evidence_not_exact_substring'}
        return {'accepted': True, 'rejection': None}
    if evidence and evidence not in description:
        return {'accepted': False, 'rejection': 'negative_evidence_not_exact_substring'}
    return {'accepted': False, 'rejection': 'model_rejected'}


class Verifier:
    def __init__(self, cache_path, concurrency=4):
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.path.exists():
            for line in self.path.open(encoding='utf-8'):
                if line.strip():
                    record = json.loads(line)
                    self.cache[record['key']] = record
        self.concurrency = concurrency
        self.network_calls = 0

    async def run(self, jobs):
        """Jobs: (description, mid, fragment, before, after). Cache by content."""
        unique = {}
        for job in jobs:
            unique.setdefault(key(*job), job)
        missing = [(hash_key, job) for hash_key, job in unique.items()
                   if hash_key not in self.cache or
                   (self.cache[hash_key]['rejection'] or '').startswith('request_or_json_error')]
        sem = asyncio.Semaphore(self.concurrency)
        lock = asyncio.Lock()
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=5.0)) as client:
            async def one(hash_key, job):
                description, mid, fragment, before, after = job
                user = (f'Категория {mid}: {LABELS[mid]}. Граница: {GUIDE[mid]}\n'
                        f'Кандидатный фрагмент: {fragment}\nПредыдущий контекст: {before}\n'
                        f'Следующий контекст: {after}\nПолное объявление:\n{description}')
                payload = {'model': MODEL, 'temperature': 0, 'max_tokens': 250,
                           'messages': [{'role': 'system', 'content': SYSTEM},
                                        {'role': 'user', 'content': user}],
                           'structured_outputs': {'json': SCHEMA}}
                async with sem:
                    start = perf_counter()
                    self.network_calls += 1
                    try:
                        response = await client.post(f'{VLLM}/v1/chat/completions', json=payload)
                        response.raise_for_status()
                        choice = response.json()['choices'][0]
                        if choice.get('finish_reason') == 'length':
                            raise ValueError('truncated')
                        answer = json.loads(choice['message']['content'])
                        checked = verify_response(answer, description)
                        record = {'key': hash_key, 'mid': mid, 'answer': answer, **checked,
                                  'seconds': round(perf_counter()-start, 3)}
                    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                        record = {'key': hash_key, 'mid': mid, 'answer': None,
                                  'accepted': False, 'rejection': f'request_or_json_error: {type(exc).__name__}',
                                  'seconds': round(perf_counter()-start, 3)}
                    async with lock:
                        with self.path.open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                        self.cache[hash_key] = record
            await asyncio.gather(*(one(hash_key, job) for hash_key, job in missing))
        return [self.cache[key(*job)] for job in jobs]
