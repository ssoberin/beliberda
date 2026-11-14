import test from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { createServer } from 'node:http';
import { generateCategories, generateListings, normalizeCategories } from '../src/api.js';
import { createFrontendServer } from '../scripts/server.mjs';

test('категории из JSON и строк: сохраняет ID, убирает дубликаты, отклоняет неверный ответ', () => {
  assert.deepEqual(normalizeCategories({ categories: [{ id: 104, name: ' Сантехника ' }, { id: 104, name: 'Сантехника' }] }), [{ id: 104, name: 'Сантехника' }]);
  assert.deepEqual(normalizeCategories(['Сантехника']), [{ id: 'Сантехника', name: 'Сантехника' }]);
  assert.deepEqual(normalizeCategories({ categories: [] }), []);
  assert.throws(() => normalizeCategories({ categories: [{ name: 'Без ID' }] }), /некорректную категорию/);
  assert.throws(() => normalizeCategories({ unexpected: [] }), /нет списка/);
});

test('API-клиент отправляет JSON на нужный адрес и проверяет ответы и ошибки', async t => {
  let response = { categories: [{ id: 105, name: 'Электрика' }] };
  let status = 200;
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, method: options.method, body: JSON.parse(options.body), headers: options.headers });
    return new Response(JSON.stringify(response), { status });
  });
  const payload = { title: 'Ремонт', description: 'Меняю розетки' };
  assert.deepEqual(await generateCategories(payload), response.categories);
  assert.equal(calls[0].url, '/api/categories');
  assert.equal(calls[0].method, 'POST');
  assert.equal(calls[0].headers['Content-Type'], 'application/json');
  assert.deepEqual(calls[0].body, payload);
  const categories = [{ id: 105, name: 'Электрика' }];
  response = { drafts: [{ categoryId: 105, title: 'Электрика', description: 'Меняю розетки' }] };
  assert.deepEqual(await generateListings({ ...payload, categories }), response.drafts);
  assert.equal(calls[1].url, '/api/generate');
  assert.deepEqual(calls[1].body.categories, categories);
  response = { drafts: [{}] };
  await assert.rejects(generateListings(payload), /некорректный список/);
  status = 503;
  response = { message: 'Сервис временно недоступен' };
  await assert.rejects(generateCategories(payload), /Сервис временно недоступен/);
});

test('API-клиент сообщает о неверном JSON и недоступном сервере, сохраняет отмену', async t => {
  const fetchMock = t.mock.method(globalThis, 'fetch', async () => new Response('<html>Ошибка</html>'));
  await assert.rejects(generateCategories({}), /некорректный JSON/);
  fetchMock.mock.mockImplementation(async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(generateCategories({}), /Не удалось связаться/);
  const controller = new AbortController();
  controller.abort();
  fetchMock.mock.mockImplementation(async (_url, options) => { options.signal.throwIfAborted(); });
  await assert.rejects(generateCategories({}, controller.signal), { name: 'AbortError' });
});

async function start(t, server) {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => new Promise((done, reject) => {
    server.close(error => error ? reject(error) : done());
    server.closeAllConnections();
  }));
  return `http://127.0.0.1:${server.address().port}`;
}

test('HTTP: страница, подбор категорий, генерация, валидация и изоляция исходников сервера', async t => {
  const base = await start(t, createFrontendServer({ mockDelay: 0 }));
  const post = (path, payload) => fetch(base + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const page = await fetch(base);
  assert.equal(page.status, 200);
  assert.match(await page.text(), /Новое объявление/);
  assert.match((await fetch(base + '/src/main.js')).headers.get('content-type'), /javascript/);
  const payload = { title: 'Услуги мастера', description: 'Укладываю плитку, устанавливаю сантехнику, меняю розетки. 🔧' };
  const response = await post('/api/categories', payload);
  assert.equal(response.status, 200);
  const { categories } = await response.json();
  assert.deepEqual(categories.map(category => category.id), [103, 104, 105]);
  const generated = await post('/api/generate', { ...payload, categories });
  assert.equal(generated.status, 200);
  const { drafts } = await generated.json();
  assert.equal(drafts.length, 3);
  assert.equal(drafts[0].categoryId, 103);
  assert.match(drafts[0].description, /🔧/u);
  assert.equal((await post('/api/categories', { description: '   ' })).status, 400);
  assert.equal((await post('/api/generate', { ...payload, categories: [] })).status, 400);
  assert.deepEqual((await (await post('/api/categories', { description: 'Предлагаю услуги' })).json()).categories, []);
  assert.equal((await fetch(base + '/api/categories')).status, 405);
  assert.equal((await fetch(base + '/scripts/server.mjs')).status, 404);
  assert.equal((await fetch(base + '/package.json')).status, 404);
});

test('HTTP: API_TARGET проксирует путь, JSON и ошибку настоящего API без подмены заглушкой', async t => {
  const upstream = await start(t, createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    assert.equal(request.url, '/api/categories');
    assert.equal(JSON.parse(Buffer.concat(chunks)).description, 'Тест прокси');
    response.writeHead(422, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify({ message: 'Ошибка внешнего API' }));
  }));
  const base = await start(t, createFrontendServer({ apiTarget: upstream }));
  const response = await fetch(base + '/api/categories', { method: 'POST', body: JSON.stringify({ description: 'Тест прокси' }) });
  assert.equal(response.status, 422);
  assert.deepEqual(await response.json(), { message: 'Ошибка внешнего API' });
});
