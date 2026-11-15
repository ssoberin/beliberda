import test from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { createServer } from 'node:http';
import { createListings } from '../src/api.js';
import { prepareSubmission } from '../src/submission.js';
import { createFrontendServer } from '../scripts/server.mjs';

const draft = { mcId: 104, title: 'Сантехника 🔧', text: 'Меняем трубы.' };
const events = [
  { type: 'start', total: 1, generateText: true },
  { type: 'drafts', index: 0, drafts: [draft] },
  { type: 'item_done', index: 0, drafts: [draft], completed: 1, total: 1 },
  { type: 'done', completed: 1, total: 1, failed: 0 },
];

test('единый API: NDJSON через разрывы строк и UTF-8, одиночный JSON, батч и CSV', async t => {
  const bytes = new TextEncoder().encode(events.map(e => JSON.stringify(e)).join('\r\n'));
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, ...options });
    return new Response(new ReadableStream({ start(controller) {
      for (let i = 0; i < bytes.length; i += 3) controller.enqueue(bytes.slice(i, i + 3));
      controller.close();
    } }), { headers: { 'Content-Type': 'application/x-ndjson' } });
  });
  const received = [];
  await createListings({ description: 'Меняем трубы.' }, { onEvent: e => received.push(e) });
  assert.deepEqual(received, events);
  assert.equal(calls[0].url, '/api/create?generateText=true');
  assert.equal(calls[0].headers.Accept, 'application/x-ndjson');
  assert.deepEqual(JSON.parse(calls[0].body), { description: 'Меняем трубы.' });
  await createListings([{ description: 'Трубы.' }], { generateText: false });
  assert.equal(calls[1].url, '/api/create?generateText=false');
  assert.ok(Array.isArray(JSON.parse(calls[1].body)));
  await createListings('testcase,description\ncase-001,Трубы', { contentType: 'text/csv' });
  assert.equal(calls[2].headers['Content-Type'], 'text/csv');
  assert.equal(calls[2].body, 'testcase,description\ncase-001,Трубы');
});

test('обрыв после первого результата сохраняет его и сообщает о неполной обработке', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response(events.slice(0, 2).map(e => JSON.stringify(e)).join('\n'), {
    headers: { 'Content-Type': 'application/x-ndjson' },
  }));
  const received = [];
  await assert.rejects(createListings({}, { onEvent: e => received.push(e) }), /обработка завершена не полностью/);
  assert.deepEqual(received[1].drafts, [draft]);
});

test('поток проверяет ошибки сервера и невалидные события', async t => {
  const mocked = t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ message: 'Некорректный CSV' }), { status: 422 }));
  await assert.rejects(createListings({}), /Некорректный CSV/);
  mocked.mock.mockImplementation(async () => new Response('{broken}\n', { headers: { 'Content-Type': 'application/x-ndjson' } }));
  await assert.rejects(createListings({}), /повреждённый поток/);
  mocked.mock.mockImplementation(async () => new Response('{"type":"surprise"}\n', { headers: { 'Content-Type': 'application/x-ndjson' } }));
  await assert.rejects(createListings({}), /некорректное событие/);
});

async function start(t, server) {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => new Promise(done => { server.close(done); server.closeAllConnections(); }));
  return `http://127.0.0.1:${server.address().port}`;
}

test('HTTP-прокси отдаёт черновик до окончания upstream и передаёт CSV без преобразования', { timeout: 5000 }, async t => {
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const upstream = await start(t, createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    assert.equal(req.url, '/api/create?generateText=true');
    assert.equal(req.headers['content-type'], 'text/csv');
    assert.equal(req.headers.accept, 'application/x-ndjson');
    assert.equal(Buffer.concat(chunks).toString(), 'description\nТрубы');
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
    res.write(events.slice(0, 2).map(e => JSON.stringify(e)).join('\n') + '\n');
    await gate;
    res.end(events.slice(2).map(e => JSON.stringify(e)).join('\n') + '\n');
  }));
  const proxy = await start(t, createFrontendServer({ apiTarget: upstream }));
  const realFetch = globalThis.fetch;
  t.mock.method(globalThis, 'fetch', (url, options) => realFetch(typeof url === 'string' && url.startsWith('/') ? proxy + url : url, options));
  const received = [];
  try {
    await createListings('description\nТрубы', { contentType: 'text/csv', onEvent: event => {
      received.push(event);
      if (event.type === 'drafts') release();
    } });
  } finally { release(); }
  assert.deepEqual(received, events);
});

test('HTTP: JSON, CSV, пустой результат и режим без текста через /api/create в деморежиме', async t => {
  const base = await start(t, createFrontendServer({ mockApi: true, mockDelay: 0 }));
  const post = (body, type = 'application/json', suffix = '') => fetch(base + '/api/create' + suffix, {
    method: 'POST', body: typeof body === 'string' ? body : JSON.stringify(body),
    headers: { 'Content-Type': type, Accept: 'application/x-ndjson' },
  });
  const result = await post({ description: 'Меняем трубы.', mcId: 101 });
  const stream = (await result.text()).trim().split('\n').map(line => JSON.parse(line));
  assert.equal(stream[0].type, 'start');
  assert.equal(stream.find(e => e.type === 'drafts').drafts[0].mcId, 104);
  const csv = await post('\uFEFFtestcase,description\r\ncase-001,"Меняем трубы, сифоны."\r\ncase-002,Собираем шкафы.\r\n', 'text/csv', '?generateText=false');
  const csvEvents = (await csv.text()).trim().split('\n').map(line => JSON.parse(line));
  const items = csvEvents.filter(e => e.type === 'item_done');
  assert.equal(items[0].testcase, 'case-001');
  assert.deepEqual(items[0].drafts, [{ mcId: 104, title: null, text: null }]);
  assert.deepEqual(items[1].drafts, []);
  assert.equal((await post({ description: ' ' })).status, 422);
});

test('multipart: файл уходит без чтения и ручного Content-Type, поток сохраняется', async t => {
  const file = new File(['\uFEFFdescription\r\n"Меняем трубы, сифоны."\r\n'], 'original.csv', { type: 'text/csv' });
  const { payload } = prepareSubmission('', 'Общее название', file);
  let expectedBytes, expectedType;
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const upstream = await start(t, createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    assert.equal(req.headers['content-type'], expectedType);
    const bytes = Buffer.concat(chunks);
    assert.deepEqual(bytes, expectedBytes);
    const form = await new Request('http://localhost', {
      method: 'POST', headers: { 'Content-Type':req.headers['content-type'] }, body:bytes,
    }).formData();
    assert.equal(form.get('file').name, 'original.csv');
    assert.equal(form.get('title'), 'Общее название');
    assert.deepEqual(await form.get('file').arrayBuffer(), await file.arrayBuffer());
    res.writeHead(200, { 'Content-Type':'application/x-ndjson' });
    res.write(events.slice(0,2).map(e => JSON.stringify(e)).join('\n') + '\n');
    await gate;
    res.end(events.slice(2).map(e => JSON.stringify(e)).join('\n') + '\n');
  }));
  const proxy = await start(t, createFrontendServer({ apiTarget:upstream }));
  const realFetch = globalThis.fetch;
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (typeof url !== 'string' || !url.startsWith('/')) return realFetch(url, options);
    assert.equal(options.body, payload);
    assert.equal(Object.hasOwn(options.headers, 'Content-Type'), false);
    const request = new Request(proxy + url, options);
    expectedType = request.headers.get('content-type');
    expectedBytes = Buffer.from(await request.clone().arrayBuffer());
    return realFetch(request);
  });
  const received = [];
  try {
    await createListings(payload, { onEvent:event => {
      received.push(event);
      if (event.type === 'drafts') release();
    } });
  } finally { release(); }
  assert.deepEqual(received, events);
});

test('деморежим принимает новый транспорт файла и вставленного текста', async t => {
  const base = await start(t, createFrontendServer({ mockApi:true, mockDelay:0 }));
  for (const payload of [
    prepareSubmission('Меняем трубы.').payload,
    prepareSubmission('', '', new File(['[{"description":"Меняем трубы."}]'], 'input.json')).payload,
  ]) {
    const result = await fetch(base + '/api/create', {
      method:'POST', body:payload, headers:{ Accept:'application/x-ndjson' },
    });
    assert.equal(result.status,200);
    const received = (await result.text()).trim().split('\n').map(line => JSON.parse(line));
    assert.equal(received[0].total,1);
    assert.equal(received.at(-1).failed,0);
  }
});

test('загрузка принимает до 300 объявлений включительно и сообщает лимит для 301', async t => {
  const base = await start(t, createFrontendServer({ mockApi:true, mockDelay:0 }));
  for (const count of [300,301]) {
    const file = new File([JSON.stringify(Array.from({ length:count }, () => ({ description:'Собираем мебель.' })))], 'input.json');
    const result = await fetch(base + '/api/create', {
      method:'POST', body:prepareSubmission('', '', file).payload, headers:{ Accept:'application/x-ndjson' },
    });
    assert.equal(result.status,count === 300 ? 200 : 422);
    if (count === 300) {
      const events = (await result.text()).trim().split('\n').map(line => JSON.parse(line));
      assert.equal(events[0].total,300);
      assert.deepEqual(events.at(-1),{ type:'done', total:300, completed:300, failed:0 });
    } else assert.match((await result.json()).message,/300/);
  }
});
