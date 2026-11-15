import { createServer } from 'node:http';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { resolve, sep } from 'node:path';
import { mockCreation, mockResponse } from './mock-api.mjs';

const projectRoot = fileURLToPath(new URL('../', import.meta.url));
const apiPaths = new Set(['/api/create', '/api/categories', '/api/generate']);
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml' };

function json(response, status, body) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  response.end(JSON.stringify(body));
}

export function createFrontendServer({ preview = false, apiTarget = 'http://127.0.0.1:18001', mockApi = false, mockDelay = 500 } = {}) {
  const root = resolve(projectRoot, preview ? 'dist' : '.');
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url, 'http://localhost');
      if (url.pathname === '/api/status' && request.method === 'GET') {
        if (mockApi) return json(response, 200, { mode: 'demo', ready: true, categoryGeneration: true });
        try {
          const upstream = await fetch(new URL('/ready', apiTarget), { signal: AbortSignal.timeout(8_000) });
          const body = await upstream.json();
          const ready = upstream.ok && body.status === 'ready';
          return json(response, ready ? 200 : 503, {
            mode: 'backend', ready,
            categoryGeneration: body.categoryGeneration === true, model: body.model,
            ...(ready ? {} : { message: body.message ?? 'Qwen пока недоступен. Проверьте backend и SSH-туннель.', component: body.component, detail: body.detail }),
          });
        } catch {
          return json(response, 503, { mode: 'backend', ready: false, categoryGeneration: false,
            message: 'Не удалось подключиться к backend. Проверьте, что он запущен на порту 18001.' });
        }
      }
      if (apiPaths.has(url.pathname)) {
        if (request.method !== 'POST') {
          response.setHeader('Allow', 'POST');
          return json(response, 405, { message: 'Используйте POST.' });
        }
        const chunks = [];
        let bytes = 0;
        for await (const chunk of request) {
          bytes += chunk.length;
          const limit = 5 * 1024 * 1024 + (/^multipart\/form-data(?:;|$)/i.test(request.headers['content-type'] ?? '') ? 64 * 1024 : 0);
          if (bytes > limit) return json(response, 413, { message: 'Слишком большой запрос.' });
          chunks.push(chunk);
        }
        const body = Buffer.concat(chunks);
        const contentType = request.headers['content-type'] || 'application/json';
        const accept = request.headers.accept || 'application/json';
        if (mockApi) {
          if (url.pathname === '/api/create') {
            const result = await mockCreation(body, contentType, url.searchParams.get('generateText'), mockDelay);
            if (result.status !== 200) return json(response, result.status, result.body);
            if (!accept.includes('application/x-ndjson')) {
              const results = [];
              for await (const event of result.events) {
                if (event.type === 'item_done') {
                  const { type, completed, total, ...item } = event;
                  results.push(item);
                }
              }
              return json(response, 200, result.batch ? { results } : { drafts: results[0].drafts });
            }
            response.writeHead(200, { 'Content-Type': 'application/x-ndjson; charset=utf-8', 'Cache-Control': 'no-cache, no-transform', 'X-Accel-Buffering': 'no' });
            response.flushHeaders();
            for await (const event of result.events) {
              if (response.destroyed) break;
              if (!response.write(JSON.stringify(event) + '\n')) await once(response, 'drain');
            }
            return response.end();
          }
          let payload;
          try { payload = JSON.parse(body.toString('utf8')); }
          catch { return json(response, 400, { message: 'Ожидается JSON.' }); }
          await new Promise(done => setTimeout(done, mockDelay));
          const result = mockResponse(url.pathname, payload);
          return json(response, result.status, result.body);
        }
        const abort = new AbortController();
        let idleTimer;
        const resetTimer = () => {
          clearTimeout(idleTimer);
          idleTimer = setTimeout(() => abort.abort(), 120_000);
        };
        const disconnected = () => { if (!response.writableEnded) abort.abort(); };
        response.on('close', disconnected);
        resetTimer();
        try {
          const upstream = await fetch(new URL(url.pathname + url.search, apiTarget), {
            method: 'POST', headers: { 'Content-Type': contentType, Accept: accept }, body, signal: abort.signal,
          });
          response.writeHead(upstream.status, {
            'Content-Type': upstream.headers.get('content-type') ?? 'application/json',
            'Cache-Control': 'no-cache, no-transform', 'X-Accel-Buffering': 'no',
            ...(upstream.headers.has('x-request-id') ? { 'X-Request-ID': upstream.headers.get('x-request-id') } : {}),
          });
          response.flushHeaders();
          // Forward every chunk as it arrives; never buffer the entire batch.
          if (upstream.body) {
            for await (const chunk of upstream.body) {
              resetTimer();
              if (!response.write(chunk)) await once(response, 'drain', { signal: abort.signal });
            }
          }
          return response.end();
        } catch {
          if (response.headersSent) return response.destroy();
          if (!response.destroyed) return json(response, 502, { message: 'API недоступен. Повторите попытку позже.' });
        } finally {
          clearTimeout(idleTimer);
          response.off('close', disconnected);
        }
      }
      if (!['GET', 'HEAD'].includes(request.method)) return json(response, 405, { message: 'Метод не поддерживается.' });
      const pathname = decodeURIComponent(url.pathname);
      const relative = pathname === '/' ? 'index.html' : pathname.slice(1);
      // Не отдаём конфигурацию, тесты или файлы за пределами frontend.
      if (!['index.html', 'favicon.svg'].includes(relative) && !/^src\/[\w/-]+\.(js|css)$/.test(relative)) {
        return json(response, 404, { message: 'Страница не найдена.' });
      }
      const path = resolve(root, relative);
      if (!path.startsWith(root + sep)) return json(response, 404, { message: 'Страница не найдена.' });
      const body = await readFile(path);
      const extension = path.slice(path.lastIndexOf('.'));
      response.writeHead(200, { 'Content-Type': types[extension] ?? 'application/octet-stream', 'Cache-Control': 'no-cache' });
      response.end(request.method === 'HEAD' ? undefined : body);
    } catch (error) {
      json(response, error.code === 'ENOENT' ? 404 : 500, { message: error.code === 'ENOENT' ? 'Файл не найден. Сначала выполните npm run build для preview.' : 'Не удалось обработать запрос.' });
    }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const apiTarget = process.env.API_TARGET || 'http://127.0.0.1:18001';
  const mockApi = process.env.MOCK_API === '1';
  if (apiTarget && !/^https?:\/\//.test(apiTarget)) throw new Error('API_TARGET должен быть URL с http:// или https://');
  const server = createFrontendServer({ preview: process.argv.includes('--preview'), apiTarget, mockApi });
  server.on('error', error => { console.error(error.message); process.exitCode = 1; });
  server.listen(Number(process.env.PORT ?? 5173), process.env.HOST ?? '127.0.0.1', () => {
    console.log(`Local: http://${process.env.HOST ?? '127.0.0.1'}:${server.address().port}`);
    console.log(mockApi ? 'DEMO: локальные заглушки, Qwen не используется' : `Генерация через backend: ${apiTarget}`);
  });
}
