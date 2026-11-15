export const API_ROUTES = {
  categories: '/api/categories',
  generate: '/api/generate',
};

async function post(url, payload, signal) {
  const timeout = AbortSignal.timeout(120_000);
  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(payload),
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error(timeout.aborted
      ? 'Сервер долго не отвечает. Попробуйте ещё раз.'
      : 'Не удалось связаться с сервером. Проверьте подключение и попробуйте ещё раз.');
  }
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(response.ok ? 'Сервер вернул некорректный JSON.' : `Ошибка сервера (${response.status}). Попробуйте ещё раз.`);
  }
  if (!response.ok) {
    throw new Error(typeof body?.message === 'string' ? body.message : `Ошибка сервера (${response.status}). Попробуйте ещё раз.`);
  }
  return body;
}

export function normalizeCategories(body) {
  const values = Array.isArray(body) ? body : body?.categories;
  if (!Array.isArray(values)) throw new Error('В ответе сервера нет списка категорий.');
  const unique = new Map();
  for (const value of values) {
    const category = typeof value === 'string' ? { id: value, name: value } : value;
    if (!category || !['string', 'number'].includes(typeof category.id)
      || !String(category.id).trim() || typeof category.name !== 'string' || !category.name.trim()) {
      throw new Error('Сервер вернул некорректную категорию.');
    }
    unique.set(String(category.id), { id: category.id, name: category.name.trim() });
  }
  return [...unique.values()];
}

export async function generateCategories(payload, signal) {
  return normalizeCategories(await post(API_ROUTES.categories, payload, signal));
}

export async function generateListings(payload, signal) {
  const body = await post(API_ROUTES.generate, payload, signal);
  if (!Array.isArray(body?.drafts) || body.drafts.some(draft => !draft
    || !['string', 'number'].includes(typeof draft.categoryId)
    || typeof draft.title !== 'string' || typeof draft.description !== 'string')) {
    throw new Error('Сервер вернул некорректный список объявлений.');
  }
  const warnings = body.warnings ?? [];
  if (!Array.isArray(warnings) || warnings.some(warning => !warning
    || !['string', 'number'].includes(typeof warning.categoryId) || typeof warning.message !== 'string')) {
    throw new Error('Сервер вернул некорректные предупреждения.');
  }
  return { drafts: body.drafts, warnings };
}

export async function getConnectionStatus(signal) {
  const response = await fetch('/api/status', { signal, headers: { Accept: 'application/json' } });
  const body = await response.json();
  if (!['backend', 'demo'].includes(body?.mode) || typeof body.ready !== 'boolean') {
    throw new Error('Не удалось определить состояние генерации.');
  }
  return body;
}

// One route for manual entries, JSON batches and raw UTF-8 CSV uploads.
export async function createListings(payload, { signal, onEvent, contentType = 'application/json', generateText = true } = {}) {
  const idle = new AbortController();
  let timer;
  let reader;
  const resetTimer = () => {
    clearTimeout(timer);
    timer = setTimeout(() => idle.abort(), 120_000);
  };
  resetTimer();
  try {
    const response = await fetch(`/api/create?generateText=${generateText}`, {
      method: 'POST',
      headers: { 'Content-Type': contentType, Accept: 'application/x-ndjson' },
      body: typeof payload === 'string' ? payload : JSON.stringify(payload),
      signal: signal ? AbortSignal.any([signal, idle.signal]) : idle.signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.message || `Ошибка сервера (${response.status}).`);
    }
    if (!response.headers.get('content-type')?.includes('application/x-ndjson') || !response.body) {
      throw new Error('Сервер не поддерживает потоковую выдачу результатов.');
    }
    reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finished = false;
    let started = false;
    const emit = line => {
      if (!line.trim()) return;
      let event;
      try { event = JSON.parse(line); } catch { throw new Error('Сервер вернул повреждённый поток JSON.'); }
      if (!['start', 'categories', 'drafts', 'category_error', 'item_done', 'done', 'heartbeat'].includes(event?.type)
        || finished || (!started && event.type !== 'start')) {
        throw new Error('Сервер вернул некорректное событие.');
      }
      if (event.type === 'start') started = true;
      if (['drafts', 'item_done'].includes(event.type) && (!Array.isArray(event.drafts)
        || event.drafts.some(d => !d || !Number.isInteger(d.mcId)
          || !(d.title === null || typeof d.title === 'string')
          || !(d.text === null || typeof d.text === 'string')))) {
        throw new Error('Сервер вернул некорректные черновики.');
      }
      if (event.type === 'done') finished = true;
      onEvent?.(event);
    };
    while (true) {
      const { value, done } = await reader.read();
      signal?.throwIfAborted();
      resetTimer();
      buffer += decoder.decode(value, { stream: !done });
      let end;
      while ((end = buffer.indexOf('\n')) !== -1) {
        emit(buffer.slice(0, end));
        buffer = buffer.slice(end + 1);
      }
      if (done) break;
    }
    if (buffer.trim()) emit(buffer);
    if (!finished) throw new Error('Соединение прервано. Уже полученные результаты сохранены; обработка завершена не полностью.');
  } catch (error) {
    if (signal?.aborted) throw error;
    if (idle.signal.aborted) throw new Error('Сервер долго не отвечает. Полученные результаты сохранены.');
    if (error instanceof TypeError) throw new Error('Не удалось связаться с сервером. Полученные результаты сохранены.');
    throw error;
  } finally {
    clearTimeout(timer);
    if (reader) {
      await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  }
}
