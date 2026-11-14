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
  return body.drafts;
}
