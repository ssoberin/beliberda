import { parseCsv } from './import.js';

// A single field accepts a service description or structured records.
export function prepareSubmission(value, title = '', filename = '') {
  const source = value.replace(/^\uFEFF/, '').trim();
  title = title.trim();
  if (!source) throw new Error('Введите описание услуги или загрузите файл.');
  if (new TextEncoder().encode(source).length > 5 * 1024 * 1024) throw new Error('Максимальный размер данных — 5 МБ.');
  if (title.length > 100) throw new Error('Название объявления должно быть не длиннее 100 символов.');

  const withTitle = item => item && typeof item === 'object' && !Array.isArray(item)
    && title && (item.title === undefined || item.title === '') ? { ...item, title } : item;
  const firstLine = source.split(/\r?\n/, 1)[0];
  const csvHeader = /(?:^|[,;])\s*(?:"description"|description)\s*(?=[,;]|$)/.test(firstLine);
  const kind = /\.json$/i.test(filename) ? 'json' : /\.csv$/i.test(filename) ? 'csv'
    : /\.txt$/i.test(filename) ? 'text' : /^[\[{]/.test(source) ? 'json' : csvHeader ? 'csv' : 'text';

  if (kind === 'json') {
    let payload;
    try { payload = JSON.parse(source); }
    catch { throw new Error('Проверьте JSON: кавычки, запятые и скобки.'); }
    const batch = Array.isArray(payload) || (payload !== null && typeof payload === 'object' && 'items' in payload);
    if (Array.isArray(payload)) payload = payload.map(withTitle);
    else if (batch && Array.isArray(payload.items)) payload = { ...payload, items: payload.items.map(withTitle) };
    else if (!batch) payload = withTitle(payload);
    return { payload, contentType: 'application/json', batch };
  }
  if (kind === 'csv') {
    // Preserve raw CSV unless a shared title needs to be added to its records.
    if (!title) return { payload: source, contentType: 'text/csv', batch: true };
    const payload = parseCsv(source).map(row => {
      const item = Object.fromEntries(Object.entries(row).filter(([, value]) => value !== ''));
      if (item.mcId !== undefined) item.mcId = Number(item.mcId);
      return withTitle(item);
    });
    return { payload, contentType: 'application/json', batch: true };
  }
  if (source.length > 12000) throw new Error('Описание услуги должно быть не длиннее 12 000 символов.');
  return { payload: { title, description: source, mcId: 101 }, contentType: 'application/json', batch: false };
}
