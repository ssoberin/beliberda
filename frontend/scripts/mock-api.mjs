import { CATEGORIES } from '../src/catalog.js';

const patterns = new Map([
  [101, /под ключ|комплексн\S* ремонт|ремонт квартир/iu],
  [102, /штукатур/iu],
  [103, /плитк|плиточ|керамогранит|мозаик/iu],
  [104, /сантех|труб|унитаз|раковин|смесител|протеч|водоснабж/iu],
  [105, /электри|розет|проводк|выключател|светильник|люстр/iu],
  [106, /обои|обоев|обоя|маляр|покрас|покраск|шпакл/iu],
]);

// Только демонстрационная логика: здесь нет обращения к ML-сервису.
export function mockResponse(path, payload) {
  if (!payload || typeof payload.description !== 'string' || !payload.description.trim()
    || payload.description.length > 12000
    || (payload.title !== undefined && (typeof payload.title !== 'string' || payload.title.length > 100))) {
    return { status: 400, body: { message: 'Укажите описание услуг (от 1 до 12 000 символов) и название до 100 символов.' } };
  }
  if (path === '/api/categories') {
    const text = `${payload.title ?? ''}\n${payload.description}`;
    return { status: 200, body: { categories: CATEGORIES.filter(category => patterns.get(category.id).test(text)) } };
  }
  if (!Array.isArray(payload.categories) || !payload.categories.length || payload.categories.some(category => !category
    || !['string', 'number'].includes(typeof category.id) || !String(category.id).trim()
    || typeof category.name !== 'string' || !category.name.trim())) {
    return { status: 400, body: { message: 'Выберите хотя бы одну категорию.' } };
  }
  const categories = [...new Map(payload.categories.map(category => [String(category.id), category])).values()];
  return {
    status: 200,
    body: {
      drafts: categories.map(category => ({
        categoryId: category.id,
        title: category.name,
        description: `Услуги: ${category.name.toLocaleLowerCase('ru-RU')}.\n\n${payload.description.trim()}`,
      })),
    },
  };
}

// RFC 4180 subset used only by the explicit local demo, not production input.
function csvItems(source) {
  const delimiter = source.replace(/^\uFEFF/, '').split(/[\r\n]/, 1)[0].includes(';') ? ';' : ',';
  const rows = [];
  let row = [], field = '', quoted = false, closed = false;
  const text = source.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (char === '"') { quoted = false; closed = true; }
      else field += char;
    } else if (char === delimiter || char === '\n' || char === '\r') {
      row.push(field); field = ''; closed = false;
      if (char !== delimiter) {
        if (char === '\r' && text[i + 1] === '\n') i++;
        if (row.some(value => value !== '')) rows.push(row);
        row = [];
      }
    } else if (char === '"' && !field && !closed) quoted = true;
    else {
      if (closed || char === '"') throw new Error('Проверьте кавычки CSV.');
      field += char;
    }
  }
  if (quoted) throw new Error('Незакрытые кавычки CSV.');
  if (field || row.length || closed) { row.push(field); rows.push(row); }
  const headers = rows.shift();
  if (headers && new Set(headers).size !== headers.length) throw new Error('В CSV повторяются названия столбцов.');
  if (!headers?.includes('description')) throw new Error('CSV должен содержать description.');
  return rows.map(values => {
    if (values.length !== headers.length) throw new Error('Число полей CSV не совпадает с заголовком.');
    return Object.fromEntries(headers.flatMap((key, i) => values[i] === '' ? [] : [[key, values[i]]]));
  });
}

export function mockCreation(source, contentType, generateText, delay) {
  try {
    const csv = /^(text|application)\/csv(?:;|$)/i.test(contentType);
    if (!csv && !/^application\/json(?:;|$)/i.test(contentType)) throw new Error('Используйте JSON или CSV.');
    const payload = csv ? csvItems(source) : JSON.parse(source.replace(/^\uFEFF/, ''));
    const batch = csv || Array.isArray(payload) || Array.isArray(payload?.items);
    const items = csv || Array.isArray(payload) ? payload : payload?.items ?? [payload];
    let mode = payload?.generateText ?? true;
    if (typeof mode !== 'boolean') throw new Error('generateText должен быть boolean.');
    if (generateText !== null) {
      if (!['true', 'false'].includes(generateText)) throw new Error('generateText должен быть boolean.');
      mode = generateText === 'true';
    }
    if (!Array.isArray(items) || !items.length || items.length > 100) throw new Error('Передайте от 1 до 100 объявлений.');
    for (const item of items) {
      if (!item || Object.keys(item).some(key => !['title', 'description', 'mcId', 'categories', 'testcase', 'generateText'].includes(key))
        || typeof item.description !== 'string' || !item.description.trim() || item.description.length > 12000
        || !CATEGORIES.some(c => c.id === Number(item.mcId ?? 101))
        || (item.title !== undefined && (typeof item.title !== 'string' || item.title.length > 100))
        || (item.testcase !== undefined && (typeof item.testcase !== 'string' || item.testcase.length > 200))
        || (item.categories !== undefined && (!Array.isArray(item.categories) || !item.categories.length
          || item.categories.some(c => !CATEGORIES.some(known => known.id === c?.id))
          || new Set(item.categories.map(c => c.id)).size !== item.categories.length))) {
        throw new Error('Проверьте description, mcId и категории объявления.');
      }
    }
    async function* events() {
      yield { type: 'start', total: items.length, generateText: mode };
      for (const [index, item] of items.entries()) {
        const identity = { index, ...(item.testcase !== undefined ? { testcase: item.testcase } : {}) };
        const categories = item.categories ?? mockResponse('/api/categories', item).body.categories.filter(c => c.id !== Number(item.mcId ?? 101));
        yield { type: 'categories', ...identity, categories };
        const drafts = [];
        for (const category of categories) {
          await new Promise(done => setTimeout(done, delay));
          const draft = { mcId: category.id, title: mode ? category.name : null, text: mode ? `Демонстрационный черновик.\n${item.description.trim()}` : null };
          drafts.push(draft);
          yield { type: 'drafts', ...identity, drafts: [draft], warnings: [] };
        }
        yield { type: 'item_done', ...identity, drafts, completed: index + 1, total: items.length };
      }
      yield { type: 'done', total: items.length, completed: items.length, failed: 0 };
    }
    return { status: 200, batch, events: events() };
  } catch (error) {
    return { status: 422, body: { message: error.message } };
  }
}
