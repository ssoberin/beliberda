import { CATEGORIES } from './catalog.js';

export const MAX_IMPORT_BYTES = 2 * 1024 * 1024;
export const MAX_IMPORT_REQUESTS = 100;

export function parseCsv(text) {
  const delimiter = text.slice(0, text.search(/[\r\n]|$/)).includes(';') ? ';' : ',';
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  let closed = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (char === '"') { quoted = false; closed = true; }
      else field += char;
    } else if (char === delimiter || char === '\r' || char === '\n') {
      row.push(field);
      field = '';
      closed = false;
      if (char !== delimiter) {
        if (row.some(value => value.trim())) rows.push(row);
        row = [];
        if (char === '\r' && text[i + 1] === '\n') i++;
      }
    } else if (char === '"' && !field && !closed) {
      quoted = true;
    } else {
      if (closed || char === '"') throw new Error('Некорректные кавычки в CSV.');
      field += char;
    }
  }
  if (quoted) throw new Error('В CSV не закрыты кавычки.');
  row.push(field);
  if (row.some(value => value.trim())) rows.push(row);
  const headers = rows.shift()?.map(value => value.trim()) ?? [];
  if (!headers.includes('description')) throw new Error('В CSV нужен столбец description.');
  if (new Set(headers).size !== headers.length) throw new Error('В CSV повторяются названия столбцов.');
  return rows.map((values, index) => {
    if (values.length !== headers.length) throw new Error(`Запись ${index + 1}: число полей не совпадает с заголовком CSV.`);
    return Object.fromEntries(headers.map((header, i) => [header, values[i]]));
  });
}

export function parseImport(text, filename) {
  text = text.replace(/^\uFEFF/, '');
  let records;
  if (/\.csv$/i.test(filename)) records = parseCsv(text);
  else if (/\.json$/i.test(filename)) {
    try { records = JSON.parse(text); }
    catch { throw new Error('Не удалось прочитать JSON. Проверьте синтаксис файла.'); }
    if (!Array.isArray(records)) records = [records];
  } else throw new Error('Выберите файл .json или .csv.');
  if (!records.length) throw new Error('В файле нет запросов.');
  if (records.length > MAX_IMPORT_REQUESTS) throw new Error(`Можно загрузить не более ${MAX_IMPORT_REQUESTS} запросов за раз.`);
  return records.map((record, index) => {
    const fail = message => { throw new Error(`Запись ${index + 1}: ${message}`); };
    if (!record || typeof record !== 'object' || Array.isArray(record)) fail('ожидается объект запроса.');
    if (typeof record.description !== 'string' || !record.description.trim()) fail('нужно непустое поле description.');
    if (record.description.trim().length > 12000) fail('description длиннее 12 000 символов.');
    if (record.title !== undefined && (typeof record.title !== 'string' || record.title.length > 100)) fail('title должен быть строкой до 100 символов.');
    if (record.testcase !== undefined && (typeof record.testcase !== 'string' || record.testcase.length > 200)) fail('testcase должен быть строкой до 200 символов.');
    const mcId = record.mcId === undefined || record.mcId === '' ? undefined : Number(record.mcId);
    if (mcId !== undefined && !CATEGORIES.some(category => category.id === mcId)) fail('mcId должен быть от 101 до 106.');
    let categories;
    if (record.categories !== undefined) {
      if (!Array.isArray(record.categories)) fail('categories должен быть массивом категорий.');
      categories = record.categories.map(value => {
        const category = CATEGORIES.find(item => item.id === Number(value?.id ?? value));
        if (!category) fail('неизвестный ID категории.');
        return category;
      });
      if (new Set(categories.map(category => category.id)).size !== categories.length) fail('категории не должны повторяться.');
    }
    return {
      testcase: record.testcase?.trim() || `Запрос ${index + 1}`,
      title: record.title?.trim() ?? '',
      description: record.description.trim(),
      ...(mcId === undefined ? {} : { mcId }),
      ...(categories === undefined ? {} : { categories }),
    };
  });
}
