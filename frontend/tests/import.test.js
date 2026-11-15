import test from 'node:test';
import assert from 'node:assert/strict';
import { parseImport } from '../src/import.js';
import { createListings } from '../src/create.js';

test('CSV формата testcase,description: отдельный запрос для каждой строки', () => {
  const records = parseImport('testcase,description\r\ncase-001,"Штукатурим стены, меняем смесители."\r\ncase-002,"Собираю шкафы."\r\n', 'example-input.csv');
  assert.deepEqual(records, [
    { testcase: 'case-001', title: '', description: 'Штукатурим стены, меняем смесители.' },
    { testcase: 'case-002', title: '', description: 'Собираю шкафы.' },
  ]);
});

test('CSV: BOM, многострочный текст, двойные кавычки и пустые строки', () => {
  const [record] = parseImport('\uFEFFtestcase,description\n\ncase-001,"Название ""Мастер""\nРаботаем, быстро."\n', 'INPUT.CSV');
  assert.equal(record.description, 'Название "Мастер"\nРаботаем, быстро.');
  assert.equal(record.testcase, 'case-001');
});

test('CSV: разделитель точка с запятой и необязательные title, mcId', () => {
  assert.deepEqual(parseImport('description;title;mcId\nМеняю трубы;Сантехник;104', 'input.csv'), [
    { testcase: 'Запрос 1', description: 'Меняю трубы', title: 'Сантехник', mcId: 104 },
  ]);
});

test('JSON: одиночный объект и массив, нормализация категорий', () => {
  assert.deepEqual(parseImport('{"description":"  Ремонт  "}', 'input.json'), [
    { testcase: 'Запрос 1', title: '', description: 'Ремонт' },
  ]);
  const records = parseImport(JSON.stringify([
    { testcase: 'one', description: 'Ремонт', mcId: 101, categories: [{ id: 104 }] },
    { description: 'Плитка', categories: [103] },
  ]), 'input.json');
  assert.deepEqual(records[0].categories, [{ id: 104, name: 'Сантехника' }]);
  assert.deepEqual(records[1].categories, [{ id: 103, name: 'Плиточные работы' }]);
});

test('импорт отклоняет некорректные файлы до отправки запросов', () => {
  const cases = [
    ['{', 'input.json', /синтаксис/],
    ['[]', 'input.json', /нет запросов/],
    ['null', 'input.json', /объект/],
    ['{"description":" "}', 'input.json', /description/],
    ['{"description":"Ремонт","mcId":999}', 'input.json', /mcId/],
    ['{"description":"Ремонт","categories":[999]}', 'input.json', /неизвестный ID/],
    ['{"description":"Ремонт","categories":[104,104]}', 'input.json', /повторяться/],
    ['{"description":"Ремонт","title":42}', 'input.json', /title/],
    [JSON.stringify({ description: 'a'.repeat(12001) }), 'input.json', /12 000/],
    [JSON.stringify(Array.from({ length: 101 }, () => ({ description: 'Ремонт' }))), 'input.json', /100/],
    ['testcase\ncase-001', 'input.csv', /столбец description/],
    ['description,description\nA,B', 'input.csv', /повторяются/],
    ['testcase,description\ncase-001,"Текст', 'input.csv', /не закрыты/],
    ['description\n"Текст"x', 'input.csv', /кавычки/],
    ['testcase,description\ncase-001,Текст,Лишнее', 'input.csv', /число полей/],
    ['Текст', 'input.txt', /\.json или .csv/],
  ];
  for (const [text, name, expected] of cases) assert.throws(() => parseImport(text, name), expected);
});

test('«Создать» последовательно подбирает категории и создаёт черновики', async () => {
  const calls = [];
  const categories = [{ id: 104, name: 'Сантехника' }];
  const drafts = [{ categoryId: 104, title: 'Сантехник', description: 'Меняю трубы' }];
  const source = { title: '', description: 'Меняю трубы', mcId: 101 };
  const result = await createListings(source, {
    classify: async payload => { calls.push('categories'); assert.deepEqual(payload, source); return categories; },
    onCategories: values => { calls.push('selected'); assert.deepEqual(values, categories); },
    generate: async payload => { calls.push('generate'); assert.deepEqual(payload, { ...source, categories }); return { drafts, warnings: [] }; },
  });
  assert.deepEqual(calls, ['categories', 'selected', 'generate']);
  assert.deepEqual(result, { categories, drafts, warnings: [] });
});

test('ручные категории работают без TF-IDF и не запускают автоподбор', async () => {
  const categories = [{ id: 104, name: 'Сантехника' }];
  const result = await createListings({ description: 'Трубы', categories }, {
    categoryGenerationAvailable: false,
    classify: () => assert.fail('Автоподбор не нужен'),
    generate: async payload => { assert.deepEqual(payload.categories, categories); return { drafts: [], warnings: [] }; },
  });
  assert.deepEqual(result.categories, categories);
});

test('пустой подбор и недоступная TF-IDF не запускают генерацию', async () => {
  const generate = () => assert.fail('Без категорий генерации быть не должно');
  await assert.rejects(createListings({ description: 'Мебель' }, { classify: async () => [], generate }), /не найдено/);
  await assert.rejects(createListings({ description: 'Трубы' }, { categoryGenerationAvailable: false, generate }), /Выберите категории вручную/);
});

test('изменение исходника отменяет цепочку между подбором и генерацией', async () => {
  const controller = new AbortController();
  await assert.rejects(createListings({ description: 'Трубы' }, {
    signal: controller.signal,
    classify: async () => { controller.abort(); return [{ id: 104, name: 'Сантехника' }]; },
    onCategories: () => assert.fail('Устаревший выбор не должен применяться'),
    generate: () => assert.fail('Отменённая цепочка не должна продолжаться'),
  }), { name: 'AbortError' });
});
