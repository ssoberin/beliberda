import test from 'node:test';
import assert from 'node:assert/strict';
import { prepareSubmission } from '../src/submission.js';

test('единое описание: обычный текст передаётся вместе с названием как title', () => {
  assert.deepEqual(prepareSubmission('  Меняем смесители.  ', '  Сантехник  '), {
    payload: { title: 'Сантехник', description: 'Меняем смесители.', mcId: 101 },
    contentType: 'application/json', batch: false,
  });
  assert.equal(prepareSubmission('{Название услуги}', '', 'description.txt').payload.description, '{Название услуги}');
});

test('JSON распознаётся автоматически: название дополняет записи и сохраняет собственные title', () => {
  const item = prepareSubmission('{"description":"Трубы"}', 'Мастер');
  assert.deepEqual(item.payload, { description: 'Трубы', title: 'Мастер' });
  assert.equal(item.batch, false);
  const batch = prepareSubmission('[{"description":"Трубы"},{"description":"Плитка","title":"Плиточник"}]', 'Мастер');
  assert.equal(batch.batch, true);
  assert.deepEqual(batch.payload.map(row => row.title), ['Мастер', 'Плиточник']);
  const wrapped = prepareSubmission('{"items":[{"description":"Трубы"}],"generateText":false}', 'Мастер');
  assert.equal(wrapped.payload.items[0].title, 'Мастер');
  assert.equal(wrapped.payload.generateText, false);
});

test('CSV распознаётся по заголовку: BOM, кавычки, запятая и точка с запятой', () => {
  for (const csv of [
    '\uFEFFtestcase,description\r\ncase-001,"Меняем трубы, сифоны."',
    'testcase;description\ncase-001;Меняем трубы.',
    '"testcase","description"\n"case-001","Трубы"',
  ]) {
    const result = prepareSubmission(csv);
    assert.equal(result.contentType, 'text/csv');
    assert.equal(result.batch, true);
  }
});

test('название из формы дополняет CSV и mcId остаётся числом', () => {
  const result = prepareSubmission('testcase,description,title,mcId\ncase-001,"Меняем трубы, сифоны.",,104\ncase-002,Плитка,Плиточник,103', 'Мастер');
  assert.equal(result.contentType, 'application/json');
  assert.equal(result.batch, true);
  assert.deepEqual(result.payload, [
    { testcase: 'case-001', description: 'Меняем трубы, сифоны.', title: 'Мастер', mcId: 104 },
    { testcase: 'case-002', description: 'Плитка', title: 'Плиточник', mcId: 103 },
  ]);
});

test('ошибочный JSON и пустой либо слишком длинный текст не отправляются как описание', () => {
  assert.throws(() => prepareSubmission('{"description":}'), /Проверьте JSON/);
  assert.throws(() => prepareSubmission('не JSON', '', 'input.json'), /Проверьте JSON/);
  assert.throws(() => prepareSubmission('   '), /Введите описание/);
  assert.throws(() => prepareSubmission('x'.repeat(12001)), /12 000/);
});
