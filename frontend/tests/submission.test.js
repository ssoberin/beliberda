import test from 'node:test';
import assert from 'node:assert/strict';
import { prepareSubmission } from '../src/submission.js';

test('введённый текст, JSON и CSV уходят без разбора и изменения содержимого', () => {
  for (const source of [
    '  Меняем смесители.  ',
    '[{"description":"Трубы"},{"description":"Плитка","title":"Плиточник"}]',
    '\uFEFFtestcase,description\r\ncase-001,"Меняем трубы, сифоны."',
    '{"description":}',
  ]) {
    const { payload } = prepareSubmission(source, ' Общее название ');
    assert.ok(payload instanceof FormData);
    assert.equal(payload.get('text'), source);
    assert.equal(payload.get('title'), ' Общее название ');
    assert.equal(payload.has('file'), false);
  }
});

test('исходный файл передаётся как File, браузер не читает его содержимое', () => {
  const file = new File(['invalid JSON checked by backend'], 'input.json', { type: 'application/octet-stream' });
  file.text = file.arrayBuffer = file.stream = () => assert.fail('Браузер не должен читать файл');
  const { payload } = prepareSubmission('не используется при выбранном файле', 'Мастер', file);
  assert.equal(payload.get('file'), file);
  assert.equal(payload.get('file').name, 'input.json');
  assert.equal(payload.get('title'), 'Мастер');
  assert.equal(payload.has('text'), false);
});

test('пустая форма без файла отклоняется, пустой файл проверит бэкенд', () => {
  assert.throws(() => prepareSubmission('   '), /Введите описание/);
  const file = new File([], 'empty.csv');
  assert.equal(prepareSubmission('', '', file).payload.get('file'), file);
});
