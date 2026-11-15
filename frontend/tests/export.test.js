import test from 'node:test';
import assert from 'node:assert/strict';
import { resultsCsv } from '../src/export.js';

test('ошибка генерации сохраняет все предсказанные классы и позиции описаний', () => {
  const csv=resultsCsv([{ index:0, testcase:'partial', microcats:[102,104],
    drafts:[{ mcId:104,text:'Меняем трубы.' }], errors:[{mcId:102,message:'Qwen timeout'}] }]);
  assert.ok(csv.includes('"[102,104]"'));
  assert.ok(csv.includes('"[null,""Меняем трубы.""]"'));
  assert.ok(csv.includes('Qwen timeout'));
});

test('нулевой результат классификации отличается от неизвестного класса', () => {
  const csv=resultsCsv([
    { index:0,microcats:[],drafts:[] },
    { index:1,microcats:null,drafts:[],error:{message:'Invalid input'} },
  ]);
  assert.ok(csv.includes('"case-001","[]","[]","null"'));
  assert.ok(csv.includes('"case-002","null","null"'));
});

test('классы сохраняются, даже если ни один текст не создан', () => {
  const csv=resultsCsv([{index:0,microcats:[105],drafts:[],errors:[{mcId:105,message:'timeout'}]}]);
  assert.ok(csv.includes('"[105]","[null]"'));
});
