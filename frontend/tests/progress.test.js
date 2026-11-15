import test from 'node:test';
import assert from 'node:assert/strict';
import { CreationProgress } from '../src/progress.js';

test('один исходник: прогресс учитывает каждую готовую категорию до завершения запроса', () => {
  const progress = new CreationProgress();
  assert.equal(progress.snapshot().indeterminate, true);
  progress.update({ type: 'start', total: 1 });
  progress.update({ type: 'categories', index: 0, categories: [{ id: 103 }, { id: 104 }] });
  let state = progress.update({ type: 'drafts', index: 0 });
  assert.equal(state.stage, 1);
  assert.equal(state.value, 1);
  assert.equal(state.max, 2);
  assert.equal(state.count, 'Категории: 1 из 2');
  progress.update({ type: 'drafts', index: 0 });
  progress.update({ type: 'item_done', index: 0 });
  state = progress.update({ type: 'done', failed: 0 });
  assert.equal(state.value, state.max);
  assert.equal(state.stage, 2);
  assert.equal(state.title, 'Готово');
});

test('пакет: готовые категории одного исходника не считаются завершением других', () => {
  const progress = new CreationProgress();
  progress.update({ type: 'start', total: 3 });
  progress.update({ type: 'categories', index: 1, categories: [{ id: 104 }] });
  progress.update({ type: 'drafts', index: 1 });
  assert.equal(progress.snapshot().value, 0);
  progress.update({ type: 'item_done', index: 1 });
  progress.update({ type: 'item_done', index: 1 });
  assert.equal(progress.snapshot().value, 1);
  assert.equal(progress.snapshot().count, 'Объявления: 1 из 3');
});

test('частичная ошибка категории завершает её обработку и отмечает итог с ошибками', () => {
  const progress = new CreationProgress();
  progress.update({ type: 'start', total: 1 });
  progress.update({ type: 'categories', index: 0, categories: [{ id: 103 }, { id: 104 }] });
  progress.update({ type: 'category_error', index: 0 });
  progress.update({ type: 'drafts', index: 0 });
  progress.update({ type: 'item_done', index: 0 });
  const state = progress.update({ type: 'done', failed: 1 });
  assert.equal(state.title, 'Завершено с ошибками');
  assert.equal(state.error, true);
  assert.equal(state.value, 2);
});

test('обрыв останавливает процесс без ложного завершения, в том числе до первого события', () => {
  const progress = new CreationProgress();
  progress.interrupted = true;
  assert.equal(progress.snapshot().indeterminate, false);
  assert.equal(progress.snapshot().title, 'Обработка прервана');
  const partial = new CreationProgress();
  partial.update({ type: 'start', total: 2 });
  partial.update({ type: 'item_done', index: 1 });
  partial.interrupted = true;
  assert.equal(partial.snapshot().value, 1);
  assert.equal(partial.snapshot().max, 2);
  assert.equal(partial.snapshot().error, true);
});
