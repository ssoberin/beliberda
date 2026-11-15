import { CreationProgress } from './progress.js';
import { prepareSubmission } from './submission.js';
import { CATEGORIES } from './catalog.js';
import { createListings, getConnectionStatus } from './api.js';
import { resultsCsv } from './export.js';

const $ = id => document.getElementById(id);
const form = $('listing-form');
const description = $('description');
const button = $('generate-listings');
const groups = new Map();
let controller = null;
let sourceFile = null;
let connectionRetry;
let checkingConnection = false;
let batchRequest = false;
let draftCount = 0;
let progressState = null;
let progressTimer = null;
let progressStarted = 0;

function message(text = '', error = false) {
  $('generate-message').textContent = text;
  $('generate-message').dataset.kind = error ? 'error' : 'info';
  $('generate-message').hidden = !text;
}

function busy(active) {
  button.disabled = active;
  $('upload-zone').setAttribute('aria-busy', String(active));
  button.setAttribute('aria-busy', String(active));
  button.querySelector('span').textContent = active ? 'Создаём…' : 'Создать';
}

function stopProgressTimer() {
  clearInterval(progressTimer);
  progressTimer = null;
}

function renderProgress() {
  if (!progressState) return;
  const state = progressState.snapshot();
  const bar = $('creation-progress');
  bar.max = state.max;
  if (state.indeterminate) bar.removeAttribute('value');
  else bar.value = state.value;
  $('creation-stage').textContent = state.title;
  $('creation-count').textContent = state.count;
  $('creation-progress-panel').dataset.state = state.error ? 'error' : 'active';
  for (let step = 0; step < 3; step++) {
    const done = (progressState.finished && !state.error) || step < state.stage;
    $('creation-step-' + step).dataset.state = done ? 'complete' : step === state.stage ? 'active' : 'pending';
  }
}

function startProgress() {
  stopProgressTimer();
  progressState = new CreationProgress();
  progressStarted = performance.now();
  $('creation-elapsed').textContent = '0 с';
  $('creation-progress-panel').hidden = false;
  renderProgress();
  progressTimer = setInterval(() => {
    $('creation-elapsed').textContent = `${Math.floor((performance.now() - progressStarted) / 1000)} с`;
  }, 1000);
}

function invalidate() {
  stopProgressTimer();
  progressState = null;
  $('creation-progress-panel').hidden = true;
  controller?.abort();
  controller = null;
  busy(false);
  groups.clear();
  draftCount = 0;
  $('results').hidden = true;
  $('draft-list').replaceChildren();
  $('generation-warnings').replaceChildren();
  $('generation-warnings').hidden = true;
  message();
}

async function checkConnection() {
  if (checkingConnection) return;
  checkingConnection = true;
  clearTimeout(connectionRetry);
  const status = $('connection-status');
  $('check-connection').disabled = true;
  let retry = false;
  try {
    const connection = await getConnectionStatus(AbortSignal.timeout(10_000));
    retry = !connection.ready || !connection.categoryGeneration;
    status.dataset.kind = connection.mode === 'demo' ? 'warning' : connection.ready && connection.categoryGeneration ? 'ready' : 'error';
    status.textContent = connection.mode === 'demo'
      ? 'Демонстрационный режим: черновики тестовые, Qwen не используется.'
      : connection.ready ? (connection.categoryGeneration
        ? 'Обработка объявлений доступна.' : 'Автоматическое определение категорий пока недоступно.')
        : [connection.message || 'Генерация пока недоступна.', connection.detail].filter(Boolean).join(' ');
  } catch {
    retry = true;
    status.dataset.kind = 'error';
    status.textContent = 'Не удалось проверить подключение. Повторите проверку.';
  } finally {
    checkingConnection = false;
    $('check-connection').disabled = false;
    if (retry) connectionRetry = setTimeout(checkConnection, 5_000);
  }
}
$('check-connection').addEventListener('click', checkConnection);

function updateDescriptionCount() {
  $('description-count').textContent = `${description.value.length.toLocaleString('ru-RU')} символов`;
}
for (const input of [$('title'), description]) {
  input.addEventListener('input', () => {
    if (input === description) {
      sourceFile = null;
      $('batch-file').value = '';
      $('clear-file').hidden = true;
      $('upload-message').hidden = true;
    }
    invalidate();
    description.setCustomValidity('');
    $('clear-title').hidden = !$('title').value;
    updateDescriptionCount();
  });
}
$('clear-title').addEventListener('click', () => {
  $('title').value = '';
  $('title').dispatchEvent(new Event('input'));
  $('title').focus();
});
$('clear-file').addEventListener('click', () => {
  sourceFile = null;
  $('batch-file').value = '';
  description.disabled = false;
  description.setCustomValidity('');
  updateDescriptionCount();
  $('upload-message').hidden = true;
  $('clear-file').hidden = true;
  invalidate();
  description.focus();
});
function loadFile(file) {
  if (!file) return;
  invalidate();
  sourceFile = file;
  description.value = '';
  description.disabled = true;
  description.setCustomValidity('');
  updateDescriptionCount();
  $('clear-file').hidden = false;
  const status = $('upload-message');
  status.textContent = `Выбран ${file.name}. Нажмите «Создать», чтобы загрузить и обработать файл.`;
  status.dataset.kind = 'info';
  status.hidden = false;
}
$('batch-file').addEventListener('change', () => loadFile($('batch-file').files[0]));

const uploadZone = $('upload-zone');
let dragDepth = 0;
const isFileDrag = event => Array.from(event.dataTransfer?.types ?? []).includes('Files');
uploadZone.addEventListener('dragenter', event => {
  if (!isFileDrag(event)) return;
  event.preventDefault();
  dragDepth += 1;
  uploadZone.classList.add('is-dragging');
});
uploadZone.addEventListener('dragover', event => {
  if (!isFileDrag(event)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = 'copy';
});
uploadZone.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) uploadZone.classList.remove('is-dragging');
});
uploadZone.addEventListener('drop', event => {
  event.preventDefault();
  dragDepth = 0;
  uploadZone.classList.remove('is-dragging');
  const files = event.dataTransfer?.files;
  if (!files?.length) return;
  if (files.length !== 1) {
    const status = $('upload-message');
    status.textContent = 'Перетащите один файл JSON, CSV или TXT. Внутри JSON/CSV может быть до 300 объявлений.';
    status.dataset.kind = 'error';
    status.hidden = false;
    return;
  }
  loadFile(files[0]);
});

function groupFor(event) {
  let group = groups.get(event.index);
  if (group) return group;
  const section = document.createElement('section');
  section.className = 'result-item';
  const heading = document.createElement('h3');
  heading.textContent = event.testcase ?? `Объявление ${event.index + 1}`;
  const status = document.createElement('p');
  status.className = 'hint item-status';
  status.setAttribute('role', 'status');
  const cards = document.createElement('div');
  section.append(heading, status, cards);
  const following = [...groups.entries()].sort((a, b) => a[0] - b[0]).find(([index]) => index > event.index);
  $('draft-list').insertBefore(section, following?.[1].section ?? null);
  group = { section, status, cards, microcats:null, drafts: [], testcase: event.testcase, index: event.index, result: null };
  groups.set(event.index, group);
  $('results').hidden = false;
  return group;
}

function appendDraft(group, draft) {
  const card = document.createElement('article');
  card.className = 'draft';
  const category = document.createElement('p');
  category.className = 'draft-category';
  category.textContent = CATEGORIES.find(c => c.id === draft.mcId)?.name ?? `Категория ${draft.mcId}`;
  card.append(category);
  const entry = { ...draft };
  if (draft.title === null && draft.text === null) {
    const hint = document.createElement('p');
    hint.className = 'hint';
    hint.textContent = 'Категория подобрана. Генерация заголовка и текста отключена.';
    card.append(hint);
  } else {
    const titleLabel = document.createElement('label');
    titleLabel.htmlFor = `draft-title-${group.index}-${draft.mcId}`;
    titleLabel.textContent = 'Название объявления';
    const input = document.createElement('input');
    input.id = titleLabel.htmlFor;
    input.value = draft.title ?? '';
    const textLabel = document.createElement('label');
    textLabel.htmlFor = `draft-text-${group.index}-${draft.mcId}`;
    textLabel.textContent = 'Описание';
    const textarea = document.createElement('textarea');
    textarea.id = textLabel.htmlFor;
    textarea.rows = 6;
    textarea.value = draft.text ?? '';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'button button-small copy-button';
    copy.textContent = 'Скопировать текст';
    const status = document.createElement('span');
    status.className = 'hint copy-status';
    status.setAttribute('role', 'status');
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(`${input.value}\n\n${textarea.value}`);
        status.textContent = 'Скопировано';
      } catch { status.textContent = 'Выделите текст и скопируйте вручную'; }
    });
    input.addEventListener('input', () => { entry.title = input.value; status.textContent = ''; });
    textarea.addEventListener('input', () => { entry.text = textarea.value; status.textContent = ''; });
    card.append(titleLabel, input, textLabel, textarea, copy, status);
  }
  group.drafts.push(entry);
  group.cards.append(card);
  draftCount += 1;
  $('result-count').textContent = String(draftCount);
}

function warning(text) {
  const line = document.createElement('p');
  line.textContent = text;
  $('generation-warnings').append(line);
  $('generation-warnings').hidden = false;
}

function handleEvent(event) {
  progressState?.update(event);
  renderProgress();
  if (event.type === 'done') stopProgressTimer();
  if (event.type === 'start') {
    batchRequest = event.batch ?? event.total > 1;
    message(`Обрабатываем объявлений: ${event.total}. Ответы появятся по мере готовности.`);
    return;
  }
  if (event.type === 'heartbeat') return;
  if (event.type === 'done') {
    message(`Обработано: ${event.completed} из ${event.total}. Черновиков: ${draftCount}.${event.failed ? ` С ошибками: ${event.failed}.` : ''}`, event.failed > 0);
    return;
  }
  const group = groupFor(event);
  if (event.type === 'categories') {
    group.microcats = event.categories.map(category => category.id);
    group.status.textContent = event.categories.length
      ? `Категории: ${event.categories.map(c => c.name).join(', ')}. Готовим результат…`
      : 'Оснований для дополнительных объявлений не найдено.';
  } else if (event.type === 'drafts') {
    event.drafts.forEach(draft => appendDraft(group, draft));
    event.warnings?.forEach(w => warning(`${group.testcase ?? `Объявление ${group.index + 1}`}: ${w.message}`));
  } else if (event.type === 'category_error') {
    warning(`${group.testcase ?? `Объявление ${group.index + 1}`}, категория ${event.mcId}: ${event.message}`);
  } else if (event.type === 'item_done') {
    if (Array.isArray(event.microcats)) group.microcats = event.microcats;
    group.result = event;
    group.status.textContent = event.error?.message ?? (event.errors?.length
      ? `Готово частично. Черновиков: ${group.drafts.length}. Ошибок: ${event.errors.length}.`
      : group.drafts.length ? `Готово. Черновиков: ${group.drafts.length}.` : 'Оснований для дополнительных объявлений не найдено.');
    if (event.error || event.errors?.length) group.status.dataset.kind = 'error';
    message(`Обработано: ${event.completed} из ${event.total}. Получено черновиков: ${draftCount}.`);
  }
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  let submission;
  try {
    submission = prepareSubmission(description.value, $('title').value, sourceFile);
  } catch (error) {
    message(error.message, true);
    return;
  }
  const { payload } = submission;
  batchRequest = false;
  invalidate();
  $('result-count').textContent = '0';
  const request = new AbortController();
  controller = request;
  busy(true);
  startProgress();
  message('Начинаем обработку…');
  try {
    await createListings(payload, {
      signal: request.signal, generateText: true,
      onEvent: event => { if (controller === request && !request.signal.aborted) handleEvent(event); },
    });
  } catch (error) {
    if (!request.signal.aborted) {
      progressState.interrupted = true;
      renderProgress();
      message(error.message, true);
      checkConnection();
    }
  } finally {
    if (controller === request) { stopProgressTimer(); controller = null; busy(false); }
  }
});

function exportResults() {
  return [...groups.values()].sort((a, b) => a.index - b.index).map(group => ({
    index: group.index, ...(group.testcase !== undefined ? { testcase: group.testcase } : {}),
    microcats:group.microcats, drafts: group.drafts,
    ...(group.result?.error ? { error: group.result.error } : {}),
    ...(group.result?.errors ? { errors: group.result.errors } : {}),
    ...(!group.result ? { incomplete: true } : {}),
  }));
}
function download(content, type, name) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('download-json').addEventListener('click', () => {
  const items = exportResults();
  const body = !batchRequest && items.length === 1 ? (({ index, ...rest }) => rest)(items[0]) : { results: items };
  download(JSON.stringify(body, null, 2), 'application/json;charset=utf-8', 'drafts.json');
});
$('download-csv').addEventListener('click', () => {
  download(resultsCsv(exportResults()), 'text/csv;charset=utf-8', 'drafts.csv');
});

checkConnection();
