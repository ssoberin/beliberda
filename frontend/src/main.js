import { CATEGORIES } from './catalog.js';
import { generateCategories, generateListings, getConnectionStatus } from './api.js';

const $ = id => document.getElementById(id);
const form = $('listing-form');
const title = $('title');
const description = $('description');
const sourceCategory = $('source-category');
const categoryList = $('category-list');
const results = $('results');
let categories = [...CATEGORIES];
let selected = new Set();
const requests = { categories: null, generate: null };
const buttons = { categories: $('generate-categories'), generate: $('generate-listings') };
const labels = { categories: 'Сгенерировать категории', generate: 'Сгенерировать черновики' };
let categoryGenerationAvailable = true;
let connectionRetry = null;
let checkingConnection = false;

function message(kind, text = '', error = false) {
  const element = $(kind === 'categories' ? 'categories-message' : 'generate-message');
  element.textContent = text;
  element.dataset.kind = error ? 'error' : 'info';
  element.hidden = !text;
}

function busy(kind, active) {
  buttons[kind].disabled = active || (kind === 'categories' && !categoryGenerationAvailable);
  buttons[kind].setAttribute('aria-busy', String(active));
  buttons[kind].querySelector('span').textContent = active
    ? (kind === 'categories' ? 'Подбираем категории…' : 'Готовим объявления…')
    : labels[kind];
}

function cancel(kind) {
  requests[kind]?.abort();
  requests[kind] = null;
  busy(kind, false);
  message(kind);
}

function invalidateDrafts() {
  cancel('generate');
  results.hidden = true;
  $('draft-list').replaceChildren();
  $('generation-warnings').hidden = true;
  $('generation-warnings').replaceChildren();
}

async function checkConnection() {
  if (checkingConnection) return;
  checkingConnection = true;
  clearTimeout(connectionRetry);
  connectionRetry = null;
  let retry = false;
  const button = $('check-connection');
  const status = $('connection-status');
  button.disabled = true;
  status.textContent = 'Проверяем подключение к генерации…';
  try {
    const connection = await getConnectionStatus(AbortSignal.timeout(10_000));
    retry = !connection.ready;
    categoryGenerationAvailable = connection.categoryGeneration === true;
    buttons.categories.disabled = !categoryGenerationAvailable || Boolean(requests.categories);
    buttons.categories.title = categoryGenerationAvailable ? 'Подбор категорий по описанию с помощью TF-IDF' : 'Автоподбор категорий пока недоступен';
    $('category-hint').textContent = categoryGenerationAvailable
      ? connection.mode === 'demo' ? 'Демонстрационный подбор категорий по ключевым словам.' : 'TF-IDF может ошибаться. Проверьте выбор вручную.'
      : 'Выберите категории вручную. Автоподбор пока недоступен.';
    status.dataset.kind = connection.mode === 'demo' ? 'warning' : connection.ready ? 'ready' : 'error';
    status.textContent = connection.mode === 'demo'
      ? 'Демонстрационный режим: черновики тестовые, Qwen не используется.'
      : connection.ready ? `Qwen подключён${connection.model ? `: ${connection.model}` : ''}`
        : [connection.message || 'Генерация пока недоступна. Проверьте подключение.', connection.detail].filter(Boolean).join(' ');
  } catch {
    retry = true;
    status.dataset.kind = 'error';
    status.textContent = 'Не удалось проверить подключение. Повторите проверку.';
  } finally {
    checkingConnection = false;
    button.disabled = false;
    if (retry) connectionRetry = setTimeout(checkConnection, 5_000);
  }
}

$('check-connection').addEventListener('click', checkConnection);

function renderCategories() {
  categoryList.replaceChildren(...categories.map(category => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'category-chip';
    button.textContent = category.name;
    button.setAttribute('aria-pressed', String(selected.has(String(category.id))));
    button.addEventListener('click', () => {
      cancel('categories');
      invalidateDrafts();
      const id = String(category.id);
      if (selected.has(id)) selected.delete(id);
      else selected.add(id);
      renderCategories();
      categoryList.children[categories.indexOf(category)]?.focus();
    });
    return button;
  }));
  $('selection-count').textContent = selected.size ? `Выбрано категорий: ${selected.size}` : 'Категории пока не выбраны';
  $('reset-categories').hidden = selected.size === 0;
}

function sourcePayload() {
  return { title: title.value.trim(), description: description.value.trim(), mcId: Number(sourceCategory.value) };
}

function validateDescription() {
  description.setCustomValidity(description.value.trim() ? '' : 'Расскажите об услугах, которые вы предлагаете.');
  return description.reportValidity();
}

for (const input of [title, description]) {
  input.addEventListener('input', () => {
    cancel('categories');
    invalidateDrafts();
    description.setCustomValidity('');
    $('clear-title').hidden = !title.value;
    $('description-count').textContent = `${description.value.length.toLocaleString('ru-RU')} / 12 000`;
  });
}

sourceCategory.addEventListener('change', () => {
  cancel('categories');
  invalidateDrafts();
  selected.clear();
  renderCategories();
});

$('clear-title').addEventListener('click', () => {
  title.value = '';
  title.dispatchEvent(new Event('input'));
  title.focus();
});

$('reset-categories').addEventListener('click', () => {
  cancel('categories');
  invalidateDrafts();
  selected.clear();
  renderCategories();
  categoryList.querySelector('button')?.focus();
});

buttons.categories.addEventListener('click', async () => {
  if (!categoryGenerationAvailable) return;
  if (!validateDescription()) return;
  cancel('categories');
  cancel('generate');
  const controller = new AbortController();
  requests.categories = controller;
  busy('categories', true);
  message('categories', 'Подбираем категории по вашему описанию…');
  try {
    const response = await generateCategories(sourcePayload(), controller.signal);
    if (controller.signal.aborted) return;
    const merged = new Map(CATEGORIES.map(category => [String(category.id), category]));
    response.forEach(category => merged.set(String(category.id), category));
    categories = [...merged.values()];
    selected = new Set(response.map(category => String(category.id)));
    invalidateDrafts();
    renderCategories();
    message('categories', response.length
      ? 'Категории подобраны автоматически. Проверьте и при необходимости измените выбор.'
      : 'Подходящих категорий не найдено. Уточните описание или выберите категории вручную.');
  } catch (error) {
    if (!controller.signal.aborted) message('categories', error.message, true);
  } finally {
    if (requests.categories === controller) {
      requests.categories = null;
      busy('categories', false);
    }
  }
});

function renderDrafts(drafts, chosen, warnings = []) {
  const elements = drafts.map((draft, index) => {
    const card = document.createElement('article');
    card.className = 'draft';
    const category = document.createElement('p');
    category.className = 'draft-category';
    category.textContent = chosen.find(item => String(item.id) === String(draft.categoryId))?.name ?? `Категория ${draft.categoryId}`;
    const warning = warnings.find(item => String(item.categoryId) === String(draft.categoryId));
    const review = document.createElement('p');
    review.className = 'hint draft-review';
    review.textContent = warning ? 'Заготовка: уточните сведения об услуге перед публикацией.' : '';
    review.hidden = !warning;
    const titleLabel = document.createElement('label');
    titleLabel.htmlFor = `draft-title-${index}`;
    titleLabel.textContent = 'Название объявления';
    const draftTitle = document.createElement('input');
    draftTitle.id = titleLabel.htmlFor;
    draftTitle.value = draft.title;
    const textLabel = document.createElement('label');
    textLabel.htmlFor = `draft-description-${index}`;
    textLabel.textContent = 'Описание';
    const draftText = document.createElement('textarea');
    draftText.id = textLabel.htmlFor;
    draftText.value = draft.description;
    draftText.rows = 6;
    const copy = document.createElement('button');
    copy.className = 'button button-small copy-button';
    copy.type = 'button';
    copy.textContent = 'Скопировать текст';
    const status = document.createElement('span');
    status.className = 'hint copy-status';
    status.setAttribute('role', 'status');
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(`${draftTitle.value}\n\n${draftText.value}`);
        status.textContent = 'Скопировано';
      } catch {
        status.textContent = 'Выделите текст и скопируйте вручную';
      }
    });
    draftTitle.addEventListener('input', () => { status.textContent = ''; });
    draftText.addEventListener('input', () => { status.textContent = ''; });
    card.append(category, review, titleLabel, draftTitle, textLabel, draftText, copy, status);
    return card;
  });
  $('draft-list').replaceChildren(...elements);
  $('result-count').textContent = String(drafts.length);
  results.hidden = drafts.length === 0;
  if (drafts.length) {
    $('results-heading').focus({ preventScroll: true });
    results.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
  }
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!validateDescription()) return;
  if (requests.categories) {
    message('generate', 'Дождитесь подбора категорий или выберите их вручную.', true);
    return;
  }
  if (!selected.size) {
    message('generate', 'Выберите хотя бы одну категорию или сгенерируйте их по описанию.', true);
    categoryList.querySelector('button')?.focus();
    return;
  }
  cancel('generate');
  const controller = new AbortController();
  requests.generate = controller;
  const chosen = categories.filter(category => selected.has(String(category.id)));
  busy('generate', true);
  message('generate', 'Готовим черновики объявлений…');
  $('generation-warnings').hidden = true;
  try {
    const { drafts, warnings } = await generateListings({ ...sourcePayload(), categories: chosen }, controller.signal);
    if (controller.signal.aborted) return;
    renderDrafts(drafts, chosen, warnings);
    const warningBox = $('generation-warnings');
    const warningLines = warnings.map(warning => {
      const line = document.createElement('p');
      const category = chosen.find(item => String(item.id) === String(warning.categoryId));
      line.textContent = `${category?.name ?? `Категория ${warning.categoryId}`}: ${warning.message}`;
      return line;
    });
    warningBox.replaceChildren(...warningLines);
    warningBox.hidden = warnings.length === 0;
    message('generate', drafts.length ? `Создано черновиков: ${drafts.length} из ${chosen.length}.` : 'Сервер не вернул объявления. Попробуйте уточнить описание.');
  } catch (error) {
    if (!controller.signal.aborted) message('generate', error.message, true);
    if (!controller.signal.aborted) checkConnection();
  } finally {
    if (requests.generate === controller) {
      requests.generate = null;
      busy('generate', false);
    }
  }
});

renderCategories();
checkConnection();
