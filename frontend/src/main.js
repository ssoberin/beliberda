import { CATEGORIES } from './catalog.js';
import { generateCategories, generateListings } from './api.js';

const $ = id => document.getElementById(id);
const form = $('listing-form');
const title = $('title');
const description = $('description');
const categoryList = $('category-list');
const results = $('results');
let categories = [...CATEGORIES];
let selected = new Set();
const requests = { categories: null, generate: null };
const buttons = { categories: $('generate-categories'), generate: $('generate-listings') };
const labels = { categories: 'Сгенерировать категории', generate: 'Сгенерировать объявления' };

function message(kind, text = '', error = false) {
  const element = $(kind === 'categories' ? 'categories-message' : 'generate-message');
  element.textContent = text;
  element.dataset.kind = error ? 'error' : 'info';
  element.hidden = !text;
}

function busy(kind, active) {
  buttons[kind].disabled = active;
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
}

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
  return { title: title.value.trim(), description: description.value.trim() };
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
      ? 'Категории выбраны автоматически. При необходимости измените выбор.'
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

function renderDrafts(drafts, chosen) {
  const elements = drafts.map((draft, index) => {
    const card = document.createElement('article');
    card.className = 'draft';
    const category = document.createElement('p');
    category.className = 'draft-category';
    category.textContent = chosen.find(item => String(item.id) === String(draft.categoryId))?.name ?? `Категория ${draft.categoryId}`;
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
    card.append(category, titleLabel, draftTitle, textLabel, draftText, copy, status);
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
  try {
    const drafts = await generateListings({ ...sourcePayload(), categories: chosen }, controller.signal);
    if (controller.signal.aborted) return;
    renderDrafts(drafts, chosen);
    message('generate', drafts.length ? '' : 'Сервер не вернул объявления. Попробуйте уточнить описание.');
  } catch (error) {
    if (!controller.signal.aborted) message('generate', error.message, true);
  } finally {
    if (requests.generate === controller) {
      requests.generate = null;
      busy('generate', false);
    }
  }
});

renderCategories();
