import { generateCategories, generateListings } from './api.js';

// One action: resolve categories first, then create drafts using the same source.
export async function createListings(payload, {
  signal, categoryGenerationAvailable = true, onCategories = () => {}, onPhase = () => {},
  classify = generateCategories, generate = generateListings,
} = {}) {
  signal?.throwIfAborted();
  const { title = '', description, mcId } = payload;
  const source = { title, description, mcId };
  let categories = payload.categories ?? [];
  if (!categories.length) {
    if (!categoryGenerationAvailable) throw new Error('Автоподбор недоступен. Выберите категории вручную и нажмите «Создать».');
    onPhase('Подбираем категории…');
    categories = await classify(source, signal);
    signal?.throwIfAborted();
    if (!categories.length) throw new Error('Подходящих категорий не найдено. Уточните описание или выберите категории вручную.');
    onCategories(categories);
  }
  onPhase('Готовим объявления…');
  const response = await generate({ ...source, categories }, signal);
  signal?.throwIfAborted();
  return { ...response, categories };
}
