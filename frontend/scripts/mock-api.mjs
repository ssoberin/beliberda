import { CATEGORIES } from '../src/catalog.js';

const patterns = new Map([
  [101, /под ключ|комплексн\S* ремонт|ремонт квартир/iu],
  [102, /штукатур/iu],
  [103, /плитк|плиточ|керамогранит|мозаик/iu],
  [104, /сантех|труб|унитаз|раковин|смесител|протеч|водоснабж/iu],
  [105, /электри|розет|проводк|выключател|светильник|люстр/iu],
  [106, /обои|обоев|обоя|маляр|покрас|покраск|шпакл/iu],
]);

// Только демонстрационная логика: здесь нет обращения к ML-сервису.
export function mockResponse(path, payload) {
  if (!payload || typeof payload.description !== 'string' || !payload.description.trim()
    || payload.description.length > 12000
    || (payload.title !== undefined && (typeof payload.title !== 'string' || payload.title.length > 100))) {
    return { status: 400, body: { message: 'Укажите описание услуг (от 1 до 12 000 символов) и название до 100 символов.' } };
  }
  if (path === '/api/categories') {
    const text = `${payload.title ?? ''}\n${payload.description}`;
    return { status: 200, body: { categories: CATEGORIES.filter(category => patterns.get(category.id).test(text)) } };
  }
  if (!Array.isArray(payload.categories) || !payload.categories.length || payload.categories.some(category => !category
    || !['string', 'number'].includes(typeof category.id) || !String(category.id).trim()
    || typeof category.name !== 'string' || !category.name.trim())) {
    return { status: 400, body: { message: 'Выберите хотя бы одну категорию.' } };
  }
  const categories = [...new Map(payload.categories.map(category => [String(category.id), category])).values()];
  return {
    status: 200,
    body: {
      drafts: categories.map(category => ({
        categoryId: category.id,
        title: category.name,
        description: `Услуги: ${category.name.toLocaleLowerCase('ru-RU')}.\n\n${payload.description.trim()}`,
      })),
    },
  };
}
