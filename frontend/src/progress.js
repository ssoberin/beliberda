// Progress counts completed records/categories, never elapsed-time estimates.
export class CreationProgress {
  constructor() {
    this.total = 0;
    this.completed = new Set();
    this.categories = new Map();
    this.processed = new Map();
    this.finished = false;
    this.failed = 0;
    this.interrupted = false;
  }

  update(event) {
    if (event.type === 'start') this.total = event.total;
    if (event.type === 'categories') this.categories.set(event.index, event.categories.length);
    if (event.type === 'drafts' || event.type === 'category_error') {
      this.processed.set(event.index, (this.processed.get(event.index) ?? 0) + 1);
    }
    if (event.type === 'item_done') this.completed.add(event.index);
    if (event.type === 'done') {
      this.finished = true;
      this.failed = event.failed;
    }
    return this.snapshot();
  }

  snapshot() {
    const categoryCount = this.total === 1 ? this.categories.get(0) ?? 0 : 0;
    const max = categoryCount || this.total || 1;
    const value = categoryCount
      ? this.completed.has(0) ? max : Math.min(max, this.processed.get(0) ?? 0)
      : this.completed.size;
    const generating = [...this.categories.values()].some(count => count > 0);
    const stage = this.finished ? 2 : generating ? 1 : 0;
    return {
      max, value, stage, indeterminate: !this.total && !this.interrupted,
      error: this.interrupted || (this.finished && this.failed > 0),
      title: this.interrupted ? 'Обработка прервана'
        : this.finished ? this.failed ? 'Завершено с ошибками' : 'Готово'
          : generating ? 'Создаём объявления' : 'Определяем услуги',
      count: this.total ? `${categoryCount ? 'Категории' : 'Объявления'}: ${value} из ${max}` : 'Отправляем запрос…',
    };
  }
}
