# Интеграция test-branch

В test-branch влит origin/main307c000 (backend, frontend, создание пачек и HTTP-контракты).
Обычный путь приложения использует C_syntax из models/c_syntax, без выбора
классификатора через флаги и без автоматической подмены на TF-IDF.

Сценарий: frontend → /api/categories или /api/create → Qwen0.6B/retrieval/SGD/
Natasha/CatBoost → дополнительные категории → отдельный Qwen7B → редактируемые
черновики. Исходная категория исключается, пустой результат поддержан.
API основного backend и legacy ml-service используют один inference adapter.

Запуск: python -m uvicorn backend.app.main:app --port 18001. Настройки внешнего генератора и SSH остаются
в backend/.env; веса/пороги классификатора не выбираются ими. Подробности:
[PIPELINE_RUN.md](PIPELINE_RUN.md), [backend/README.md](backend/README.md).

В интеграцию включена исходная проверенная C_syntax без дополнения60.
Эксперимент Luna с исправлением признаков находится в отдельном worktree/ветке
experiment/syntax-offer-scope и не включается до отдельного решения по результатам.
Исследовательские отчёты и B+60 сохранены. Серверы production этой интеграцией
не перезапускались. Веса Qwen0.6B не хранятся в Git; маленький проверяемый комплект
SGD/CatBoost/query vectors хранится в этой ветке.

В lifespan модель загружается и прогревается один раз; запросы обслуживаются
одним выделенным inference-потоком с очередью16. Shell на запрос не запускается.
