# Сквозной запуск: TF-IDF → Qwen → черновики

Тестовая конфигурация для сервера с локальным Qwen (`avito-qwen` на
`127.0.0.1:8000/v1`) и доверенным baseline вне Git. Эти команды **не**
перезапускают старый `avito-ml.service` на порту 8001. Из корня репозитория:

```bash
CLASSIFIER_MODEL_PATH=/srv/Hakaton/models/baseline/tfidf_baseline.joblib \
QWEN_BASE_URL=http://127.0.0.1:8000/v1 \
QWEN_MODEL=avito-qwen \
QWEN_TOKENIZE_BEFORE_GENERATE=0 \
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 18001
```

`QWEN_TOKENIZE_BEFORE_GENERATE=0` нужен для локального llama.cpp:
vLLM-специфичный `/tokenize` здесь не гарантирован. Перед боевым использованием
проверить контекстное окно и максимальную длину ввода, поскольку бюджет токенов
в этом режиме автоматически не измеряется. Дефолтный адрес и SSH-туннель
для другого сервера описаны в [backend/README.md](backend/README.md).

В другом терминале, из корня репозитория:

```bash
cd frontend
API_TARGET=http://127.0.0.1:18001 PORT=15173 node scripts/server.mjs
```

Откройте `http://127.0.0.1:15173` на той же машине. Можно выбрать исходную
категорию, автоматически подобрать дополнительные категории или отметить их
вручную, затем запросить редактируемые черновики. **Это не публикация**.
Автоподбор и тексты нужно проверить человеком; TF-IDF не умеет гарантированно
распознавать новые формулировки, а Qwen может ошибаться.

В третьем терминале из корня репозитория:

```bash
python scripts/smoke_pipeline.py --base-url http://127.0.0.1:15173
```

Smoke использует только встроенное синтетическое объявление и проверяет
HTTP-маршрут `frontend → /api/categories → /api/generate → Qwen`. После проверки
оба процесса можно остановить `Ctrl+C` в их терминалах. Для автоматизированной
сквозной проверки с **поддельным** Qwen и настоящим baseline:

```bash
PIPELINE_MODEL_PATH=/srv/Hakaton/models/baseline/tfidf_baseline.joblib \
NODE_BIN=$(command -v node) \
python -m pytest -q backend/tests/test_pipeline_http.py
```

Тест сам запускает и останавливает временные процессы на свободных локальных
портах. Без настроенных переменных он пропускается; остальные unit-тесты
запускаются независимо от весов и Qwen. На сервере Node может быть доступен
по `/home/maaric/.lmstudio/.internal/utils/node`, если `node` нет в PATH.

Никакие веса, ключи, приватные данные или ответы разметчиков не коммитятся.
