
я крутой чувак

## ML-сервис Авито

- `ml-service/app/` — исходный код FastAPI: классификация TF-IDF и генерация
  черновиков через Qwen/vLLM.
- `avito_compare/scripts/` — подготовка данных, обучение и отдельный эксперимент
  сравнения TF-IDF с Qwen. Инструкции: [avito_compare/README.md](avito_compare/README.md).
- `datasets/` — все датасеты, генераторы и [EDA в Jupyter Notebook](datasets/eda/EDA.ipynb).
  Для обучения на сложной семантике используйте [hard_dataset/v2](datasets/hard_dataset/v2/README.md).
- `mvp/` — материалы постановки задачи (существовавшие в Git).

Единственный поддерживаемый исходный код API — `ml-service/app/` в этом Git-репозитории.
Старая `/srv/Hakaton/app/` сохранена как историческая копия, но не используется для
запуска. Эксперименты `/srv/Hakaton/avito_compare/` также не обновляются через Git.
Не коммитьте `.venv`, секреты, веса модели и результаты экспериментов.
Датасеты собраны в `datasets/`; их содержимое при переносе сохранено.

### Запуск на существующем сервере

```bash
cd /srv/Hakaton/innopolis_2026/ml-service
/srv/Hakaton/.venv/bin/python -m pytest -q -p no:cacheprovider app/tests
./run.sh  # вручную, только если avito-ml.service остановлен
```

`run.sh` выбирает модель `/srv/Hakaton/models/baseline/tfidf_baseline.joblib`
и Python `/srv/Hakaton/.venv/bin/python`; пути можно переопределить через
`AVITO_MODEL_PATH` и `AVITO_PYTHON`. Он всегда импортирует `app` из Git-репозитория.
Шаблон пользовательского systemd-сервиса: `ml-service/avito-ml.service`.
На существующем сервере он подключён как `avito-ml.service`:

```bash
systemctl --user status avito-ml.service
systemctl --user restart avito-ml.service  # после изменения кода/API
curl -sS http://127.0.0.1:8001/health
```

Unit запускается в пользовательской сессии; если нужен запуск до входа пользователя
после перезагрузки, администратор должен отдельно включить linger или настроить
системный сервис. Фронтенд пока использует отдельный несовместимый контракт и
локальные заглушки: его демонстрация не является проверкой ML API.
Qwen ожидается по адресу `http://127.0.0.1:8000`; подробности и ограничения API
описаны в [ml-service/app/README.md](ml-service/app/README.md). Не открывайте
эти порты в публичную сеть без авторизации.
