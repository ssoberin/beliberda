
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
Старые копии `/srv/Hakaton/app/` и `/srv/Hakaton/app_backup_v1/` удалены; API
запускается только из Git-репозитория. Эксперименты `/srv/Hakaton/avito_compare/`
также не обновляются через Git.
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

## Эксперименты в test-branch

Код, измерения, выводы и команды воспроизведения: [avito_compare/EXPERIMENTS.md](avito_compare/EXPERIMENTS.md).

- [Выполненный notebook всех экспериментов](output/experiment-review/experiments.ipynb)
- [Длины реальных и синтетических текстов](output/dataset-length-comparison/lengths.ipynb)
- [Парное сравнение до/после добавления синтетики](output/paired-data-evaluation/report.md)

Веса, кэши и подготовленные выборки локальные; production-модель не заменялась.

### Контрольная точка: запуск и сквозной запрос (14 ноября 2026)

- API запускается из `ml-service/app/` пользовательским сервисом `avito-ml.service`;
  старая копия `/srv/Hakaton/app/` удалена. Рабочий сервис использует **старую**
  модель `/srv/Hakaton/models/baseline/tfidf_baseline.joblib`, а не экспериментальную.
- `GET /health` вернул `{"status":"ok"}`. Для примера из постановки с
  `mcId=101` и явно предлагаемыми сантехническими работами `POST /generate`
  вернул единственную категорию `104` как при `generate_text=false`, так и при
  `generate_text=true`. Во втором случае vLLM/Qwen вернула заголовок и текст.
- Этот удачный запрос **не доказывает качество на всех объявлениях**: в полученном
  тексте было вводное «А также отдельно выполняем…», мешающее воспринимать его
  как самостоятельное объявление; на другом коротком примере черновик остался
  с `null` в текстовых полях. Интеграция с фронтендом пока не выполнена.
- Тесты Python API из Git-копии: `6 passed`. Оценки старой и экспериментальной
  моделей на public validation/test и hard v2 validation/test сохранены в
  [datasets/model_points.ipynb](datasets/model_points.ipynb). Новая модель в
  работающий API **не включена**; он пока не умеет возвращать целевую `101`.

Повторить проверку (на сервере, когда оба сервиса работают):

```bash
PAYLOAD='{"description":"Делаем ремонт квартир под ключ, а также отдельно выполняем сантехнические работы: разводку труб, установку раковин и унитазов, замену смесителей и подключение стиральных машин. Помогаем подобрать материалы и согласовываем стоимость до начала работ.","mcId":101}'
curl -fsS -X POST 'http://127.0.0.1:8001/generate?generate_text=false' -H 'Content-Type: application/json' -d "$PAYLOAD"
curl -fsS --max-time 180 -X POST 'http://127.0.0.1:8001/generate?generate_text=true' -H 'Content-Type: application/json' -d "$PAYLOAD"
```

**Следующий этап:** повышать метрики на публичных текстах, исследуя ошибки и
пороговые решения. Обучать только на public train (и явно выбранной добавке);
public calibration использовать для настройки, validation — для контроля, а
public test не включать в обучение. Фиксировать precision, recall, micro-F1,
полное совпадение наборов и изменения относительно этой контрольной точки.
Слабая автоматическая разметка и пересечения public test с обучением требуют
осторожной интерпретации результатов; закрытый скоринг остаётся отдельной оценкой.
