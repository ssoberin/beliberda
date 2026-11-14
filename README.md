# innopolis_2026

я крутой чувак

## ML-сервис Авито

- `ml-service/app/` — исходный код FastAPI: классификация TF-IDF и генерация
  черновиков через Qwen/vLLM.
- `avito_compare/scripts/` — подготовка данных, обучение и отдельный эксперимент
  сравнения TF-IDF с Qwen. Инструкции: [avito_compare/README.md](avito_compare/README.md).
- `datasets/` — все датасеты, генераторы и [EDA в Jupyter Notebook](datasets/eda/EDA.ipynb).
  Для обучения на сложной семантике используйте [hard_dataset/v2](datasets/hard_dataset/v2/README.md).
- `mvp/` — материалы постановки задачи (существовавшие в Git).

На сервере рабочая копия API находится в `/srv/Hakaton/app/`, а эксперименты —
в `/srv/Hakaton/avito_compare/`. Они **не обновляются автоматически** при
`git pull`: перед развёртыванием сравните и явно синхронизируйте нужные файлы.
Не коммитьте `.venv`, секреты, веса модели и результаты экспериментов.
Датасеты собраны в `datasets/`; их содержимое при переносе сохранено.

### Запуск на существующем сервере

```bash
cd /srv/Hakaton
source .venv/bin/activate
python -m pytest -q app/tests
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Модель для рабочей копии: `/srv/Hakaton/models/baseline/tfidf_baseline.joblib`.
Для запуска кода **непосредственно из Git** требуется задать
`AVITO_MODEL_PATH=/srv/Hakaton/models/baseline/tfidf_baseline.joblib` и запускать
из `/srv/Hakaton/innopolis_2026/ml-service` (`uvicorn app.main:app ...`).
Qwen ожидается по адресу `http://127.0.0.1:8000`; подробности и ограничения API
описаны в [ml-service/app/README.md](ml-service/app/README.md). Не открывайте
эти порты в публичную сеть без авторизации.
