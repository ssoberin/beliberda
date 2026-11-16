# Наблюдаемая конфигурация сайта 15181 (16.11.2026)

**Статус:** технический снимок, **не подтверждение**, что именно эта конфигурация дала лучший сабмит/score 89. Для такого утверждения нужен идентификатор сабмита, вход, ответ и подтверждение команды. Ничего не переобучали и не переключали.

| Компонент | Наблюдаемое состояние |
|---|---|
| Frontend | `avito-test-branch-site.service`, `/srv/Hakaton/innopolis_2026-test-branch/frontend`, `192.168.1.181:15181`, прокси `127.0.0.1:18009`; в рабочей копии `test-branch` HEAD `39313e7` на момент снимка |
| Backend | `avito-test-branch-compatible-backend.service`, `/srv/Hakaton/innopolis_2026-test-branch-runtime`, detached HEAD `857c6d2`, `/srv/Hakaton/transformer-venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 18009` |
| Выбранный классификатор | Локальный `backend/.env` содержит `CLASSIFIER_MODEL_PATH=models/h123`; журнал запуска подтверждает загрузку CatBoost `30ca7413…` из `models/h123`, пороги 0,5. **Не default `models/c_syntax`.** Полный `.env` и секреты не копировать в Git/архив без очистки. |
| Веса H123 | `catboost.cbm` SHA-256 `30ca7413d79ff3cce258210c75cfddb031dc3f08077811e3ac99aebde7b75c90`; `sgd_heads.joblib` `49d95c79e2633d4ba2fc95593e04e8acc4ce0845549a7dcf7348f73af007c45f`; `queries.npz` `988e0d8fe92a3d10299cadb4c0bd58cf1946f1fb10e597c02c6babc1587482b7` |
| Энкодер | `Qwen/Qwen3-Embedding-0.6B`, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, из локального кэша; заморожен |
| Генератор | Локальный `127.0.0.1:8000`, alias `avito-qwen`; `/props` сообщает `/models/Qwen3.5-9B-Q8_0.gguf`, контекст 32768. Отдельный от классификатора. |
| Проверка | Сайт HTTP 200, `/api/status`: backend ready и categoryGeneration=true; H123 ранее успешно холодно стартовал в отдельном тестовом сервисе на 18010 и был остановлен. |

**Важно:** `models/c_syntax/manifest.json` в этих checkout требует старый SHA-256 `syntax_features.py` (`95f89fc0…`), но текущий файл имеет `1a0a2f75…`. Попытка новой загрузки default C_syntax выдаёт ошибку проверки. Это не проблема выбранного H123, manifest которого согласован с кодом. Не отключать проверку хешей и не подменять веса ради запуска.

Перед архивированием повторно проверить HEAD обоих worktree, значения конфигурации **без секретов**, SHA-256 файлов и точный результат контрольного CSV. Если интерфейс изменится после этого снимка, зафиксировать его новую ревизию отдельно. Презентация и CLI должны использовать ту же пару backend/frontend и тот же формат `testcase,microcats,descriptions`, где два последних поля — синхронные JSON-массивы; ошибки не должны незаметно превращаться в `[]`.
