# Локальный Streamlit-интерфейс слепой разметки

Интерфейс для **97 уникальных Public-заданий** и **114 train-заданий**. Показывает только полный `description`, исходный `mcId=101`, возможность выбрать любое подмножество 102–106 (включая пустое) и комментарий. `current_labels` из train-файла намеренно скрыты, а папка `private/` приложением **не читается**. Предсказания, gold и вероятности не отображаются. Доступ ограничен loopback, учётных записей/паролей у приложения нет: каждый разметчик выбирает свой профиль (`annotator_1` или `annotator_2`). Не передавайте общий туннель третьим лицам.

## Запуск

На сервере отдельно от рабочей `.venv`:

```bash
python3 -m venv /srv/Hakaton/data-quality-ui-venv
/srv/Hakaton/data-quality-ui-venv/bin/pip install 'streamlit>=1.40,<2' pytest
cd /srv/Hakaton/innopolis_2026/datasets/data_quality_audit/annotation_ui
/srv/Hakaton/data-quality-ui-venv/bin/streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false
```

Открыть **на самом сервере** `http://127.0.0.1:8501`. С другого компьютера:

```bash
ssh -N -L 8501:127.0.0.1:8501 hakaton
```

Затем открыть `http://127.0.0.1:8501` **на своём компьютере**. При занятом локальном порте используйте `-L 18501:127.0.0.1:8501` и адрес `http://127.0.0.1:18501`. Не открывать порт в firewall, не менять `--server.address` на `0.0.0.0`.

Для установленного фонового запуска (если сервис создан): `systemctl --user status data-quality-annotation.service`, `systemctl --user restart data-quality-annotation.service`, `journalctl --user -u data-quality-annotation.service -n 50`; локальный порт остаётся тем же. Остановка: `systemctl --user stop data-quality-annotation.service`.

## Работа и сохранение

1. Выберите набор и свой профиль в боковой панели. Два профиля записывают **разные файлы**, но приложение не проверяет личность человека; заранее договоритесь, кто использует каждый профиль.
2. Прочитайте полный текст, отметьте дополнительные категории (или ни одной), добавьте комментарий к спорному случаю и нажмите **«Сохранить»**, даже для `[]`. Несохранённые изменения блокируют переход по «Следующее»/«Предыдущее»; новые пустые задания можно пропустить, не помечая заполненными.
3. Прогресс считается по сохранённым `item_id`. Боковая панель показывает незаполненные задания и предупреждает о неполном экспорте. Кнопка экспорта скачивает только сохранённые ответы текущего профиля и набора.
4. После закрытия и повторного открытия сохранённые ответы загружаются из JSONL. При сохранении используется блокировка и атомарная замена файла; запись ответа не правит задания.

Файлы ответов (вне Git): `/srv/Hakaton/results/data_quality_audit/human_labels/{public,train}_{annotator_1,annotator_2}.jsonl`. Папка доступна только владельцу, файлы `0600`. Исходные `annotation/public_blind.jsonl` и `annotation/train_review.jsonl` не изменяются. Каждая строка ответа содержит `item_id`, `source_mcId`, метки `human_labels`, `human_comment`, временную отметку и исходные индексы; никаких исходных gold/предсказаний.

Полный экспорт Public JSONL подходит для скрипта:

```bash
/srv/Hakaton/transformer-venv/bin/python \
  /srv/Hakaton/innopolis_2026/datasets/data_quality_audit/score_human_labels.py \
  /srv/Hakaton/results/data_quality_audit/human_labels/public_annotator_1.jsonl
```

Этот скрипт запускайте **после** независимой разметки обоих профилей и согласования спорных примеров. Он требует заполненности всех 97 заданий; его показатели на обогащённой ошибками выборке **не являются независимой тестовой метрикой**. Никакое обучение в интерфейсе не выполняется.

## Тесты

```bash
cd /srv/Hakaton/innopolis_2026/datasets/data_quality_audit/annotation_ui
/srv/Hakaton/data-quality-ui-venv/bin/python -m unittest -v test_storage.py
```

Тесты используют временные файлы, не меняя реальные ответы. Проверяют слепую проекцию train, отдельные профили, сохранение/обновление/повторную загрузку, пустую разметку, экспорт и отказ от неверных категорий.
