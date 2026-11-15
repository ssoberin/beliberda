# Локальный запуск пайплайна на сервере

Два пользовательских systemd-сервиса из этой папки используют конкретные пути
и локальный IP данного сервера (`192.168.1.181`). Перед повторной установкой
на другой машине замените пути, адрес и имя модели. Ключи и веса не хранятся
в Git. Frontend слушает **только** указанный адрес локальной сети на порту 15173,
backend — только `127.0.0.1:18001`. Старый `avito-ml.service` на 8001 не меняется.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/avito-pipeline-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now avito-pipeline-backend.service avito-pipeline-frontend.service
systemctl --user status avito-pipeline-backend.service avito-pipeline-frontend.service
```

Страница: `http://192.168.1.181:15173` (если компьютер видит локальную сеть).
VPN или сетевой экран могут заблокировать доступ с другого компьютера.
Сервис **не имеет входящей авторизации**, поэтому не открывайте этот порт
в Интернет и не перенаправляйте его на публичный интерфейс. На странице
предусмотрена только подготовка черновиков, не публикация.

Проверка на сервере:

```bash
python scripts/smoke_pipeline.py --base-url http://192.168.1.181:15173
```

Остановить, если больше не требуется:

```bash
systemctl --user disable --now avito-pipeline-frontend.service avito-pipeline-backend.service
```
