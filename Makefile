# Robast = H123_robust classifier + Qwen3.5 generator. Run `make help`.
# Commands work from either the repository or project/ inside the delivery bundle.
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUNDLE_DIR := $(abspath $(PROJECT_DIR)/..)

VENV_DIR ?= $(if $(wildcard $(BUNDLE_DIR)/START_HERE.md),$(BUNDLE_DIR)/.venv,$(PROJECT_DIR)/.venv)
PYTHON ?= $(VENV_DIR)/bin/python
NODE ?= node
SITE_HOST ?= 127.0.0.1
SITE_PORT ?= 15181
BACKEND_PORT ?= 18009
QWEN_MODE ?= managed
GGUF_DIR ?= $(BUNDLE_DIR)/models
PROXY_FILE ?= $(BUNDLE_DIR)/proxy.py
INPUT ?=
OUTPUT ?=

# Override these together to run a second copy without replacing the working site.
BACKEND_UNIT ?= robast-backend
SITE_UNIT ?= robast-site
LLAMA_CONTAINER ?= robast-llama
PROXY_CONTAINER ?= robast-proxy

.PHONY: help install test verify up qwen backend site check status smoke csv logs down

help:
	@printf '%s\n' \
	  'Robast (H123_robust) — команды:' \
	  '  make install                  Python-окружение (нужен интернет или wheelhouse)' \
	  '  make verify                  Проверка SHA-256 архива (только внутри bundle)' \
	  '  make up                      Контейнеры Qwen + backend + сайт' \
	  '  make up QWEN_MODE=existing   Использовать уже работающий Qwen; его не останавливаем' \
	  '  make status / check / smoke  Состояние / HTTP-проверка / тестовый CSV' \
	  '  make csv INPUT=in.csv OUTPUT=out.csv  CSV для бота, без подмены модели' \
	  '  make logs                    Последние журналы управляемых сервисов' \
	  '  make down                    Остановить только сервисы с именами из этого Makefile' \
	  'Параметры: PYTHON, NODE, SITE_HOST, SITE_PORT, BACKEND_PORT, GGUF_DIR, PROXY_FILE.'

install:
	python3.12 -m venv "$(VENV_DIR)"
	"$(VENV_DIR)/bin/python" -m pip install -r "$(PROJECT_DIR)/backend/requirements.txt"
	@echo 'Для GPU проверьте CUDA-сборку torch; версии рабочего хоста — installed-packages.txt в архиве.'

test:
	PYTHONPATH="$(PROJECT_DIR)/scripts" PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" -m unittest -v test_run_submission

verify:
	@test -f "$(BUNDLE_DIR)/SHA256SUMS" || { echo 'SHA256SUMS не найден: команда для распакованного bundle.' >&2; exit 1; }
	cd "$(BUNDLE_DIR)" && sha256sum -c SHA256SUMS
	@test "$$(find "$(BUNDLE_DIR)" -xtype l | wc -l)" -eq 0

up: qwen backend site check
	@echo "Сайт: http://$(SITE_HOST):$(SITE_PORT) (robast/H123_robust)"

qwen:
	@set -eu; \
	case "$(QWEN_MODE)" in \
	  existing) echo 'Используем уже запущенный Qwen; контейнеры не меняем.' ;; \
	  managed) \
	    test -f "$(GGUF_DIR)/Qwen3.5-9B-Q8_0.gguf" || { echo 'Укажите GGUF_DIR с весом Qwen3.5-9B-Q8_0.gguf.' >&2; exit 1; }; \
	    test -f "$(PROXY_FILE)" || { echo 'Укажите PROXY_FILE (proxy.py).' >&2; exit 1; }; \
	    if ! docker container inspect "$(LLAMA_CONTAINER)" >/dev/null 2>&1 && ss -ltn | grep -q ':8002 '; then \
	      echo 'Порт 8002 занят: используйте QWEN_MODE=existing или освободите порт.' >&2; exit 1; fi; \
	    if ! docker container inspect "$(PROXY_CONTAINER)" >/dev/null 2>&1 && ss -ltn | grep -q ':8000 '; then \
	      echo 'Порт 8000 занят: используйте QWEN_MODE=existing или освободите порт.' >&2; exit 1; fi; \
	    if docker container inspect "$(LLAMA_CONTAINER)" >/dev/null 2>&1; then \
	      test "$$(docker container inspect -f '{{range .Mounts}}{{.Source}}{{end}}' "$(LLAMA_CONTAINER)")" = "$(GGUF_DIR)" || { echo 'Имя llama занято контейнером с другим GGUF_DIR.' >&2; exit 1; }; \
	      if ! docker container inspect -f '{{.State.Running}}' "$(LLAMA_CONTAINER)" | grep -qx true; then docker start "$(LLAMA_CONTAINER)"; fi; \
	    else \
	      docker run -d --name "$(LLAMA_CONTAINER)" --restart unless-stopped --gpus all --network host \
	        --health-cmd='curl -fsS http://127.0.0.1:8002/health' --health-interval=30s \
	        --health-timeout=5s --health-start-period=120s --health-retries=3 \
	        -v "$(GGUF_DIR):/models:ro" ghcr.io/ggml-org/llama.cpp:server-cuda \
	        -m /models/Qwen3.5-9B-Q8_0.gguf --host 127.0.0.1 --port 8002 \
	        --ctx-size 65536 --n-gpu-layers 99 --parallel 2 --alias avito-qwen --reasoning off; \
	    fi; \
	    if docker container inspect "$(PROXY_CONTAINER)" >/dev/null 2>&1; then \
	      test "$$(docker container inspect -f '{{range .Mounts}}{{.Source}}{{end}}' "$(PROXY_CONTAINER)")" = "$(PROXY_FILE)" || { echo 'Имя proxy занято контейнером с другим PROXY_FILE.' >&2; exit 1; }; \
	      if ! docker container inspect -f '{{.State.Running}}' "$(PROXY_CONTAINER)" | grep -qx true; then docker start "$(PROXY_CONTAINER)"; fi; \
	    else \
	      docker run -d --name "$(PROXY_CONTAINER)" --restart unless-stopped --network host \
	        -v "$(PROXY_FILE):/app/proxy.py:ro" python:3.12-slim python -u /app/proxy.py; \
	    fi ;; \
	  *) echo 'QWEN_MODE: managed или existing.' >&2; exit 1 ;; \
	esac; \
	for i in $$(seq 1 180); do \
	  if curl -fsS --max-time 2 http://127.0.0.1:8000/props 2>/dev/null | grep -q 'Qwen3.5-9B-Q8_0.gguf'; then echo 'Qwen3.5 ready'; exit 0; fi; \
	  sleep 1; \
	done; echo 'Qwen не готов на 8000; проверьте docker logs.' >&2; exit 1

backend:
	@test -x "$(PYTHON)" || { echo 'Нет Python-окружения: make install или PYTHON=/path/to/venv/bin/python.' >&2; exit 1; }
	@test -f "$(PROJECT_DIR)/models/h123/manifest.json" || { echo 'Не найдены веса H123.' >&2; exit 1; }
	@if systemctl --user is-active --quiet "$(BACKEND_UNIT).service"; then \
	  test "$$(systemctl --user show "$(BACKEND_UNIT).service" -p WorkingDirectory --value)" = "$(PROJECT_DIR)" || { echo 'Имя backend unit занято другим каталогом.' >&2; exit 1; }; \
	  systemctl --user show "$(BACKEND_UNIT).service" -p Environment --value | grep -q 'CLASSIFIER_MODEL_PATH=models/h123' || { echo 'Backend unit не использует H123.' >&2; exit 1; }; \
	  systemctl --user show "$(BACKEND_UNIT).service" -p ExecStart --value | grep -Fq -- '--port $(BACKEND_PORT)' || { echo 'Backend unit запущен с другим портом.' >&2; exit 1; }; \
	  echo 'Backend уже запущен'; \
	else \
	  systemctl --user reset-failed "$(BACKEND_UNIT).service" 2>/dev/null || true; \
	  systemd-run --user --unit="$(BACKEND_UNIT)" --property="WorkingDirectory=$(PROJECT_DIR)" \
	    --setenv=CLASSIFIER_MODEL_PATH=models/h123 --setenv=QWEN_BASE_URL=http://127.0.0.1:8000/v1 \
	    --setenv=PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" -m uvicorn backend.app.main:app \
	    --host 127.0.0.1 --port "$(BACKEND_PORT)"; \
	fi
	@for i in $$(seq 1 120); do \
	  if curl -fsS --max-time 2 "http://127.0.0.1:$(BACKEND_PORT)/ready" 2>/dev/null | grep -q '"categoryGeneration":true'; then echo 'H123 backend ready'; exit 0; fi; \
	  sleep 1; \
	done; echo 'Backend не готов; make logs.' >&2; exit 1

site:
	@command -v "$(NODE)" >/dev/null || { echo 'Укажите NODE=/path/to/node (Node >=22).' >&2; exit 1; }
	@if systemctl --user is-active --quiet "$(SITE_UNIT).service"; then \
	  test "$$(systemctl --user show "$(SITE_UNIT).service" -p WorkingDirectory --value)" = "$(PROJECT_DIR)/frontend" || { echo 'Имя site unit занято другим каталогом.' >&2; exit 1; }; \
	  systemctl --user show "$(SITE_UNIT).service" -p Environment --value | grep -Fq "API_TARGET=http://127.0.0.1:$(BACKEND_PORT)" || { echo 'Сайт смотрит на другой backend.' >&2; exit 1; }; \
	  systemctl --user show "$(SITE_UNIT).service" -p Environment --value | grep -Fq "PORT=$(SITE_PORT)" || { echo 'Сайт запущен с другим портом.' >&2; exit 1; }; \
	  echo 'Сайт уже запущен'; \
	else \
	  systemctl --user reset-failed "$(SITE_UNIT).service" 2>/dev/null || true; \
	  systemd-run --user --unit="$(SITE_UNIT)" --property="WorkingDirectory=$(PROJECT_DIR)/frontend" \
	    --setenv="API_TARGET=http://127.0.0.1:$(BACKEND_PORT)" --setenv="HOST=$(SITE_HOST)" \
	    --setenv="PORT=$(SITE_PORT)" "$(NODE)" scripts/server.mjs; \
	fi
	@for i in $$(seq 1 20); do \
	  if curl -fsS --max-time 2 "http://$(SITE_HOST):$(SITE_PORT)/api/status" 2>/dev/null | grep -q '"ready":true'; then echo 'Сайт ready'; exit 0; fi; \
	  sleep 1; \
	done; echo 'Сайт не готов; make logs.' >&2; exit 1

check:
	@curl -fsS --max-time 5 http://127.0.0.1:8000/props | grep -q 'Qwen3.5-9B-Q8_0.gguf'
	@curl -fsS --max-time 5 "http://127.0.0.1:$(BACKEND_PORT)/ready" | grep -q '"categoryGeneration":true'
	@curl -fsS --max-time 5 "http://$(SITE_HOST):$(SITE_PORT)/api/status" | grep -q '"ready":true'
	@curl -fsS --max-time 5 -o /dev/null "http://$(SITE_HOST):$(SITE_PORT)/"
	@echo 'Проверка Qwen, H123 backend и сайта пройдена.'

status:
	@systemctl --user --no-pager --plain status "$(BACKEND_UNIT).service" "$(SITE_UNIT).service" 2>&1 | head -35 || true
	@docker ps --filter "name=$(LLAMA_CONTAINER)" --filter "name=$(PROXY_CONTAINER)" --format '{{.Names}} {{.Status}}' || true
	@curl -fsS --max-time 3 "http://$(SITE_HOST):$(SITE_PORT)/api/status" || true
	@echo

smoke:
	@tmp=$$(mktemp -d /tmp/opencode/robast-smoke.XXXXXX); trap 'rm -f "$$tmp/input.csv" "$$tmp/output.csv"; rmdir "$$tmp"' EXIT; \
	printf 'testcase,description\nsmoke-001,"Укладываем плитку в санузле. Работаем по договору."\nsmoke-002,"Собираем мебель. Ремонт не выполняем."\n' > "$$tmp/input.csv"; \
	"$(PYTHON)" "$(PROJECT_DIR)/scripts/run_submission.py" "$$tmp/input.csv" "$$tmp/output.csv" \
	  --base-url "http://127.0.0.1:$(BACKEND_PORT)" --deadline 180; \
	"$(PYTHON)" -c 'import csv,json,sys; r=list(csv.DictReader(open(sys.argv[1],newline=""))); assert len(r)==2 and all(len(json.loads(x["microcats"]))==len(json.loads(x["descriptions"])) for x in r); print("CSV OK:",len(r),"rows")' "$$tmp/output.csv"

csv:
	@test -n "$(INPUT)" && test -n "$(OUTPUT)" || { echo 'Пример: make csv INPUT=input.csv OUTPUT=submission.csv' >&2; exit 1; }
	"$(PYTHON)" "$(PROJECT_DIR)/scripts/run_submission.py" "$(INPUT)" "$(OUTPUT)" \
	  --base-url "http://127.0.0.1:$(BACKEND_PORT)" --deadline 900

logs:
	journalctl --user -u "$(BACKEND_UNIT).service" -u "$(SITE_UNIT).service" -n 100 --no-pager
	@docker logs --tail 30 "$(LLAMA_CONTAINER)" 2>&1 || true
	@docker logs --tail 30 "$(PROXY_CONTAINER)" 2>&1 || true

down:
	@for spec in "$(SITE_UNIT):$(PROJECT_DIR)/frontend" "$(BACKEND_UNIT):$(PROJECT_DIR)"; do \
	  unit=$${spec%%:*}; expected=$${spec#*:}; \
	  actual=$$(systemctl --user show "$$unit.service" -p WorkingDirectory --value 2>/dev/null || true); \
	  if [ "$$actual" = "$$expected" ]; then systemctl --user stop "$$unit.service"; \
	  elif [ -n "$$actual" ]; then echo "Не останавливаю $$unit: чужой WorkingDirectory $$actual" >&2; fi; \
	done
	@if [ "$(QWEN_MODE)" = managed ]; then \
	  for spec in "$(PROXY_CONTAINER):$(PROXY_FILE)" "$(LLAMA_CONTAINER):$(GGUF_DIR)"; do \
	    name=$${spec%%:*}; expected=$${spec#*:}; \
	    actual=$$(docker container inspect -f '{{range .Mounts}}{{.Source}}{{end}}' "$$name" 2>/dev/null || true); \
	    if [ "$$actual" = "$$expected" ]; then docker stop "$$name"; \
	    elif [ -n "$$actual" ]; then echo "Не останавливаю $$name: чужой mount $$actual" >&2; fi; \
	  done; \
	  echo 'Остановлены управляемые контейнеры (не удалены).'; \
	else echo 'Существующий Qwen не тронут.'; fi
