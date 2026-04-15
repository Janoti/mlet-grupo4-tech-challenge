SHELL := /bin/bash

POETRY ?= poetry
NB_TIMEOUT ?= -1
IOPUB_TIMEOUT ?= 120
NB_LOG_LEVEL ?= ERROR
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000
MLFLOW_LOG ?= .tmp/mlflow.log
MLFLOW_PID ?= .tmp/mlflow.pid

EDA_NOTEBOOK := notebooks/01_eda.ipynb
BASELINE_NOTEBOOK := notebooks/02_baselines.ipynb
MLP_NOTEBOOK      := notebooks/03_mlp_pytorch.ipynb
DATA_FILE := data/raw/telecom_churn_base_extended.csv

.PHONY: help install run-all notebooks notebooks-eda notebooks-baselines notebooks-mlp analyze mlflow mlflow-up mlflow-down mlflow-clean generate-data

help:
	@echo "Targets disponiveis:"
	@echo "  make install             - instala dependencias com poetry"
	@echo "  make generate-data       - gera arquivo de dados sinteticos se nao existir"
	@echo "  make run-all             - limpa runs, gera dados, executa todos os notebooks, analisa e sobe MLflow"
	@echo "  make notebooks           - executa 01_eda, 02_baselines e 03_mlp_pytorch em sequencia"
	@echo "  make notebooks-eda       - executa somente 01_eda.ipynb"
	@echo "  make notebooks-baselines - executa somente 02_baselines.ipynb"
	@echo "  make notebooks-mlp       - executa somente 03_mlp_pytorch.ipynb"
	@echo "  make analyze             - analisa automaticamente as runs no mlruns/"
	@echo "  make mlflow              - sobe o MLflow UI em $(MLFLOW_HOST):$(MLFLOW_PORT)"
	@echo "  make mlflow-up           - sobe MLflow em background e mostra link"
	@echo "  make mlflow-down         - derruba MLflow em background"
	@echo "  make mlflow-clean        - limpa artefatos locais em mlruns/"

install:
	@echo "[install] Instalando dependencias..."
	$(POETRY) install
	@echo "[install] OK"

generate-data:
	@if [ ! -f "$(DATA_FILE)" ]; then \
		echo "[generate-data] Arquivo nao encontrado: $(DATA_FILE)"; \
		echo "[generate-data] Gerando dados sinteticos..."; \
		$(POETRY) run python scripts/generate_synthetic.py --n-rows 50000 --seed 42 --out-dir data/raw; \
		echo "[generate-data] OK"; \
	else \
		echo "[generate-data] Arquivo ja existe: $(DATA_FILE)"; \
	fi

run-all: mlflow-clean generate-data notebooks analyze mlflow-up
	@echo "[run-all] Steps concluidos: clean -> generate-data -> notebooks -> analysis -> mlflow"
	@echo "[run-all] Abra: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"

notebooks: notebooks-eda notebooks-baselines notebooks-mlp
	@echo "[notebooks] Execucao completa finalizada."

notebooks-eda:
	@echo "[notebooks-eda] Iniciando execucao de $(EDA_NOTEBOOK)..."
	$(POETRY) run jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=$(NB_TIMEOUT) \
		--ExecutePreprocessor.iopub_timeout=$(IOPUB_TIMEOUT) \
		--Application.log_level=$(NB_LOG_LEVEL) \
		$(EDA_NOTEBOOK)
	@echo "[notebooks-eda] Concluido."

notebooks-baselines:
	@echo "[notebooks-baselines] Iniciando execucao de $(BASELINE_NOTEBOOK)..."
	$(POETRY) run jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=$(NB_TIMEOUT) \
		--ExecutePreprocessor.iopub_timeout=$(IOPUB_TIMEOUT) \
		--Application.log_level=$(NB_LOG_LEVEL) \
		$(BASELINE_NOTEBOOK)
	@echo "[notebooks-baselines] Concluido."

notebooks-mlp:
	@echo "[notebooks-mlp] Iniciando execucao de $(MLP_NOTEBOOK)..."
	$(POETRY) run jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=$(NB_TIMEOUT) \
		--ExecutePreprocessor.iopub_timeout=$(IOPUB_TIMEOUT) \
		--Application.log_level=$(NB_LOG_LEVEL) \
		$(MLP_NOTEBOOK)
	@echo "[notebooks-mlp] Concluido."

analyze:
	@echo "[analysis] Lendo mlruns e analisando resultados..."
	$(POETRY) run python scripts/analyze_mlruns.py
	@echo "[analysis] Concluido."

mlflow:
	@echo "[mlflow] Subindo MLflow UI em http://$(MLFLOW_HOST):$(MLFLOW_PORT)"
	$(POETRY) run mlflow ui --backend-store-uri ./mlruns --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

mlflow-up:
	@mkdir -p .tmp
	@if lsof -ti tcp:$(MLFLOW_PORT) -s tcp:listen >/dev/null 2>&1; then \
		echo "[mlflow] Ja em execucao na porta $(MLFLOW_PORT) (PID: $$(lsof -ti tcp:$(MLFLOW_PORT) -s tcp:listen | head -1))"; \
		echo "[mlflow] Link: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"; \
	else \
		rm -rf venv-teste 2>/dev/null; \
		echo "[mlflow] Subindo em background..."; \
		nohup $(POETRY) run mlflow ui --backend-store-uri ./mlruns --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) > $(MLFLOW_LOG) 2>&1 & \
		GUNICORN_PID=""; \
		for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do \
			sleep 1; \
			GUNICORN_PID=$$(lsof -ti tcp:$(MLFLOW_PORT) -s tcp:listen | head -1); \
			if [ -n "$$GUNICORN_PID" ]; then break; fi; \
			echo "[mlflow] Aguardando inicializacao... ($${i}s)"; \
		done; \
		if [ -n "$$GUNICORN_PID" ]; then \
			echo "$$GUNICORN_PID" > $(MLFLOW_PID); \
			echo "[mlflow] PID: $$GUNICORN_PID"; \
			echo "[mlflow] Link: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"; \
			echo "[mlflow] Log: $(MLFLOW_LOG)"; \
		else \
			echo "[mlflow] ERRO: MLflow nao iniciou apos 20s. Verifique $(MLFLOW_LOG)"; \
		fi; \
	fi

mlflow-down:
	@PIDS=$$(lsof -ti tcp:$(MLFLOW_PORT) -s tcp:listen 2>/dev/null); \
	if [ -n "$$PIDS" ]; then \
		echo "[mlflow] Encerrando processos na porta $(MLFLOW_PORT): $$PIDS"; \
		echo "$$PIDS" | xargs kill; \
		rm -f $(MLFLOW_PID); \
		sleep 1; \
		REMAINING=$$(lsof -ti tcp:$(MLFLOW_PORT) -s tcp:listen 2>/dev/null); \
		if [ -n "$$REMAINING" ]; then \
			echo "[mlflow] Forcando encerramento: $$REMAINING"; \
			echo "$$REMAINING" | xargs kill -9; \
		fi; \
		echo "[mlflow] Processo encerrado."; \
	else \
		rm -f $(MLFLOW_PID); \
		echo "[mlflow] Nenhum processo na porta $(MLFLOW_PORT) para encerrar."; \
	fi

mlflow-clean:
	@echo "[mlflow-clean] Limpando conteudo de mlruns/"
	@if [ -d mlruns ]; then find mlruns -mindepth 1 -delete; fi
	@echo "[mlflow-clean] OK"
