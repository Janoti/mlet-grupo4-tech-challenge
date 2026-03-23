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

.PHONY: help install run-all notebooks notebooks-eda notebooks-baselines analyze mlflow mlflow-up mlflow-down mlflow-clean

help:
	@echo "Targets disponiveis:"
	@echo "  make install            - instala dependencias com poetry"
	@echo "  make run-all            - limpa runs, executa notebooks, analisa e sobe MLflow"
	@echo "  make notebooks          - executa 01_eda e 02_baselines em sequencia"
	@echo "  make notebooks-eda      - executa somente 01_eda.ipynb"
	@echo "  make notebooks-baselines- executa somente 02_baselines.ipynb"
	@echo "  make analyze            - analisa automaticamente a ultima run no mlruns/"
	@echo "  make mlflow             - sobe o MLflow UI em $(MLFLOW_HOST):$(MLFLOW_PORT)"
	@echo "  make mlflow-up          - sobe MLflow em background e mostra link"
	@echo "  make mlflow-down        - derruba MLflow em background"
	@echo "  make mlflow-clean       - limpa artefatos locais em mlruns/"

install:
	@echo "[install] Instalando dependencias..."
	$(POETRY) install
	@echo "[install] OK"

run-all: mlflow-clean notebooks analyze mlflow-up
	@echo "[run-all] Steps concluidos: clean -> notebooks -> analysis -> mlflow"
	@echo "[run-all] Abra: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"

notebooks: notebooks-eda notebooks-baselines
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

analyze:
	@echo "[analysis] Lendo mlruns e analisando resultados..."
	$(POETRY) run python scripts/analyze_mlruns.py
	@echo "[analysis] Concluido."

mlflow:
	@echo "[mlflow] Subindo MLflow UI em http://$(MLFLOW_HOST):$(MLFLOW_PORT)"
	$(POETRY) run mlflow ui --backend-store-uri ./mlruns --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

mlflow-up:
	@mkdir -p .tmp
	@if [ -f $(MLFLOW_PID) ] && kill -0 $$(cat $(MLFLOW_PID)) 2>/dev/null; then \
		echo "[mlflow] Ja em execucao no PID $$(cat $(MLFLOW_PID))"; \
		echo "[mlflow] Link: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"; \
	else \
		echo "[mlflow] Subindo em background..."; \
		nohup $(POETRY) run mlflow ui --backend-store-uri ./mlruns --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) > $(MLFLOW_LOG) 2>&1 & echo $$! > $(MLFLOW_PID); \
		sleep 1; \
		echo "[mlflow] PID: $$(cat $(MLFLOW_PID))"; \
		echo "[mlflow] Link: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"; \
		echo "[mlflow] Log: $(MLFLOW_LOG)"; \
	fi

mlflow-down:
	@if [ -f $(MLFLOW_PID) ] && kill -0 $$(cat $(MLFLOW_PID)) 2>/dev/null; then \
		kill $$(cat $(MLFLOW_PID)); \
		rm -f $(MLFLOW_PID); \
		echo "[mlflow] Processo encerrado."; \
	else \
		echo "[mlflow] Nenhum processo em background para encerrar."; \
	fi

mlflow-clean:
	@echo "[mlflow-clean] Limpando conteudo de mlruns/"
	@if [ -d mlruns ]; then find mlruns -mindepth 1 -delete; fi
	@echo "[mlflow-clean] OK"
