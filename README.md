
# Predicao de Churn em Telecom - Tech Challenge (Grupo 4)

Projeto de Machine Learning para prever churn em telecom, com foco em EDA, baselines e organizacao de pipeline para evolucao do modelo.

## 1. Objetivo

- Estimar a probabilidade de churn por cliente.
- Priorizar campanhas de retencao com base em risco.
- Medir impacto com metricas tecnicas (AUC-ROC, PR-AUC, top-K) e de negocio (churn evitado, ROI).

## 2. Dataset principal

A base de referencia do projeto e:

- `data/raw/telecom_churn_base_extended.csv`

Ela contem dados sinteticos com variaveis de:

- Perfil e contrato:
	- `plan_type`: tipo de plano do cliente (pre, controle, pos, empresarial).
	- `months_to_contract_end`: meses restantes para termino do contrato atual.
	- `has_loyalty`: indica se o cliente esta em periodo de fidelizacao (0/1).
- Qualidade de servico:
	- `network_outages_30d`: quantidade de quedas de rede nos ultimos 30 dias.
	- `avg_signal_quality`: qualidade media de sinal percebida pelo cliente.
	- `call_drop_rate`: taxa de chamadas interrompidas.
- Uso e engajamento:
	- `minutes_monthly`: minutos de voz consumidos no mes.
	- `data_gb_monthly`: consumo mensal de dados moveis em GB.
	- `usage_delta_pct`: variacao percentual recente no uso (tendencia de queda/subida).
	- `app_login_30d`: numero de logins no app nos ultimos 30 dias.
- Financeiro:
	- `monthly_charges`: valor cobrado mensalmente do cliente.
	- `invoice_shock_flag`: flag de aumento relevante de fatura (0/1).
	- `late_payments_6m`: quantidade de pagamentos em atraso nos ultimos 6 meses.
	- `default_flag`: indica historico de inadimplencia (0/1).
- Atendimento:
	- `support_calls_30d`: numero de contatos com suporte nos ultimos 30 dias.
	- `complaints_30d`: indicador de reclamacao recente (0/1).
	- `resolution_time_avg`: tempo medio de resolucao dos atendimentos.
- Satisfacao:
	- `nps_score`: nota NPS individual do cliente (0 a 10).
	- `nps_category`: classificacao NPS (promoter, passive, detractor).
	- `csat_score`: nota de satisfacao (CSAT) do cliente.

Colunas-chave de saida:

- `churn` (target binario)
- `churn_probability` (probabilidade sintetica de referencia)

## 3. Estrutura do repositorio

```text
mlet-grupo4-tech-challenge/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── ml_canvas.md
│   └── model_card.md
├── models/
│   ├── artifacts/
│   └── model_registry/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baselines.ipynb
├── scripts/
│   ├── generate_dataset.py
│   └── generate_synthetic.py
├── src/churn_prediction/
├── tests/
├── pyproject.toml
└── README.md
```

## 4. Ambiente e instalacao

Pre-requisitos:

- Python 3.11+ (recomendado 3.12)
- Poetry

Instalacao:

```bash
git clone https://github.com/Janoti/mlet-grupo4-tech-challenge.git
cd mlet-grupo4-tech-challenge
poetry install
```

## 5. Geracao dos dados

### Base estendida (recomendada)

```bash
poetry run python scripts/generate_synthetic.py --n-rows 50000 --seed 42 --out-dir data/raw
```

Arquivo gerado:

- `data/raw/telecom_churn_base_extended.csv`

### Base simplificada (legado)

```bash
poetry run python scripts/generate_dataset.py --n-rows 50000 --seed 42 --out-dir data/raw
```

Arquivo gerado:

- `data/raw/telecom_churn_base.csv`

## 6. Fluxo recomendado

1. Gerar/validar dataset em `data/raw`.
2. Executar exploracao no notebook `notebooks/01_eda.ipynb`.
3. Treinar e comparar baselines no notebook `notebooks/02_baselines.ipynb`.
4. Registrar experimentos no MLflow (quando habilitado no fluxo).
5. Consolidar resultados em `docs/model_card.md`.

## 7. Qualidade de codigo

Lint:

```bash
poetry run ruff check src scripts tests
```

Auto-fix:

```bash
poetry run ruff check src scripts tests --fix
```

Testes:

```bash
poetry run pytest -q
```

## 8. MLflow (opcional local)

```bash
poetry run mlflow ui --backend-store-uri ./mlruns
```

Abrir em http://127.0.0.1:5000

## 9. Documentacao

- Canvas de negocio e modelagem: `docs/ml_canvas.md`
- Card do modelo: `docs/model_card.md`

## 10. Proximos passos

1. Fechar baseline tabular com validacao estratificada.
2. Definir corte operacional por top-K para retencao.
3. Evoluir pipeline em `src/churn_prediction/pipelines/`.
4. Criar camada de inferencia (FastAPI) e testes de contrato.

