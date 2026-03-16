# ML Canvas – Previsão de Churn em Telecomunicações (Fase 1)

## 1. Problema de Negócio

Operadora de telecomunicações deseja **reduzir o churn (cancelamento)** de clientes no segmento pós‑pago, identificando com antecedência quais clientes têm **alta probabilidade de cancelar** o serviço, para direcionar ações de retenção (ofertas, planos personalizados, contato proativo).

**Pergunta central:**  
“Quais clientes, ativos hoje, têm alta probabilidade de churnar nos próximos meses?”

---

## 2. Objetivo de Negócio

- **Reduzir a taxa de churn** em X p.p. (pontos percentuais; por exemplo, de 25% ao ano para ≤ 20% ao ano).  
- **Aumentar a receita recorrente** por meio da retenção de clientes de alto valor (alto ticket / LTV – *Lifetime Value*, valor do cliente ao longo do tempo).  
- **Otimizar o custo das campanhas de retenção**, contatando prioritariamente clientes com maior risco e maior valor.

---

## 3. Stakeholders

### 3.1. Negócio / Operação
- **Diretoria de Marketing / CRM (Customer Relationship Management – gestão de relacionamento com o cliente)**  
  - Define políticas de retenção, campanhas e ofertas.
- **Time de Customer Success / Atendimento**  
  - Usa a lista de clientes com alto risco de churn para contato proativo.
- **Time de Vendas / Cross-sell & Upsell**  
  - Aproveita informações de risco para ofertas segmentadas.

### 3.2. Dados / Tecnologia
- **Time de Data Science / Machine Learning (ML)**  
  - Constrói, treina e avalia o modelo de churn.
- **Time de Engenharia de Dados / MLOps (Machine Learning Operations)**  
  - Automatiza pipelines de dados, treino e deploy.
- **Time de BI (Business Intelligence) / Analytics**  
  - Monitora métricas de churn, receita e performance do modelo.

### 3.3. Gestão
- **C-Level (CEO/COO/CMO)**  
  - Acompanha indicadores estratégicos de churn e receita.
- **Gestores de Produto**  
  - Ajustam jornadas e produtos com base nos insights do modelo.

---

## 4. Usuários Finais do Modelo

- Analistas de CRM / Marketing que vão:
  - Consumir uma **tabela com score de churn** (0–1 ou baixa/média/alta) por cliente.
  - Criar **segmentações**: “alto risco e alto valor”, “médio risco”, “baixo risco”.
- Profissionais de Atendimento / Retenção que:
  - Recebem uma **fila de clientes prioritários** para contato.
- Eventualmente, serviços internos (APIs) que:
  - Consultam o score de churn em tempo real para **personalizar ofertas** no app/site.

---

## 5. Dados de Entrada (Features)

- **Perfil do cliente**
  - Idade (se disponível), gênero (se disponível), região, tipo de contrato (mensal/anual).
- **Produtos/Serviços**
  - Plano de voz, dados, TV, internet fixa, pacotes adicionais.
  - Tempo de casa (tenure).
- **Uso**
  - Minutos de ligações, consumo de dados, uso de SMS.
  - Variação de uso nos últimos meses (queda brusca é sinal de risco).
- **Financeiro**
  - Valor médio da fatura, inadimplência, atrasos de pagamento.
- **Atendimento**
  - Número de chamadas no call center, reclamações recentes, tickets abertos.
- **Rótulo (target)**
  - `churn` (1 se cancelou em uma janela de tempo, 0 caso contrário).

---

## 6. Saídas do Modelo

- **Score de churn** para cada cliente (probabilidade de cancelar: 0–1).
- **Classificação de risco**:
  - Baixo risco (0–0.3)
  - Médio risco (0.3–0.7)
  - Alto risco (0.7–1.0)  → foco principal das campanhas.

---

## 7. Métricas de Negócio

### 7.1. Principais

- **Taxa de churn (%)**
  - `clientes que cancelaram / base total de clientes` em um período.  
  - Objetivo: **reduzir** após adoção do modelo.

- **Receita recorrente preservada (R$) / Custo de churn evitado (R$)**
  - Soma de receita dos clientes de alto valor **retidos** graças à campanha baseada no modelo.  
  - Pode ser vista também como **custo de churn evitado** (quanto a empresa deixou de perder).

- **ROI (Return on Investment – retorno sobre investimento) das campanhas de retenção**
  - `(Receita adicional / custo da campanha)`  
  - Objetivo: **ROI > 1** (campanhas rentáveis).

### 7.2. Secundárias

- **Custo por contato de retenção**
  - Quanto se gasta por cliente contatado nas campanhas.
- **Aderência às ofertas**
  - % de clientes de alto risco que **aceitam** uma oferta de retenção.

---

## 8. Métricas de Modelo

- **AUC-ROC (Area Under the ROC Curve – área sob a curva ROC)**  
  - Mede capacidade de separação entre churners e não churners.
- **Precision @ Top-K%** (por exemplo, top 10% de clientes com maior score)  
  - De todos que o modelo diz que são “mais arriscados”, quantos realmente churnam.
- **Recall (Sensibilidade) para churn = 1**
  - % dos churners reais que o modelo consegue identificar como alto risco.
- **F1-score** (média harmônica entre precision e recall, balanceia as duas métricas).
- **Matriz de confusão** para entender falsos positivos/negativos.

---

## 9. SLOs (Service Level Objectives – objetivos de nível de serviço) de ML

Nesta seção, definimos o que é "bom o suficiente" para o modelo funcionar com segurança no dia a dia.
Em outras palavras: são metas objetivas para garantir qualidade, rapidez e continuidade de operação.

### 9.1. Qualidade do Modelo

- **SLO 1 – Performance mínima**
  - **Meta:** AUC-ROC **≥ 0.80** em validação/produção.
  - **Em linguagem simples:** o modelo precisa separar bem quem tende a cancelar de quem tende a ficar.
- **SLO 2 – Recall em churners de alto risco**
  - **Meta:** Recall para churn = 1 **≥ 70%** no top 20% dos clientes com maior score.
  - **Em linguagem simples:** entre os clientes que realmente vão cancelar, queremos capturar pelo menos 70% dentro da lista prioritária de maior risco.
- **SLO 3 – Estabilidade**
  - **Meta:** variação da AUC entre rodadas de treino **≤ 5 p.p.**
  - **Em linguagem simples:** o modelo não pode oscilar muito de um mês para outro. Se variar demais, há sinal de instabilidade ou mudança relevante nos dados.

### 9.2. Disponibilidade / Latência (se houver API ou uso em produção)

- **SLO 4 – Disponibilidade da API de scoring**
  - **Meta:** disponibilidade **≥ 99%** no horário comercial (ex.: 08h–22h).
  - **Em linguagem simples:** a API precisa estar no ar praticamente o tempo todo para não travar campanhas e operações.
- **SLO 5 – Latência**
  - **Meta:** tempo de resposta por requisição **≤ 300 ms** (p95).
  - **Em linguagem simples:** 95% das respostas devem voltar em até 300 ms, garantindo uso fluido por sistemas e equipes.

### 9.3. Operação / Atualização

- **SLO 6 – Frequência de re-treino**
  - **Meta:** re-treinar e reavaliar o modelo **ao menos 1x por mês** (ou antes, se houver drift).
  - **Em linguagem simples:** o modelo precisa ser atualizado com frequência para não ficar "desatualizado" com o comportamento atual dos clientes.
- **SLO 7 – Monitoramento em produção**
  - **Meta:** monitorar diariamente os principais sinais de saúde do modelo:
    - Distribuição das features vs. treino (data drift).
    - Distribuição do score de churn.
    - Métricas de performance (AUC/precision), quando os rótulos reais estiverem disponíveis.
  - **Em linguagem simples:** acompanhar todos os dias se os dados mudaram e se o modelo continua acertando. Isso evita decisões ruins por degradação silenciosa.

---

## 10. Riscos e Restrições

- **Dados desbalanceados** (poucos churners comparados a não churners).
- **Dados incompletos ou ruidosos** (erros de cadastro, eventos faltantes).
- **Mudanças de mercado** (novos planos, concorrentes, crises econômicas) gerando **drift**.
- **Viés**: modelo aprendendo padrões injustos contra certos grupos/segmentos.

---

## 11. Plano de Ação (Fase 1)

1. **Entender a base de dados de churn** (exploração, limpeza, engenharia básica de features).
2. **Definir pipeline de treino** em Python (notebook ou script):
   - Split treino/validação/teste.
   - Modelos baseline (Logistic Regression, árvore, Random Forest, etc.).
3. **Avaliar o modelo conforme métricas de modelo** definidas acima.
4. **Gerar tabela de scores de churn por cliente** (CSV/Parquet) para ser consumida por times de negócio (mesmo que ainda de forma “manual” nesta fase).
5. **Documentar os resultados**:
   - Comparar o modelo com baseline simples (regra de negócio).
   - Simulação de impacto: “Se eu contatar top 10% de risco, quantos churners eu alcanço?”.