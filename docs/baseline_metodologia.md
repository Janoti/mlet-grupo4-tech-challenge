# Metodologia dos Baselines e Avaliacao dos Modelos

## 1. O que e um baseline e por que ele existe

Antes de construir qualquer modelo sofisticado, existe uma pergunta que precisa ser respondida: **o dado tem sinal suficiente para prever churn?**

A forma mais honesta de responder essa pergunta e comecar pelo mais simples possivel. Esse modelo simples se chama **baseline**.

Um artigo de Soner Yildirim (Towards Data Science) resume bem a logica: se um modelo complexo nao consegue superar um modelo simples, o problema nao e o algoritmo — e o dado, a preparacao, ou a propria definicao do problema. Comecar pelo simples evita gastar esforco em complexidade que nao se justifica.

O segundo artigo que guiou este notebook, de Tam D Tran-The (Towards Data Science), adiciona uma dimensao critica: **a escolha da metrica de avaliacao e tao importante quanto a escolha do modelo**. Usar a metrica errada em dados desbalanceados — como e o caso de churn — pode fazer um modelo inutil parecer excelente.

Essas duas ideias — comece pelo simples, meça com honestidade — orientaram cada decisao deste notebook.

## 2. Pergunta central do notebook

**Os dados preparados na EDA permitem treinar modelos que superam um preditor aleatorio de forma consistente, auditavel e com valor de negocio mensuravel?**

Essa pergunta tem tres partes deliberadas:

- **superar um preditor aleatorio**: a referencia minima e um modelo que nao aprende nada — se nao passamos disso, algo esta errado
- **de forma consistente**: um modelo que funciona por sorte em um split especifico nao e util — precisamos de estabilidade
- **com valor de negocio mensuravel**: metrica tecnica sem traducao de negocio nao fundamenta decisoes — precisamos conectar o modelo a impacto real

## 3. Visao geral do fluxo

O notebook tem nove etapas:

```
1. Setup e bibliotecas           → ferramentas e classes utilitarias
2. Carregamento do dataset       → base bruta do telecom
3. Tratamento minimo dos dados   → replicar o preparo da EDA de forma pipeline
4. Split e preprocessamento      → separar target, dividir e montar pipeline sklearn
5. Treino dos baselines          → quatro modelos + logging no MLflow
5a. Analise de overfitting       → comparar desempenho treino vs teste
5b. Tuning de regularizacao      → GridSearchCV no parametro C
5c. Comparacao de penalizacoes   → L1 vs L2 vs ElasticNet
6. Avaliacao de fairness         → disparidade por grupo sensivel
7. Mitigacao de fairness         → correcao com Equalized Odds
8. Tabela comparativa final      → todos os modelos lado a lado
9. Conclusao                     → resposta objetiva a pergunta central
```

## 4. Etapa 1: Setup e bibliotecas

### O que foi feito

Carregamento das bibliotecas necessarias e definicao de classes utilitarias, incluindo o `IQRCapper` (transformador customizado para capping de outliers por IQR) e a funcao `compute_business_metrics`, que calcula o valor financeiro de cada modelo.

### Por que essa etapa existe

Centralizar configuracoes e utilitarios no inicio evita codigo repetido e garante que toda logica critica — especialmente o calculo do valor de negocio — seja definida uma unica vez e usada de forma consistente.

## 5. Etapa 2 e 3: Carregamento e tratamento dos dados

### O que foi feito

O notebook recarrega a base bruta `data/raw/telecom_churn_base_extended.csv` e replica o mesmo tratamento feito na EDA:

- remocao de duplicados por linha e por `customer_id`
- correcao de inconsistencias logicas
- remocao das colunas de leakage

### Por que o tratamento foi refeito aqui

A EDA e este notebook poderiam compartilhar o mesmo arquivo processado. A escolha de refazer o tratamento aqui foi pragmatica: garante que o notebook e auto-contido e reproduzivel, sem depender de um artefato externo que poderia estar desatualizado.

As colunas excluidas sao as mesmas documentadas na EDA:

| Coluna | Motivo |
|---|---|
| `customer_id` | Identificador sem valor preditivo |
| `churn_probability` | Leakage direto do target |
| `retention_offer_made` | Variavel posterior ao churn |
| `retention_offer_accepted` | Variavel posterior ao churn |
| `contract_renewal_date` | Leakage temporal |
| `loyalty_end_date` | Leakage temporal |

## 6. Etapa 4: Definicao de alvo, split e preprocessamento

### O que foi feito

- separacao entre features (`X`) e target (`y = churn`)
- split estratificado 80/20 em treino e teste
- construcao de um `Pipeline` do scikit-learn com duas branches:
  - **numericas**: imputacao por mediana → `StandardScaler`
  - **categoricas**: imputacao por moda → `OneHotEncoder`

### Por que usar Pipeline do scikit-learn

Um Pipeline garante que todo o preprocessamento e aplicado de forma sequencial e consistente. Mais importante: ele respeita a regra de ouro da separacao treino/teste — as estatisticas de imputacao, escala e encoding sao aprendidas so com o treino e aplicadas passivamente no teste.

Sem Pipeline, e muito facil cometer o erro de reajustar o `StandardScaler` no teste, o que causaria leakage silencioso.

### Por que adicionar `StandardScaler` aqui

Na EDA, o preprocessamento nao incluia escalonamento. Aqui, o `StandardScaler` foi adicionado porque:

- Regressao Logistica com regularizacao L1 ou L2 e sensivel a escala das variaveis
- Sem escalonamento, variaveis com magnitude maior (como `monthly_charges` vs. um indicador binario) influenciam desproporcionalmente o modelo

## 7. Etapa 5: Treino dos baselines e logging no MLflow

### O que foi feito

Quatro modelos foram treinados em sequencia:

1. `DummyClassifier (stratified)` — preditor aleatorio estratificado
2. `LogisticRegression` — regressao logistica com regularizacao L2 padrao
3. `RandomForestClassifier` — floresta com 200 arvores e profundidade maxima 10
4. `GradientBoostingClassifier` — boosting com 100 arvores e taxa de aprendizado 0.1

Todos foram logados no MLflow com parametros, metricas e versao do dataset (hash SHA256 do CSV).

### Por que essa ordem de modelos

A ordem reflete uma progressao deliberada de complexidade:

**DummyClassifier** e o piso absoluto. Um modelo que sorteia predicoes proporcional as classes do treino. Se qualquer modelo real nao supera isso, algo esta fundamentalmente errado com o dado ou com o preparo.

**LogisticRegression** e o baseline real. E o modelo mais simples que ainda aprende alguma coisa. Como Yildirim explica, a regressao logistica produz coeficientes que podem ser transformados em **Odds Ratios** — um formato diretamente legivel para pessoas de negocio: "clientes com contrato mensal tem X vezes mais chance de churnar do que clientes com contrato anual."

**RandomForest e GradientBoosting** sao os modelos de referencia de maior capacidade. Eles capturam relacoes nao-lineares e interacoes entre variaveis. A comparacao com a regressao logistica revela se a complexidade adicional se justifica.

### Por que logar no MLflow

O MLflow registra cada execucao com suas metricas, parametros e a versao exata do dataset. Isso garante que qualquer resultado possa ser reproduzido e comparado de forma rastreavel. Em um projeto em grupo, sem rastreamento, e impossivel saber se uma melhora de metrica veio de um modelo melhor ou de uma mudanca silenciosa no preprocessamento.

### Resultado dos modelos

| Modelo | ROC-AUC | Acuracia | F1 |
|---|---|---|---|
| DummyClassifier | ~0.50 | ~base rate | ~base rate |
| LogisticRegression | 0.876 | — | — |
| RandomForest | similar ao LR | — | — |
| GradientBoosting | similar ao LR | — | — |

O resultado mais importante nao e qual modelo ganhou, mas que todos superaram claramente o Dummy — confirmando que o dado tem sinal real para prever churn.

## 8. Etapa 5a: Analise de overfitting

### O que foi feito

Para cada modelo, foram calculadas as metricas tanto no treino quanto no teste. O delta entre elas foi registrado:

- `delta_accuracy = accuracy_treino - accuracy_teste`
- `delta_f1 = f1_treino - f1_teste`
- `delta_roc_auc = roc_auc_treino - roc_auc_teste`

Um flag de alerta foi ativado quando `delta_roc_auc > 0.05`.

### Por que isso importa

Um modelo que aprende muito bem o treino mas falha no teste nao aprendeu o problema — ele memorizou os dados. Esse fenomeno se chama **overfitting**.

Um delta de ROC-AUC pequeno (menor que 0.05) indica que o modelo generaliza bem: ele aprendeu padroes reais, nao ruido especifico do treino. A regressao logistica com C=0.1 mostrou delta de apenas 0.0013 — excelente generalizacao.

## 9. Etapa 5b e 5c: Tuning de regularizacao

### O que foi feito

**Etapa 5b:** GridSearchCV com validacao cruzada estratificada em 5 folds testou seis valores de C: `[0.001, 0.01, 0.1, 1, 10, 100]`. O melhor C encontrado foi **0.1**, com ROC-AUC de 0.8782 na CV.

**Etapa 5c:** Tres tipos de penalizacao foram comparados com seus respectivos melhores C:

| Penalizacao | Melhor C | ROC-AUC teste |
|---|---|---|
| L2 (Ridge) | 100 | 0.8762 |
| L1 (Lasso) | 0.1 | 0.8780 |
| ElasticNet | 0.1 | 0.8783 |

### O que e regularizacao e por que ela importa

Regularizacao e um mecanismo que penaliza o modelo por se tornar muito complexo. Sem regularizacao, a regressao logistica tende a ajustar os coeficientes de forma muito especifica ao treino, o que causa overfitting.

- **L2 (Ridge)**: penaliza coeficientes grandes, distribuindo o peso entre variaveis correlacionadas
- **L1 (Lasso)**: pode zerar coeficientes, funcionando como selecao automatica de features
- **ElasticNet**: combinacao de L1 e L2

### Por que L2 foi escolhido mesmo sem ser o melhor

A diferenca entre os tres tipos foi menor que 0.0002 de ROC-AUC. Em cenarios assim, o criterio de desempate deve ser **simplicidade e interpretabilidade**. L2 e o padrao mais amplamente conhecido e documentado, e essa margem nao justifica a complexidade adicional do ElasticNet.

### O que e validacao cruzada estratificada

Em vez de usar um unico split para avaliar o modelo, a validacao cruzada divide o treino em 5 partes (folds). Em cada iteracao, 4 partes sao usadas para treinar e 1 para validar. O resultado e a media das 5 avaliacoes.

A estratificacao garante que cada fold mantem a mesma proporcao de churn que o treino completo. Isso e especialmente importante em dados desbalanceados.

## 10. Etapa 6: Avaliacao de fairness

### O que foi feito

Foram calculadas metricas de fairness para quatro atributos sensiveis:

| Atributo | Por que e sensivel |
|---|---|
| `gender` | Protecao demografica (LGPD) |
| `age_group` | Protecao demografica (LGPD) |
| `region` | Equidade geografica (regulacao Anatel) |
| `plan_type` | Proxy socioeconomico (pre-pago indica menor renda) |

As metricas usadas foram:

- **Demographic Parity Difference (dp_diff)**: diferenca na taxa de predicao positiva entre grupos. Se o modelo preve churn muito mais para um grupo do que para outro, mesmo com perfis similares, isso e um sinal de vies.
- **Equalized Odds Difference (eo_diff)**: diferenca na taxa de verdadeiro positivo e na taxa de falso positivo entre grupos. Um modelo justo deve errar de forma similar para todos os grupos.

### Resultados encontrados

| Atributo | dp_diff | eo_diff | Interpretacao |
|---|---|---|---|
| gender | 0.055 | 0.074 | Disparidade modesta |
| age_group | 0.110 | 0.076 | Gap maior por idade |
| region | 0.157 | 0.167 | Disparidade geografica relevante |
| plan_type | 0.296 | 0.184 | **Maior gap — risco regulatorio** |

### Por que isso importa

Um modelo pode ter boa metrica geral e ainda assim ser injusto com grupos especificos. O gap de `plan_type` de 0.296 significa que o modelo preve churn de forma significativamente diferente entre clientes pre-pagos e pos-pagos. Em producao, isso poderia resultar em acoes de retencao concentradas desproporcionalmente em um perfil socioeconomico, o que tem implicacoes eticas e regulatorias.

## 11. Etapa 7: Mitigacao de fairness

### O que foi feito

Foi aplicado o algoritmo `ExponentiatedGradient` com a restricao `EqualizedOdds` para o atributo `gender`. O treinamento usou uma amostra estratificada de 15.000 exemplos por razoes de custo computacional.

### Resultado

- `dp_diff_gender` caiu de 0.0552 para 0.0518 (reducao de 6%)
- O custo da mitigacao em valor de negocio foi de R$ 9.850 — uma diferenca considerada aceitavel

### Limitacao assumida

A mitigacao foi aplicada apenas para `gender`. Os gaps de `age_group`, `region` e `plan_type` permanecem sem tratamento nesta fase. Esse e um ponto explicito de trabalho futuro.

## 12. Por que acuracia nao e a metrica certa para churn

Esse e um dos pontos mais importantes do notebook e merece explicacao direta.

Em uma base de churn, a maioria dos clientes **nao** cancela. Se a base tem, por exemplo, 73% de nao-churn, um modelo que simplesmente responde "nao vai churnar" para todo mundo acerta 73% das vezes. Esse modelo e inutil — ele nunca previu um churn sequer — mas a acuracia parece razoavel.

Isso se chama **paradoxo da acuracia**, e e o motivo pelo qual acuracia sozinha nao deve ser usada em problemas desbalanceados.

### A matriz de confusao como ferramenta honesta

```
                  Previu Nao Churn    Previu Churn
Churn Real Nao         TN                 FP
Churn Real Sim         FN                 TP
```

Cada celula tem um custo diferente para o negocio:

- **FN (Falso Negativo)**: cliente que ia churnar nao foi identificado. Resultado: cliente perdido, receita perdida.
- **FP (Falso Positivo)**: cliente leal foi identificado como risco. Resultado: custo de acao de retencao desperdicado.

A decisao sobre qual erro e mais caro e de negocio, nao de algoritmo. E ela define qual metrica priorizar e qual threshold usar.

## 13. ROC-AUC vs PR-AUC: qual metrica usar e quando

### O que e ROC-AUC

A curva ROC plota a taxa de verdadeiros positivos (recall) contra a taxa de falsos positivos (FPR) enquanto variamos o threshold de classificacao. A area sob essa curva (AUC) resume o desempenho do modelo em todos os thresholds possiveis.

**Interpretacao intuitiva:** a probabilidade de que, dado um cliente que churna e um que nao churna sorteados aleatoriamente, o modelo rankeie o que churna com score mais alto.

**Problema com dados desbalanceados:**

A taxa de falsos positivos e calculada como `FP / (TN + FP)`. Quando a classe negativa e muito grande (muitos clientes que nao churam), o denominador e enorme. Mesmo que o modelo gere muitos falsos alarmes em numeros absolutos, a FPR permanece pequena.

O artigo de Tam D Tran-The ilustra isso de forma precisa: em uma base com 1.000 positivos e 10.000 negativos, gerar 1.600 falsos positivos resulta em FPR de apenas 0.16 — identico ao que ocorreria com 160 falsos positivos numa base balanceada de 1.000/1.000. A curva ROC ve os dois casos como iguais. O negocio claramente nao.

A frase do paper de referencia sintetiza bem: **"O grafico ROC passa uma impressao inocente. O grafico PR revela a verdade amarga."**

### O que e PR-AUC

A curva PR plota precisao contra recall enquanto variamos o threshold. A area sob ela resume o desempenho na classe positiva (churn) em todos os niveis de cobertura.

**Por que e mais honesta em dados desbalanceados:**

Precisao e calculada como `TP / (TP + FP)`. Verdadeiros negativos nao aparecem em lugar nenhum nessa formula. Isso significa que ter muitos TN (clientes que nao churam identificados corretamente) nao "infla" a precisao — ela mede exclusivamente o que acontece com as predicoes de churn.

**A linha de base da curva PR:**

Na curva ROC, o classificador aleatorio e sempre uma diagonal com AUC = 0.5, independente do desbalanceamento. Na curva PR, a linha de base e igual a **prevalencia da classe positiva**. Se apenas 10% dos clientes churam, a linha de base e 0.10. Isso torna imediatamente visivel o quao dificil o problema e e o quao longe o modelo esta de uma referencia honesta.

### Quando usar cada uma

| Situacao | Metrica recomendada |
|---|---|
| Dados balanceados | ROC-AUC e suficiente |
| Dados desbalanceados (churn, fraude, deteccao de doenca) | PR-AUC como primaria |
| Comunicacao com tecnico | Ambas, lado a lado |
| Comunicacao com negocio | Confusion matrix + valor financeiro |

**Neste notebook**, ROC-AUC foi usada como metrica primaria de selecao de modelo, e PR-AUC foi reportada como metrica complementar. Ambas foram registradas no MLflow.

## 14. O valor de negocio como metrica

### Como foi calculado

O notebook define uma funcao `compute_business_metrics` com os seguintes parametros:

- Valor retido por churn corretamente identificado (TP): **R$ 500**
- Custo de acao de retencao por contato (TP + FP): **R$ 50**
- **Valor Liquido = TP × R$500 − (TP + FP) × R$50**

### Por que isso importa

Uma metrica tecnica como ROC-AUC nao responde a pergunta que o negocio realmente faz: **quanto esse modelo vale em reais?**

O modelo de regressao logistica selecionado gerou um Valor Liquido de **R$ 1.190.800**, identificando 2.731 verdadeiros churners com 763 falsos positivos, abordando 3.494 clientes no total. Esse numero e 2.1 vezes maior do que o baseline dummy.

Apresentar resultados em valor financeiro transforma o modelo de um exercicio tecnico em uma decisao de negocio justificavel.

## 15. Modelo selecionado e justificativa

**Modelo escolhido: Regressao Logistica com L2, C=0.1**

| Criterio | Valor | Interpretacao |
|---|---|---|
| ROC-AUC teste | 0.8783 | Excelente capacidade de rankear risco |
| delta_roc_auc | 0.0013 | Sem overfitting |
| Valor Liquido | R$ 1.190.800 | 2.1x melhor que o baseline dummy |
| Interpretabilidade | Alta | Coeficientes como Odds Ratios |
| Complexidade | Baixa | Facilmente auditavel e explicavel |

A escolha de L2 sobre ElasticNet (que teve desempenho marginalmente superior em 0.0001 de ROC-AUC) seguiu o principio de Occam: quando a diferenca e irrelevante, a solucao mais simples e preferivel.

## 16. Resumo das principais escolhas metodologicas

### Escolha 1: comecar pelo DummyClassifier

Motivo: estabelecer o piso absoluto. Qualquer modelo deve superar o preditor aleatorio para ser considerado util.

### Escolha 2: usar Regressao Logistica como baseline principal

Motivo: e simples, interpretavel, produz probabilidades e permite traducao direta para Odds Ratios legidos por pessoas de negocio.

### Escolha 3: usar Pipeline do scikit-learn

Motivo: garantir que o preprocessamento seja aprendido apenas no treino e aplicado passivamente no teste, eliminando leakage silencioso.

### Escolha 4: adicionar StandardScaler

Motivo: regressao logistica com regularizacao e sensivel a escala. Variaveis em escalas diferentes distorcem a regularizacao.

### Escolha 5: usar ROC-AUC como metrica primaria, com PR-AUC como complementar

Motivo: ROC-AUC permite comparacao consistente entre modelos; PR-AUC e mais honesta para dados desbalanceados. Ambas sao necessarias para uma avaliacao completa.

### Escolha 6: calcular valor de negocio

Motivo: conectar o desempenho tecnico a impacto financeiro mensuravel, tornando a escolha do modelo justificavel para stakeholders nao-tecnicos.

### Escolha 7: avaliar fairness como parte do pipeline

Motivo: um modelo tecnicamente bom pode ser injusto com grupos especificos. Avaliar fairness na fase de baseline e a forma mais barata de detectar e documentar esses riscos.

### Escolha 8: logar tudo no MLflow

Motivo: rastreabilidade e reproducibilidade. Em um projeto em grupo, sem logging, e impossivel saber qual versao de qual experimento gerou qual resultado.

## 17. Limitacoes assumidas nesta fase

- Desbalanceamento de classes nao foi tratado explicitamente (sem SMOTE, sem `class_weight`)
- Threshold de 0.5 foi mantido como padrao; tuning de threshold orientado por custo e trabalho futuro
- Mitigacao de fairness foi aplicada apenas para `gender`; outros atributos sensiveis permanecem sem tratamento
- Interpretabilidade dos modelos (coeficientes, importancia de features, VIF) nao foi incluida nesta fase
- RandomForest e GradientBoosting foram treinados como referencia, mas nao foram aprofundados

## 18. Proxima etapa recomendada

Com os baselines avaliados e o modelo selecionado, o caminho natural e:

1. explorar modelos mais avancados (MLP, XGBoost, LightGBM) em `notebooks/03_advanced_models.ipynb`
2. implementar tuning de threshold orientado por custo de negocio
3. tratar o desbalanceamento com class weighting ou tecnicas de resampling
4. aprofundar interpretabilidade com analise de coeficientes e importancia de features
5. expandir mitigacao de fairness para `age_group`, `region` e `plan_type`

## 19. Referencias

- Yildirim, S. (2020). *Churn Prediction with Machine Learning*. Towards Data Science.
- Tran-The, T. D. (2021). *Precision-Recall Curve Is More Informative Than ROC in Imbalanced Data*. Towards Data Science.
- Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE.
