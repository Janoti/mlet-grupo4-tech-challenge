# Metodologia da EDA e Preparo da Base

## 1. O que e EDA e por que ela existe

EDA significa Exploratory Data Analysis, ou Analise Exploratoria de Dados. E a etapa em que o cientista de dados para, olha com cuidado para o que tem em maos e pergunta: "esse dado esta em condicoes de ser usado?"

Dois artigos que guiaram a construcao deste notebook resumem bem essa ideia:

O primeiro, de Soner Yildirim (Towards Data Science), comeca com uma frase direta: **"We should never just dump the raw data into a machine learning model. Garbage in, garbage out."** Ou seja, se voce joga dado sujo dentro de um modelo, o que sai e lixo. A EDA e exatamente o ato de limpar, entender e validar antes de modelar.

O segundo, de Miriam Santos (Towards Data Science), usa outra analogia igualmente forte: **"It's like running a diagnosis on your data."** Da mesma forma que um medico nao prescreveria remedio sem antes examinar o paciente, um cientista de dados nao deveria treinar modelo sem antes fazer o diagnostico do dado.

Essas duas ideias — lixo na entrada, lixo na saida; e diagnostico antes de prescricao — orientaram cada decisao deste notebook.

## 2. Pergunta central do notebook

Toda EDA precisa de uma pergunta que a guie. A pergunta deste notebook foi:

**A base `telecom_churn_base_extended.csv` esta pronta para treinar modelos baseline depois de um tratamento minimo e sem leakage metodologico?**

Essa pergunta foi escolhida porque a base foi gerada de forma propositalmente imperfeita. O objetivo da Fase 1 nao era so fazer graficos bonitos, mas verificar se a base poderia ser usada de forma tecnicamente defensavel.

## 3. Visao geral do fluxo

O notebook tem dez etapas. A ordem nao e arbitraria: ela separa claramente o que foi observado, o que foi corrigido e o que ficou pronto para modelagem.

```
0. Business Understanding       → por que esse problema existe e o que queremos prever
1. Entendimento da base         → carregamento, tipos, estrutura inicial
2. Qualidade dos dados          → o que esta quebrado na base bruta
3. Target e sinais de churn     → o que a base revela sobre o problema de negocio
4. Data readiness e leakage     → o que pode e o que nao pode entrar no modelo
5. Tratamento para baseline     → correcoes minimas e auditaveis
6. Split, imputacao e encoding  → preparar a matriz sem contaminar o treino com o teste
7. Validacao final              → provar que o preparo funcionou
8. Conclusao executiva          → resposta objetiva a pergunta central
9. Diagnostico de fairness      → verificar se o modelo tem vies por grupo demografico
```

## 4. Etapa 0: Business Understanding

### O que foi feito

Antes de qualquer codigo, o notebook apresenta:

- o problema de negocio: prever quais clientes de uma operadora de telecom vao cancelar o servico (churn)
- quem e afetado por esse problema e o que esta em jogo
- quais metricas de sucesso fazem sentido para avaliar o modelo

### Por que essa etapa existe

Um modelo de churn so faz sentido se quem o constroi entende o negocio por tras dele. Sem isso, e facil cometer erros como:

- incluir variaveis que o negocio nunca teria disponivel no momento da predicao
- escolher metricas erradas (acuracia num dataset desbalanceado, por exemplo, e uma metrica enganosa)
- nao conseguir explicar os resultados para quem vai usar o modelo

Essa etapa e a ancora conceitual de tudo que vem depois.

## 5. Etapa 1: Entendimento da base

### O que foi feito

- leitura do arquivo `data/raw/telecom_churn_base_extended.csv`
- identificacao das colunas principais (`customer_id` como identificador, `churn` como target)
- conversao de colunas de data para o tipo correto
- separacao inicial entre colunas numericas e categoricas
- inspecao de amostra e metadados gerais

### Por que essa etapa existe

Antes de qualquer analise, e necessario confirmar que voce esta lendo o arquivo certo e que os tipos de dado fazem sentido. E o equivalente a olhar o mapa antes de comecar uma viagem.

Erros classicos que essa etapa evita:

- usar o arquivo errado sem perceber
- interpretar uma data como texto e perder toda informacao temporal
- tratar uma coluna categorica como numerica e calcular media de CEP
- confundir target com feature

### Decisoes tomadas

- `customer_id` foi tratado como identificador, nao como feature. Ele distingue clientes, mas nao deve entrar no modelo porque nao carrega informacao preditiva.
- `churn` foi definido como target desde o inicio para evitar que qualquer analise subsequente trate ele como feature.
- As datas foram convertidas imediatamente para permitir analise temporal correta.

## 6. Etapa 2: Qualidade dos dados

### O que foi feito

- contagem de IDs duplicados
- contagem de linhas completamente duplicadas
- identificacao de colunas com valores ausentes (missing)
- identificacao de colunas constantes (que tem o mesmo valor em todas as linhas)
- identificacao de colunas com alta cardinalidade
- visualizacao da distribuicao de missings por coluna
- visualizacao de distribuicoes extremas com boxplots

### Por que essa etapa existe

Em uma base sintetica "suja", a primeira pergunta relevante nao e "qual variavel mais explica churn". E: "posso confiar nessa base para treinar alguma coisa?"

Miriam Santos coloca bem: *"Finding these data quality issues at the beginning of a project is critical. If not identified and addressed prior to model building, they can jeopardize the whole ML pipeline."* Problemas de qualidade encontrados depois do treino sao muito mais caros de corrigir do que os encontrados na EDA.

### Por que cada verificacao foi feita

**Duplicidade de linha:** uma linha completamente igual a outra nao traz informacao nova. Ela apenas infla a amostra e pode fazer o modelo "aprender" um mesmo exemplo mais de uma vez, distorcendo o treino.

**Duplicidade por `customer_id`:** um mesmo cliente aparecendo duas vezes no dataset pode criar uma situacao onde o mesmo individuo esta no treino e no teste ao mesmo tempo, o que invalida a avaliacao do modelo.

**Valores ausentes (missing):** dados faltantes nao sao so inconvenientes. Eles podem comprometer classificadores inteiros ou distorcer predicoes. A decisao de como trata-los — dropar, imputar ou manter — depende de quanto falta e de por que falta.

**Colunas constantes:** uma coluna que tem o mesmo valor em todas as linhas nao tem capacidade preditiva. Ela nao ajuda o modelo a distinguir clientes que churam de clientes que nao churam.

**Boxplots para outliers:** um outlier nao e automaticamente um erro. Pode ser um valor legitimo e extremo. O boxplot ajuda a visualizar onde estao esses valores e a decidir se merecem investigacao. Como Soner Yildirim explica, comparar a media com a mediana tambem e util: se a media e bem maior que a mediana, a distribuicao esta puxada para cima por valores extremos.

### Leitura correta dessa etapa

Tudo nessa secao descreve a **base bruta**, antes de qualquer correcao. Isso e importante para apresentacao: aqui estamos mostrando o que estava errado, nao o estado final.

## 7. Etapa 3: Distribuicao do target e sinais de churn

### O que foi feito

- medicao da proporcao de churn na base
- comparacao da taxa de churn por tipo de plano
- comparacao da taxa de churn por categoria de NPS
- comparacao da taxa de churn por inadimplencia
- distribuicoes de variaveis numericas separadas por churn/nao-churn
- correlacoes lineares simples com o target

### Por que essa etapa existe

Depois de entender a qualidade da base, o proximo passo e verificar se a base tem sinais minimos coerentes para o problema de negocio. Nao adianta tratar e modelar uma base que, no fundo, nao tem nenhuma informacao sobre o que queremos prever.

As perguntas que guiaram essa secao:

- o target esta muito raro ou razoavelmente distribuido?
- existem segmentos de clientes com churn mais alto?
- as variaveis numericas parecem carregar algum sinal?

**Por que o desbalanceamento importa:** em problemas de churn, a maioria dos clientes nao cancela. Uma base com 90% de nao-churn e 10% de churn e considerada desbalanceada. Um modelo que classifica todo mundo como "nao churn" acerta 90% das vezes — mas nao aprende nada util. Por isso, entender o balanceamento do target e essencial antes de escolher metricas e algoritmos.

### Por que essas variaveis foram escolhidas

- `plan_type`, `nps_category` e `default_flag` foram escolhidas porque fazem sentido de negocio: cliente inadimplente, insatisfeito ou com plano especifico tende a ter comportamento diferente.
- Foram usadas taxas de churn por grupo em vez de testes causais porque a Fase 1 e exploratoria, nao inferencial.
- As correlacoes foram mantidas como leitura auxiliar, nao como criterio unico de selecao. Churn dificilmente depende so de relacoes lineares simples.

### Importante

Essas analises ainda foram feitas na base bruta. Isso foi intencional: o objetivo era entender o comportamento original antes de alterar qualquer registro.

## 8. Etapa 4: Data readiness e risco de leakage

### O que foi feito

- classificacao de cada coluna por papel: ID, target ou feature
- marcacao de colunas com risco de leakage
- marcacao de colunas sensiveis a janela temporal
- geracao de uma recomendacao de acao por coluna
- checklist executivo de readiness da base

### Por que essa etapa existe

Nem toda coluna disponivel deve entrar no modelo. Em problemas de churn, alguns campos ficam perigosamente proximos do evento que queremos prever, ou representam acoes que so acontecem depois que o churn ja ocorreu.

Se essas colunas entrarem no treino, o modelo aprende um atalho indevido: ele "ve o futuro" durante o treino e depois falha completamente em producao, quando esse futuro nao existe. Isso se chama **data leakage**.

Como Miriam Santos descreve, identificar leakage na fase de EDA e a forma mais barata de preveni-lo. Descobrir leakage depois que o modelo esta em producao e catastrofico.

### O que foi excluido e por que

| Coluna | Motivo da exclusao |
|---|---|
| `customer_id` | Identificador. Nao carrega informacao preditiva. |
| `churn_probability` | Artefato do gerador sintetico. Representa diretamente o target e causaria leakage total. |
| `retention_offer_made` | Acao de retencao que so existe porque o churn ja estava proximo. Variavel posterior ao evento. |
| `retention_offer_accepted` | Mesma logica: resposta a uma oferta que so foi feita por causa do risco de churn. |
| `contract_renewal_date` | Data com alto risco de leakage temporal: saber a data de renovacao implica saber quando o contrato termina, o que esta diretamente ligado ao momento do churn. |
| `loyalty_end_date` | Mesma logica de leakage temporal: implica informacao sobre o estado do vinculo do cliente no momento do evento. |

### Variaveis com aviso de janela temporal

Colunas como `usage_delta_pct`, `days_past_due` e `support_calls_30d` foram marcadas como **sensiveis a janela temporal**. Elas podem ser usadas, mas exigem governanca clara em producao: a janela de calculo precisa ser sempre anterior ao momento da predicao.

### Por que isso importa na apresentacao

Essa etapa mostra maturidade metodologica. Em vez de usar tudo que existe, o notebook justifica o que entra e o que sai da modelagem. Isso e algo que qualquer banca tecnica vai perguntar.

## 9. Etapa 5: Tratamento da base para baseline

### O que foi feito

- copia da base original para uma base de trabalho (a base original nunca e alterada)
- auditoria de duplicidade por `customer_id` com medicao do impacto
- remocao de linhas duplicadas exatas
- remocao de duplicidade por ID, mantendo a primeira ocorrencia
- conversao de valores fora de faixa para `NaN`
- limpeza de categorias invalidas
- correcao de inconsistencias logicas simples

### Por que essa etapa existe

O objetivo nao era construir o melhor pipeline possivel. Era um tratamento minimo, reproduzivel e justificavel para permitir o treino de baselines.

### Decisoes detalhadas

**Auditoria antes de remover:** antes de apagar qualquer duplicado, o notebook mede quantos existem, quantas linhas participam e se ha conflito no target. Isso nao e so burocracia — e o que permite defender a decisao em apresentacao com numeros.

**Manter a primeira ocorrencia por `customer_id`:** decisao pragmatica para Fase 1. Em um pipeline maduro, a regra seria temporal (manter o registro mais recente) ou consolidada (agregar registros do mesmo cliente). Aqui, a regra foi simplificada e documentada.

**Valores fora de faixa viram `NaN`, nao valores "corrigidos":** para colunas como idade, NPS, CSAT e indicadores de taxa, valores impossiveis nao foram substituidos por um numero inventado. Eles foram marcados como ausentes. Isso preserva a informacao de que o registro estava inconsistente e permite imputacao controlada depois.

**Categorias invalidas viram `NaN`:** quando uma categoria nao fazia parte do dominio esperado, ela foi tratada como ausente. Em Fase 1, o objetivo e estabilizar a base, nao criar novas classes.

**Inconsistencias logicas corrigidas deterministicamente:** alguns problemas tinham solucao obvia, por exemplo:
- cliente sem fidelidade nao deveria ter meses restantes de fidelidade
- cliente com atraso confirmado nao deveria ter `default_flag = 0`
- oferta aceita nao deveria coexistir com `retention_offer_made = 0`

Essas correcoes sao diretamente verificaveis e nao dependem de suposicoes estatisticas.

## 10. Etapa 6: Split, imputacao e matriz final sem leakage

### O que foi feito

1. remocao das colunas que nao devem entrar na modelagem
2. separacao entre features (`X`) e target (`y`)
3. split estratificado 80/20 em treino e teste
4. imputacao de numericas com mediana do treino
5. imputacao de categoricas com moda do treino
6. capping de outliers por IQR, calculado no treino e aplicado no teste
7. one-hot encoding em treino e teste
8. alinhamento das colunas do teste com as colunas do treino

### Por que essa etapa existe

Essa e a etapa que transforma a base tratada em uma matriz pronta para modelos baseline. E tambem onde o risco de leakage e maior, porque e aqui que as transformacoes sao aprendidas.

### A decisao mais importante: split antes de tudo

O split acontece antes da imputacao, do capping e do encoding. Isso nao e detalhe tecnico — e a correcao metodologica principal do notebook.

Se a mediana, a moda ou as categorias fossem calculadas usando a base inteira, o treino estaria recebendo informacao do teste sem saber. O modelo "aprenderia" algo sobre dados que nunca deveria ter visto. Em producao, esse comportamento nao existiria, e o modelo falharia.

A regra e simples: qualquer transformacao que "aprende algo dos dados" deve aprender **so com o treino** e ser aplicada no teste de forma passiva.

### Por que 80/20 estratificado

- 80/20 e uma proporcao simples e amplamente aceita para baseline
- a estratificacao preserva a taxa de churn entre treino e teste, garantindo que ambos representam o mesmo problema
- sem estratificacao, e possivel ter um teste com proporcao de churn muito diferente do treino, o que invalida a comparacao de metricas

### Por que mediana para numericas

A mediana e robusta a outliers. Se uma coluna tem alguns valores extremos, a media seria puxada por eles e imputaria um valor que nao representa o comportamento tipico. A mediana ignora esses extremos e representa melhor o centro da distribuicao.

### Por que moda para categoricas

A moda e a categoria mais frequente. Em uma impute simples de Fase 1, substituir ausentes pela categoria mais comum e uma solucao interpretavel e suficiente para tirar a base do estado invalido.

### Por que capping de outliers por IQR

Apos a imputacao, valores extremos que sobreviveram ao tratamento anterior sao limitados ao intervalo definido pelo IQR (distancia entre o primeiro e o terceiro quartil). Isso nao elimina os outliers, mas controla seu impacto sobre o treino do modelo. O limite e calculado no treino e aplicado tanto no treino quanto no teste.

### Por que one-hot encoding

- e o padrao para modelos tabulares simples
- torna cada categoria uma coluna binaria, explicita e interpretavel
- evita impor ordem artificial entre categorias (o que aconteceria com label encoding)

### Por que alinhar colunas apos o encoding

Depois do `get_dummies`, treino e teste podem gerar conjuntos de colunas diferentes — por exemplo, se uma categoria so aparece no treino ou so no teste. O teste e reindexado com base nas colunas do treino para garantir que ambos tenham a mesma estrutura de entrada.

## 11. Etapa 7: Validacao final da base preparada

### O que foi feito

- verificacao de IDs duplicados apos tratamento
- verificacao de linhas duplicadas apos tratamento
- verificacao de missing no treino e no teste
- verificacao de ausencia das colunas de leakage na matriz final
- verificacao de colunas constantes no treino
- comparacao da taxa de churn entre base bruta, base tratada, treino e teste

### Por que essa etapa existe

Sem validacao, o notebook terminaria exatamente no ponto em que a maioria assume que "deu certo" sem provar.

Essa secao transforma a preparacao em algo auditavel. Cada verificacao responde uma pergunta especifica:

| Verificacao | O que prova |
|---|---|
| 0 missing no treino e teste | A imputacao foi suficiente. Nao sobrou dado faltante. |
| 0 leakage na matriz final | As colunas excluidas realmente ficaram fora do treino. |
| 0 duplicidade apos tratamento | A base final nao herdou os problemas estruturais da base bruta. |
| Taxa de churn identica entre treino e teste | A estratificacao preservou o balanceamento do target. |

### Por que isso importa

Em uma apresentacao, qualquer pessoa pode perguntar: "voce tem certeza que nao tem leakage?" ou "o teste e representativo do treino?". Essa etapa permite responder com evidencia, nao com suposicao.

## 12. Etapa 8: Conclusao executiva

### O que a conclusao sintetiza

1. a base tem sinais de negocio coerentes para churn
2. a base bruta tinha problemas reais de qualidade
3. depois do tratamento e da validacao, a base ficou pronta para treino de baseline

### Por que essa etapa existe

Nenhuma apresentacao termina bem se o publico sair sem saber a resposta da pergunta principal. A conclusao executiva responde de forma objetiva:

- o dataset serve para o problema?
- havia problemas relevantes?
- eles foram tratados de forma consistente?
- a proxima etapa pode comecar com seguranca?

## 13. Etapa 9: Diagnostico inicial de fairness

### O que foi feito

Usando a biblioteca Fairlearn, o notebook verifica se o modelo baseline apresenta disparidade de desempenho entre grupos demograficos — por exemplo, se ele funciona de forma significativamente pior para clientes de determinada faixa etaria, genero ou regiao.

### Por que essa etapa existe

Um modelo pode ter boa acuracia geral e ainda assim ser injusto com grupos especificos. Isso nao e so uma questao etica — e um risco regulatorio e de reputacao para o negocio.

Miriam Santos destaca que a analise de bias e parte da EDA responsavel: descobrir que uma feature esta correlacionada com um atributo protegido (como raca ou genero) ja na fase de exploracao permite tomar decisoes conscientes antes de colocar o modelo em producao.

O diagnostico de fairness nesta fase nao e definitivo — e um alerta inicial que orienta as fases seguintes.

## 14. Resumo das principais escolhas metodologicas

### Escolha 1: analisar a base bruta antes de tratar

Motivo: mostrar transparencia sobre os problemas encontrados e evitar mascarar erros antes do diagnostico.

### Escolha 2: separar readiness de tratamento

Motivo: deixar claro o que era problema original da base e o que passou a ser estado da base preparada.

### Escolha 3: auditar duplicados antes de remover

Motivo: fortalecer a justificativa em apresentacao e mostrar o impacto quantificado do problema.

### Escolha 4: transformar valores invalidos em `NaN`

Motivo: tratar inconsistencias sem inventar substituicoes arbitrarias.

### Escolha 5: split antes de imputacao, capping e encoding

Motivo: evitar leakage metodologico. Qualquer transformacao que aprende dos dados deve aprender so com o treino.

### Escolha 6: usar mediana e moda

Motivo: simplicidade, robustez a outliers e boa adequacao para baseline de Fase 1.

### Escolha 7: capping por IQR calculado no treino

Motivo: controlar o impacto de outliers sem remove-los, mantendo a janela de aprendizado restrita ao treino.

### Escolha 8: validar a base final explicitamente

Motivo: garantir que a preparacao terminou em estado realmente apto para modelagem, com evidencias documentadas.

## 15. Limitacoes assumidas nesta fase

Algumas decisoes foram intencionalmente simples porque o foco era baseline:

- a regra de manter a primeira ocorrencia por `customer_id` e pragmatica, nao definitiva
- nao houve engenharia de features mais sofisticada
- nao foi implementado pipeline reutilizavel ainda
- a governanca temporal foi apontada, mas nao operacionalizada no notebook
- o diagnostico de fairness e inicial e precisa ser aprofundado nas fases seguintes

Esses pontos nao invalidam a EDA. Eles apenas delimitam o escopo da Fase 1 e mostram consciencia sobre o que ainda falta.

## 16. Proxima etapa recomendada

Com a base preparada e validada, o passo seguinte e usar exatamente o mesmo criterio de preparo no notebook de baselines e, depois, transformar esse fluxo em pipeline reutilizavel.

Em termos praticos, a sequencia recomendada e:

1. treinar baselines em `notebooks/02_baselines.ipynb`
2. comparar modelos por AUC-ROC, PR-AUC e top-K
3. consolidar o preparo em pipeline reprodutivel
4. documentar o modelo final no model card

## 17. Referencias

- Yildirim, S. (2020). *A Practical Guide for Exploratory Data Analysis — Churn Dataset*. Towards Data Science.
- Santos, M. (2023). *A Data Scientist's Essential Guide to Exploratory Data Analysis*. Towards Data Science.
