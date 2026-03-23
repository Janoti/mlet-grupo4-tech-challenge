# Metodologia da EDA e Preparo da Base

## 1. Objetivo deste documento

Este documento explica, em linguagem direta, o que foi feito no notebook `notebooks/01_eda.ipynb`, por que cada etapa existe e qual foi a justificativa das principais escolhas metodologicas.

A ideia nao e apenas descrever comandos, mas registrar a logica do trabalho para que qualquer pessoa do grupo consiga:

- entender o fluxo completo da EDA
- defender as decisoes em apresentacao
- reproduzir o preparo da base para baseline
- separar claramente diagnostico, tratamento e validacao

## 2. Pergunta central da EDA

A pergunta que guiou o notebook foi:

**a base `telecom_churn_base_extended.csv` esta pronta para treinamento de baselines depois do tratamento minimo necessario e sem leakage metodologico?**

Essa pergunta e importante porque a base foi gerada de forma propositalmente imperfeita. Ou seja, o objetivo da Fase 1 nao era apenas fazer graficos bonitos, mas verificar se a base poderia ser usada de forma tecnicamente defensavel.

## 3. Visao geral do fluxo

O notebook foi organizado em oito etapas:

1. Carregamento e contexto da base
2. Qualidade estrutural dos dados
3. Distribuicao do target e sinais de churn na base bruta
4. Data readiness e risco de leakage
5. Tratamento da base para baseline
6. Split, imputacao e matriz final sem leakage
7. Validacao final da base preparada
8. Conclusao executiva

Essa ordem foi escolhida para manter uma separacao clara entre:

- o que foi observado na base original
- o que foi corrigido
- o que ficou apto para modelagem

## 4. Etapa 1: carregamento e contexto da base

### O que foi feito

- leitura do arquivo `data/raw/telecom_churn_base_extended.csv`
- definicao de colunas principais, como `customer_id` e `churn`
- conversao de colunas de data
- identificacao inicial de colunas numericas e categoricas
- exibicao de amostra da base e metadados gerais

### Por que essa etapa existe

Antes de qualquer analise, e necessario confirmar que o notebook esta lendo a base correta e que os tipos de dados fazem sentido. Essa etapa evita erros basicos, como:

- usar arquivo errado
- interpretar datas como texto
- tratar colunas categoricas como numericas
- misturar target com feature sem perceber

### Por que essas escolhas foram feitas

- `customer_id` foi tratado como identificador e nao como feature porque ele distingue clientes, mas nao deve ser usado para treinar modelo.
- `churn` foi definido como target porque representa a variavel de negocio que queremos prever.
- As datas foram convertidas logo no inicio para permitir leitura mais correta da estrutura da base.

## 5. Etapa 2: qualidade estrutural dos dados

### O que foi feito

- contagem de IDs duplicados
- contagem de linhas duplicadas
- identificacao de colunas com missing
- identificacao de colunas constantes
- identificacao de alta cardinalidade
- visualizacao de missing por coluna
- visualizacao de distribuicoes extremas com boxplots

### Por que essa etapa existe

Em uma base sintetica "suja", a primeira pergunta relevante nao e "qual variavel mais explica churn", mas sim "posso confiar nessa base para treinar alguma coisa?".

Se o dataset tiver duplicidade, categorias invalidas, missing ou valores absurdos, qualquer baseline treinado depois pode parecer melhor ou pior do que realmente e.

### Por que essas escolhas foram feitas

- Duplicidade de linha foi verificada porque pode inflar a amostra artificialmente.
- Duplicidade de `customer_id` foi verificada porque um mesmo cliente repetido pode distorcer treino e avaliacao.
- Missing foi mapeado logo cedo porque impacta diretamente a estrategia de imputacao.
- Boxplots foram usados porque sao uma forma simples de localizar valores extremos relevantes para baseline tabular.

### Leitura correta dessa etapa

Essa secao descreve a **base bruta**, nao a base final de modelagem. Isso e importante para apresentacao: aqui estamos mostrando os problemas encontrados, nao o estado final do dataset.

## 6. Etapa 3: distribuicao do target e sinais de churn na base bruta

### O que foi feito

- medicao da proporcao de churn
- comparacao de churn por tipo de plano
- comparacao de churn por categoria de NPS
- comparacao de churn por inadimplencia
- exploracao de distribuicoes numericas relevantes
- correlacoes lineares simples com o target

### Por que essa etapa existe

Depois de entender a qualidade da base, o proximo passo foi verificar se existem sinais minimamente coerentes para o problema de negocio.

Essa secao serve para responder perguntas como:

- o target esta muito raro ou razoavelmente balanceado?
- existem segmentos com churn mais alto?
- as variaveis numericas parecem carregar algum sinal?

### Por que essas escolhas foram feitas

- `plan_type`, `nps_category` e `default_flag` foram escolhidas porque fazem sentido de negocio em churn de telecom.
- Foram usadas taxas de churn por grupo em vez de testes causais porque a Fase 1 e exploratoria.
- As correlacoes foram mantidas como leitura auxiliar, nao como criterio unico de selecao, porque churn dificilmente depende de relacoes lineares simples.

### Importante

Essas analises ainda foram feitas na base bruta. Isso foi intencional. O objetivo era entender o comportamento original da base antes de alterar registros.

## 7. Etapa 4: data readiness e risco de leakage

### O que foi feito

- classificacao de colunas por papel: ID, target ou feature
- marcacao de colunas com risco de leakage
- marcacao de colunas sensiveis a janela temporal
- geracao de uma recomendacao por coluna
- criacao de um checklist executivo de readiness

### Por que essa etapa existe

Nem toda coluna disponivel deve ir para um baseline. Em problemas de churn, alguns campos ficam perigosamente proximos do evento que queremos prever, ou ate representam acao posterior ao problema.

Sem essa etapa, o modelo pode aprender um atalho indevido.

### Por que essas escolhas foram feitas

- `churn_probability` foi excluida por ser um artefato do gerador sintetico. Ela nao representa uma variavel observada no negocio e pode vazar informacao do target.
- `retention_offer_made` e `retention_offer_accepted` foram excluidas porque representam acao de retencao, ou seja, variaveis muito proximas ou posteriores ao churn.
- `customer_id` foi removido por ser identificador.
- Variaveis como `usage_delta_pct`, `days_past_due`, `support_calls_30d` e similares foram marcadas como sensiveis a janela temporal porque precisam de governanca clara em producao.

### Beneficio para apresentacao

Essa etapa mostra maturidade metodologica. Em vez de usar tudo que existe, o notebook justifica o que entra e o que sai da modelagem.

## 8. Etapa 5: tratamento da base para baseline

### O que foi feito

- copia da base original para uma base de trabalho
- auditoria de duplicidade por `customer_id`
- remocao de linhas duplicadas exatas
- remocao de duplicidade por ID, mantendo a primeira ocorrencia
- conversao de valores fora de faixa para `NaN`
- limpeza de categorias invalidas
- ajuste de inconsistencias logicas simples
- geracao de tabelas-resumo com contagem de correcoes

### Por que essa etapa existe

O objetivo aqui nao era construir o melhor pipeline possivel, mas sim um tratamento minimo, reproduzivel e justificavel para permitir o treino de baselines.

### Por que essas escolhas foram feitas

#### 8.1 Auditoria de duplicados por ID

Antes de remover duplicados, o notebook mede:

- quantos `customer_id` repetidos existem
- quantas linhas participam dessas duplicidades
- quantos IDs possuem conflito no target
- qual a maior quantidade de registros por cliente

Isso foi incluido porque apenas apagar duplicados sem auditoria deixa a metodologia fragil para apresentacao.

#### 8.2 Remocao de duplicidade exata

Linhas totalmente iguais nao trazem informacao nova. Elas apenas repetem observacoes e podem enviesar estatisticas e o treino do modelo.

#### 8.3 Manter a primeira ocorrencia por `customer_id`

Essa foi uma decisao pragmatica para Fase 1. O notebook deixa essa regra explicita. Em um pipeline mais maduro, seria melhor definir regra temporal ou de consolidacao por cliente.

#### 8.4 Valores fora de faixa convertidos para `NaN`

Para colunas como idade, NPS, CSAT, charges e indicadores de taxa, valores impossiveis nao foram "corrigidos no chute". Eles foram convertidos para `NaN`.

Essa escolha foi feita porque:

- evita inventar valor arbitrario
- preserva a informacao de que o registro esta inconsistente
- permite imputacao controlada depois

#### 8.5 Categorias invalidas convertidas para `NaN`

Quando uma categoria nao fazia parte do dominio esperado, ela foi tratada como ausente. Isso foi preferido a criar classe nova porque, na Fase 1, o interesse principal e estabilizar a base para baseline.

#### 8.6 Ajustes logicos simples

Algumas inconsistencias foram corrigidas de forma deterministica, por exemplo:

- cliente sem fidelidade nao deveria ter meses restantes de fidelidade
- cliente com atraso nao deveria manter `default_flag = 0`
- oferta aceita nao deveria coexistir com `retention_offer_made = 0`

Essa escolha foi feita porque sao contradicoes simples e diretamente verificaveis.

## 9. Etapa 6: split, imputacao e matriz final sem leakage

### O que foi feito

- remocao de colunas que nao devem ir para a modelagem
- separacao entre features e target
- split estratificado em treino e teste
- imputacao de numericas com mediana do treino
- imputacao de categoricas com moda do treino
- one-hot encoding em treino e teste
- alinhamento das colunas do teste com as colunas do treino

### Por que essa etapa existe

Essa e a etapa que transforma a base tratada em uma matriz pronta para modelos baseline.

### Por que essas escolhas foram feitas

#### 9.1 Split antes de imputacao e encoding

Essa foi a principal correcao metodologica do notebook.

O split acontece antes das transformacoes para evitar leakage. Se a mediana, a moda ou as categorias fossem aprendidas usando a base inteira, o treino estaria recebendo informacao do teste.

#### 9.2 Split estratificado 80/20

Essa escolha foi feita porque:

- 80/20 e proporcao simples e aceita para baseline
- a estratificacao preserva a taxa de churn entre treino e teste
- facilita comparacao consistente entre modelos na fase seguinte

#### 9.3 Imputacao por mediana em numericas

A mediana foi escolhida porque e robusta a outliers e adequada para uma etapa inicial em base tabular.

#### 9.4 Imputacao por moda em categoricas

A moda foi escolhida por simplicidade e interpretabilidade. Em baseline, essa solucao costuma ser suficiente para tirar a base do estado invalido sem acrescentar complexidade desnecessaria.

#### 9.5 One-hot encoding

Foi escolhido porque:

- e padrao para modelos tabulares simples
- torna categorias explicitamente modelaveis
- evita impor ordem artificial entre categorias

#### 9.6 Alinhamento entre treino e teste

Depois do `get_dummies`, treino e teste podem gerar conjuntos de colunas diferentes. Por isso, o teste foi reindexado com base nas colunas do treino.

Essa etapa e necessaria para garantir que ambos tenham a mesma estrutura de entrada para os modelos.

## 10. Etapa 7: validacao final da base preparada

### O que foi feito

- verificacao de IDs duplicados apos tratamento
- verificacao de linhas duplicadas apos tratamento
- verificacao de missing no treino e no teste
- verificacao de ausencia de colunas de leakage na matriz final
- verificacao de colunas constantes no treino
- comparacao da taxa de churn entre base bruta, base tratada, treino e teste

### Por que essa etapa existe

Sem uma validacao final, o notebook terminaria no ponto exato em que muita gente assume que "deu certo" sem provar.

Essa secao transforma a preparacao em algo auditavel.

### Por que essas escolhas foram feitas

- `0` missing em treino e teste confirma que a imputacao foi suficiente.
- `0` leakage na matriz final confirma que as colunas excluidas realmente ficaram fora do treino.
- `0` duplicidade apos tratamento confirma que a base final nao herdou o problema estrutural da base bruta.
- a taxa de churn quase identica entre treino e teste confirma que a estratificacao preservou o balanceamento do target.

## 11. Etapa 8: conclusao executiva

### O que a conclusao resume

A conclusao do notebook sintetiza tres mensagens:

1. a base tem sinais de negocio coerentes para churn
2. a base bruta possui problemas reais de qualidade
3. depois do tratamento e da validacao, a base fica pronta para treino de baseline

### Por que essa etapa e importante

Em uma apresentacao, o publico normalmente nao quer revisar cada linha de codigo. A conclusao executiva serve para responder de forma objetiva:

- o dataset serve para o problema?
- havia problemas relevantes?
- eles foram tratados de forma consistente?
- a proxima etapa pode comecar com seguranca?

## 12. Resumo das principais escolhas metodologicas

### Escolha 1: analisar a base bruta antes de tratar

Motivo: mostrar transparência sobre os problemas da base e evitar mascarar erro antes do diagnostico.

### Escolha 2: separar readiness de tratamento

Motivo: deixar claro o que era problema da base original e o que passou a ser estado da base preparada.

### Escolha 3: auditar duplicados antes de remover

Motivo: fortalecer a justificativa em apresentacao e mostrar o impacto do problema.

### Escolha 4: transformar valores invalidos em `NaN`

Motivo: tratar inconsistencias sem inventar substituicoes arbitrarias.

### Escolha 5: split antes de imputacao e encoding

Motivo: evitar leakage metodologico.

### Escolha 6: usar mediana e moda

Motivo: simplicidade, robustez e boa adequacao para baseline de Fase 1.

### Escolha 7: validar a base final explicitamente

Motivo: garantir que a preparacao terminou em estado realmente apto para modelagem.

## 13. Limitacoes assumidas nesta fase

Algumas decisoes foram intencionalmente simples porque o foco era baseline:

- a regra de manter a primeira ocorrencia por `customer_id` e pragmatica, nao definitiva
- nao houve engenharia de features mais sofisticada
- nao foi implementado pipeline reutilizavel ainda
- a governanca temporal foi apontada, mas nao operacionalizada no notebook

Esses pontos nao invalidam a EDA. Eles apenas delimitam o escopo da Fase 1.

## 14. Proxima etapa recomendada

Com a base preparada e validada, o passo seguinte e usar exatamente o mesmo criterio de preparo no notebook de baselines e, depois, transformar esse fluxo em pipeline reutilizavel.

Em termos praticos, a sequencia recomendada e:

1. treinar baselines em `notebooks/02_baselines.ipynb`
2. comparar modelos por AUC-ROC, PR-AUC e top-K
3. consolidar o preparo em pipeline reprodutivel
4. documentar o modelo final no model card