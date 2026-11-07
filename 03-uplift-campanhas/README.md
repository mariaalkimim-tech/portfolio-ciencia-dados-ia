# 03 — Incrementalidade de campanhas e uplift causal

## Resumo executivo

**Pergunta de negócio:** quais clientes geram resultado incremental quando recebem uma ação, em comparação ao que teria ocorrido sem a ação? O alvo é o efeito de tratamento, não a propensão de compra.

**Dataset inicial:** Criteo Uplift Prediction Dataset, criado a partir de testes de incrementalidade randomizados. A fonte primária reporta 25.309.483 linhas e 12 atributos anonimizados (`f0`–`f11`), com tratamento definido pela participação da empresa em leilão RTB e exposição efetiva como variável distinta. A documentação informa taxas agregadas de tratamento 0,846, visita 0,04132 e conversão 0,00229 no conjunto publicado. A subamostragem não uniforme impede recuperar diretamente a incrementalidade original da população.

## Estimando e desenho

Definir outcome (visita ou conversão), horizonte, unidade de randomização, tratamento (`treatment`) e população-alvo antes de modelar. O estimando principal recomendado é ITT (intenção de tratar): efeito da designação ao tratamento. Não substituir por `exposure` como se fosse designação randomizada; isso pode introduzir seleção pós-tratamento.

Começar por diferença de médias e intervalos de confiança no desenho experimental. Em seguida, comparar modelos de CATE/uplift com randomization-aware validation. Fazer avaliação Qini/AUUC conforme definição adotada, curvas de ganho incremental por população priorizada e políticas top-k. Documentar estimando e convenção matemática usada, pois implementações de Qini variam.

## Avaliação de decisão

Separar treino/validação/teste sem quebrar a estrutura experimental. Comparar política de targeting com tratamento aleatório e política de tratar todos, estimando incremento esperado com intervalos. Reportar conversões incrementais e custo por conversão incremental somente como avaliação do benchmark; não projetar para negócio real. Fazer análise de sensibilidade a custo de contato e valor de outcome fornecidos por negócio.

## Critério de sucesso

O modelo deve superar baselines em dados não usados para seleção, com incerteza reportada e política acionável. O critério final é um teste prospectivo randomizado em população autorizada, com métricas primárias/pré-registro, guardrails e análise de heterogeneidade. O dataset público não é evidência suficiente de ROI numa empresa específica.

## Entregáveis para completar

Dicionário de tratamento/outcomes; notebook de estimando e teste de médias; comparação de modelos de uplift; curvas de ganho e incerteza; simulador de política; plano de experimento prospectivo e recomendação executiva.

## Riscos e limitações

Features anonimizadas/projetadas limitam interpretação e transporte para outros mercados. A subamostragem altera representatividade. Outcomes raros tornam estimativas de subgrupos incertas. Exposição efetiva é pós-tratamento e pode enviesar a comparação se usada incorretamente.

## Fontes

- [Criteo AI Lab — dataset e documentação](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- [TensorFlow Datasets — Criteo](https://www.tensorflow.org/datasets/catalog/criteo)
- [Uplift Modeling — fetch Criteo](https://www.uplift-modeling.com/en/v0.3.1/api/datasets/fetch_criteo.html)
