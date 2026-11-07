# 05 — Governança, explicabilidade e monitoramento do ciclo de vida

## Resumo executivo

**Pergunta de negócio:** quais evidências, controles e alarmes são necessários para avaliar, aprovar e acompanhar um modelo de decisão desde a concepção até sua retirada?

**Benchmark didático:** UCI Adult/Census Income prevê se a renda anual ultrapassa US$ 50 mil. O UCI reporta 48.842 instâncias, 14 atributos e valores ausentes. É um benchmark histórico dos EUA com dados de 1994, não uma população empresarial atual. Fairlearn documenta Adult como benchmark de comparação de fairness e também aponta ACSIncome como alternativa mais recente.

## Aplicação do NIST AI RMF

Organizar o trabalho pelas quatro funções do NIST AI RMF 1.0: **Govern** (papéis, política, supervisão e documentação), **Map** (contexto, usuários afetados e riscos), **Measure** (métricas, testes e evidências) e **Manage** (priorização, mitigação, monitoramento e resposta). O framework é voluntário; não substitui obrigações legais nem avaliação específica do setor.

Criar model card/model inventory com finalidade, proprietário, população, dados, versão, uso proibido, validação, limiares, grupos avaliados, limitações, explicações, dependências e plano de rollback. Diferenciar modelo experimental, shadow, assistido e automatizado.

## Experimentos no benchmark

Treinar baseline interpretável e candidato. Comparar desempenho e fairness sob definições explicitadas: demographic parity difference, equalized odds difference ou métricas pertinentes ao caso. Informar unidade, grupo de referência, limiar e denominador. Não existe métrica universalmente suficiente; trade-offs e requisitos legais/negociais precisam de revisão apropriada. Explicabilidade global/local serve a diagnóstico, não prova de justiça.

Simular monitoramento de qualidade de dados, drift de características e scores, calibração e desempenho quando rótulos ficam disponíveis. Como Adult não é stream temporal, construir avaliação de drift somente com novos dados autorizados ou cenários controlados claramente identificados como teste, nunca fingir que são observações reais. Definir alertas, owner, severidade, investigação, rollback e comunicação.

## Critério de sucesso

Um modelo não é aprovado por uma pontuação isolada. Exigir documentação completa, avaliação reprodutível, desempenho aceitável nos critérios do caso, investigação de disparidades, controle de acesso, monitoramento testado e responsável com poder de interromper o uso. As metas numéricas devem vir de contexto e governança, não ser inventadas no benchmark.

## Entregáveis para completar

Inventário; model card; relatório de avaliação por grupo e limiar; monitor de dados/modelo; playbook de incidentes; log de decisão; apresentação executiva sobre riscos residuais e limites. Incluir testes de regressão e revisão periódica.

## Riscos e limitações

A base histórica pode reproduzir desigualdades e relações não representativas do presente. Métricas de fairness dependem de definição e limiar. O conjunto não mede drift de implantação nem riscos de um domínio regulado real. Framework NIST é orientação voluntária, não certificação nem garantia de conformidade.

## Fontes

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [NIST AI RMF 1.0 — PDF](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf)
- [NIST AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)
- [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult)
- [Fairlearn — datasets e ACSIncome](https://fairlearn.org/main/user_guide/datasets/acs_income.html)
