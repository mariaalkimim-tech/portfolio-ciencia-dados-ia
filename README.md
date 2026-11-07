# Portfólio de Ciência de Dados, Analytics e IA

Portfólio voltado a problemas de empresas de grande porte. Há projetos executáveis com avaliação reprodutível e propostas de projetos ainda não implementados; o status de cada um está indicado abaixo. Os três projetos mais recentes miram bancos e operações corporativas de TI — sem foco em segurança cibernética.

## Projetos

1. **[Previsão de demanda e estoque](01-previsao-estoque/README.md)** — proposta de previsão temporal e avaliação de políticas de reposição; ainda sem resultados experimentais.
2. **[Detecção de fraude em pagamentos](02-fraude-pagamentos/README.md)** — pipeline executável e avaliação cronológica de priorização da fila de revisão.
3. **[Incrementalidade e uplift](03-uplift-campanhas/README.md)** — proposta de avaliação causal para campanhas; ainda sem resultados experimentais.
4. **[Assistente corporativo com RAG](04-rag-corporativo/README.md)** — proposta de busca e respostas ancoradas em fontes; ainda sem resultados experimentais.
5. **[Governança e monitoramento de modelos](05-governanca-modelos/README.md)** — proposta de documentação, auditoria e monitoramento; ainda sem resultados experimentais.
6. **[Manutenção preditiva industrial](06-manutencao-preditiva/README.md)** — pipeline executável em benchmark sintético UCI AI4I, com split cronológico e métricas de capacidade.
7. **[Qualidade de placas de aço](07-qualidade-placas-aco/README.md)** — pipeline executável de classificação de sete defeitos com validação cruzada estratificada.
8. **[Risco de inadimplência para bancos](08-risco-credito/README.md)** — comparação de modelos, calibração, ranking por capacidade e auditoria descritiva por grupo.
9. **[Priorização de campanhas bancárias](09-analytics-marketing-bancario/README.md)** — score de propensão com controles contra vazamento pós-contato; não faz alegação causal.
10. **[Analytics de operações de TI (ITSM)](10-analytics-operacoes-itsm/README.md)** — process analytics e Kaplan–Meier com censura explícita em incidentes de service desk; não é projeto de cyber.

## Resultados dos projetos bancários e ITSM

**Risco de crédito:** no teste estratificado de 7.500 registros do UCI Default of Credit Card Clients, HistGradientBoosting obteve Average Precision de **0,556**. No topo de 5% dos scores, 292 de 375 clientes eram defaults observados (77,9% de precisão; captura de 17,6% dos defaults). É um benchmark histórico de Taiwan, sem LGD ou custos para calcular perda esperada ou ROI.

**Campanhas bancárias:** no teste de 11.303 registros, HistGradientBoosting obteve Average Precision de **0,458**, com resposta observada de 11,7%. No topo de 1% (114 contatos), 92 subscreveram (taxa observada de 80,7%). Essa concentração não é uplift nem efeito causal; os dados não são um experimento e não fornecem custo de contato ou valor financeiro.

**Operações de TI:** o log BPI Challenge 2013 contém **7.554 casos e 65.533 eventos**. Foram classificados 5.663 como encerrados (`Closed`/`Resolved`) e 1.891 como censurados. A mediana Kaplan–Meier do tempo até encerramento foi **200,2 horas**. Um diagnóstico temporal encontrou zero casos encerrados acima do p75 histórico na janela final, por isso não se publica um classificador de duração longa sem classe positiva no holdout. O log tem fortes mudanças de volume e é antigo, de uma única organização; não se inferem SLA ou desempenho atual.

Os projetos 6 e 7 também têm métricas verificadas em seus READMEs. O AI4I é explicitamente sintético; o benchmark de placas é pequeno e tem classes raras.

## Reproduzir e integridade

Cada projeto executável tem `requirements.txt`, instruções e script de linha de comando. Os projetos 06–10 também incluem testes unitários de invariantes e métricas. Dados públicos são carregados da fonte original ou baixados conforme as instruções; arquivos de dados brutos, modelos serializados, ambientes e caches são excluídos do Git. Métricas e gráficos são versionados.

Resultados de benchmarks não são evidência de desempenho em produção, garantia de generalização ou comprovação de retorno financeiro. Consulte as limitações e citações nos READMEs individuais. Aplicações empresariais exigem dados autorizados, validação prospectiva, custos reais, revisão especializada e governança adequada.
