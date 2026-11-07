# Portfólio de Ciência de Dados, Analytics e IA

Este repositório reúne projetos voltados a problemas empresariais de dados e inteligência artificial. Os projetos 2, 6 e 7 incluem código executável e resultados em benchmarks públicos. Os projetos 1, 3, 4 e 5 são propostas técnicas e ainda não têm implementação experimental.

## Projetos

1. **[Previsão de demanda e estoque](01-previsao-estoque/README.md)** — proposta de previsão temporal e avaliação de política de reposição.
2. **[Detecção de fraude em pagamentos](02-fraude-pagamentos/README.md)** — pipeline cronológico de detecção e priorização da fila de revisão.
3. **[Incrementalidade e uplift](03-uplift-campanhas/README.md)** — proposta de avaliação causal para campanhas.
4. **[Assistente corporativo com RAG](04-rag-corporativo/README.md)** — proposta de busca e resposta apoiadas por fontes, com avaliação e controles de segurança.
5. **[Governança e monitoramento de modelos](05-governanca-modelos/README.md)** — proposta de documentação, fairness e monitoramento do ciclo de vida.
6. **[Manutenção preditiva industrial](06-manutencao-preditiva/README.md)** — classificação de falhas com split cronológico e métricas de capacidade de inspeção. Usa o AI4I, que é explicitamente sintético.
7. **[Qualidade de placas de aço](07-qualidade-placas-aco/README.md)** — classificação multiclasse de sete defeitos, com validação cruzada estratificada e métricas macro.

## Resultados dos dois projetos B2B adicionados

No AI4I, o Random Forest balanceado obteve Average Precision de **0,685** no teste cronológico de 2.000 observações. Ao encaminhar os 1% com maior score para inspeção, o ranking incluiu 18 das 39 falhas do teste (precisão de 90%; recall de 46,2%). O conjunto e os mecanismos de falha são sintéticos; não se infere redução real de downtime nem ROI.

No Steel Plates Faults, o Random Forest obteve **F1 macro de 0,826**, **acurácia balanceada de 0,792** e **Average Precision macro one-vs-rest de 0,866** no split estratificado de teste com 486 linhas. O teste tem somente 14 exemplos da classe Dirtiness, então métricas de classes raras são incertas e não equivalem a validação em uma usina.

## Reproduzir

Cada projeto executável tem seu próprio `requirements.txt`, instruções e script de linha de comando. Os scripts buscam os dados nos repositórios UCI via `ucimlrepo`; é necessária conexão à internet na primeira execução. Métricas e gráficos são versionados. Dados brutos, modelos serializados, ambientes e cache local são excluídos do Git.

## Integridade dos dados

Os resultados são de partições de benchmark e estão acompanhados por limitações. Eles não são evidência de produção, garantia de generalização ou prova de retorno financeiro. Os dados originais não são redistribuídos aqui; consulte as fontes e licenças nos READMEs dos projetos. Para implantação B2B, são necessários dados autorizados da operação, validação prospectiva, custos e revisão de especialistas.
