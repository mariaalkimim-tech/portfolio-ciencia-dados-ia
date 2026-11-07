# Portfólio de Ciência de Dados, Analytics e IA

Este repositório reúne cinco projetos orientados a problemas empresariais de dados e inteligência artificial. O projeto de detecção de fraude inclui pipeline Python executado, métricas e gráficos sobre um benchmark público. Os outros quatro são propostas técnicas; ainda não têm implementação ou resultados empíricos.

## Projetos

1. **[Previsão de demanda e estoque](01-previsao-estoque/README.md)** — proposta de previsão temporal e avaliação de políticas de reposição.
2. **[Detecção de fraude em pagamentos](02-fraude-pagamentos/README.md)** — pipeline executável, resultados no conjunto OpenML 42175 e análise de capacidade de revisão.
3. **[Incrementalidade e uplift](03-uplift-campanhas/README.md)** — proposta de avaliação causal para campanhas.
4. **[Assistente corporativo com RAG](04-rag-corporativo/README.md)** — proposta de busca e resposta com fontes, avaliação e controles de segurança.
5. **[Governança e monitoramento de modelos](05-governanca-modelos/README.md)** — proposta de documentação, avaliação de fairness e monitoramento do ciclo de vida.

## Resultado do projeto de fraude

O pipeline compara regressão logística balanceada e HistGradientBoosting, com divisão cronológica em treino, validação e teste. A seleção do modelo é feita na validação; o teste é reservado para avaliação final. Na partição de teste usada, a regressão logística obteve Average Precision de 0,762. Entre os 0,1% de registros de maior score, o ranking incluiu 52 das 75 fraudes da partição (precisão de 91,2% e recall de 69,3%). Os detalhes, incluindo outras capacidades e ressalvas, estão no README e em `metrics.json` do projeto.

Esses números descrevem um split retrospectivo de um benchmark anonimizado que cobre apenas dois dias. Não são resultados de produção, garantia de generalização ou estimativa de economia. O conjunto original não é redistribuído neste repositório; as fontes e instruções para obtê-lo estão documentadas nos projetos.

## Como reproduzir o projeto de fraude

Consulte o [README do projeto de fraude](02-fraude-pagamentos/README.md) para instalar dependências e executar `fraud_pipeline.py`. A primeira execução precisa acessar o OpenML, ou pode receber um CSV local compatível. Os gráficos e as métricas publicados foram produzidos pelo pipeline. O arquivo de modelo serializado é excluído do controle de versão; pode ser recriado localmente ao executar o script.

## Integridade dos dados

Cada projeto lista suas fontes e limitações. As bases de fraude e uplift não devem ser tratadas como representativas de uma empresa específica. A base de varejo UCI é histórica. O benchmark Adult é antigo e deriva dos EUA. Um conjunto de dados de benchmark ajuda a testar métodos, mas não substitui validação com dados autorizados e contexto operacional real.
