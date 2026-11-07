# Como avaliar um modelo de fraude quando fraudes são raras?

**Neste projeto de detecção de fraude em pagamentos, a principal conclusão não é simplesmente “qual modelo venceu”. A métrica e a capacidade de investigação mudam completamente a leitura do resultado.**

Usei o benchmark público Credit Card Fraud Detection, associado à ULB/Worldline e disponível via OpenML/Kaggle. A base tem 284.807 transações, com 492 fraudes no conjunto completo. Os atributos `V1`–`V28` são componentes transformados por PCA e anonimizados. O dataset é adequado para estudar avaliação de classe rara, mas não representa uma operação bancária real nem oferece contexto completo das transações.

O pipeline em Python valida o esquema, ordena os registros por `Time`, separa treino, validação e teste cronologicamente (60%/20%/20%) e compara uma regressão logística balanceada com HistGradientBoosting. A seleção foi feita pela **Average Precision** na validação; só depois o modelo escolhido foi ajustado em treino + validação e medido uma vez no teste reservado.

No teste temporal — 56.962 transações e 75 fraudes — a regressão logística balanceada obteve **Average Precision de 0,762**. A prevalência de fraude nessa partição foi **0,132%**, então reportar acurácia isoladamente esconderia o que importa. ROC-AUC ficou em 0,986, mas trato esse número como secundário diante da classe rara.

A análise de capacidade tornou a avaliação mais operacional: se a equipe pudesse revisar os **0,1% de transações com maior score** no teste, seriam 57 alertas. Nesse recorte, o ranking incluiu 52 das 75 fraudes: **precisão de 91,2% e recall de 69,3%**. Já no limiar padrão de 0,5, o modelo marcou 947 alertas, dos quais 67 eram fraudes — **precisão de 7,1% e recall de 89,3%**. Isso mostra por que 0,5 não deve ser adotado automaticamente como regra operacional: aumenta a captura, mas também a quantidade de falsos positivos.

Esses números são resultados retrospectivos neste split específico, não uma promessa de desempenho futuro. A base cobre apenas dois dias, é anonimizada e não informa custos de revisão, perdas, recuperações ou impacto sobre clientes. Portanto, não estimei economia nem ROI. Para avançar em direção a uma decisão operacional, seria necessário validar em períodos mais longos, estimar incerteza, definir custos com a área de risco e testar prospectivamente com supervisão humana.

Código, métricas e gráfico da curva Precisão–Recall: [repositório/arquivos do projeto](LINK_DO_REPOSITORIO). O pipeline salva também métricas detalhadas e a curva de fraude capturada por capacidade de revisão.

**Para quem trabalha com risco, dados ou prevenção a fraude:** qual restrição pesa mais na prática ao priorizar alertas — capacidade da equipe, custo dos falsos positivos ou latência para bloquear uma transação?

#DataScience #FraudDetection #MachineLearning #RiskAnalytics #Python

---

**Antes de publicar:** substitua `LINK_DO_REPOSITORIO` pelo link do GitHub ou portfólio onde o código e os artefatos estejam acessíveis. As métricas acima foram geradas pelo script neste projeto; a avaliação é de benchmark, não de produção. Fonte e limitações: [OpenML 42175](https://www.openml.org/d/42175) e [dataset ULB/Worldline no Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
