# 02 — Detecção de fraude em pagamentos com avaliação operacional

## Resumo executivo

**Pergunta de negócio:** com uma capacidade limitada de revisão ou bloqueio, quais transações devem ser investigadas para reduzir perdas sem criar fricção excessiva para clientes legítimos?

**Dataset inicial:** Credit Card Fraud Detection da ULB/Worldline, distribuído via Kaggle/OpenML. A descrição reporta 284.807 transações de dois dias de setembro de 2013, com 492 fraudes (aprox. 0,172%). `V1`–`V28` são componentes transformados por PCA; `Time` é relativo ao início da amostra; `Amount` é valor; `Class` é rótulo. A base é fortemente anonimizada e não adequada para inferir generalização comercial.

## Executar

Requer Python 3.10 ou superior. Na pasta deste projeto:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python fraud_pipeline.py
```

Sem arquivo local, o script busca o dataset OpenML ID `42175`, exigindo internet na primeira execução. Alternativamente, baixe a base por um canal autorizado e passe o caminho ao CSV:

```bash
python fraud_pipeline.py --csv data/creditcard.csv --output-dir artifacts
```

A base não é incorporada ao repositório. O programa verifica o alvo binário e as colunas temporais, ordena os registros por `Time` e faz uma separação cronológica 60% treino, 20% validação e 20% teste. Compara regressão logística com ponderação de classes e HistGradientBoosting. A validação seleciona o candidato; o teste fica reservado para avaliação final. Os arquivos gerados incluem `metrics.json`, `precision_recall_test.png`, `capture_vs_review_capacity.png` e `fraud_model.joblib`. O limiar 0,5 é somente descritivo. Não há estimativa de economia ou ROI sem custos reais fornecidos pela operação.

## Método

1. Validar contagens, duplicatas, definição do rótulo e proveniência. Preservar ordem temporal e separar treino, validação e teste por tempo; documentar qualquer validação walk-forward. Não usar divisão aleatória como avaliação principal.
2. Estabelecer baseline de regra, regressão logística e modelo não linear. Tratar o desbalanceamento sem contaminar validação/teste; qualquer reamostragem ocorre somente no treino.
3. Estimar precisão-revocação e PR-AUC, recall em limites de alertas por mil transações, precisão no topo (`Precision@K`), perdas capturadas e calibração. ROC-AUC pode ser secundária, nunca substituta da análise de operação.
4. Simular política de decisão com custo de falso positivo, falso negativo, revisão e recuperação de valor. Quando os custos não existirem, publicar curvas de trade-off e cenários explicitamente hipotéticos, sem apresentar lucro estimado como fato.
5. Usar explicações locais e globais como ferramenta de diagnóstico, não como prova causal. Analisar estabilidade por períodos e segmentos permitidos/eticamente justificados.

## Avaliação e critérios de implantação

Escolher limiar pela capacidade de revisão e custo esperado definido com o negócio. Comparar baseline e candidato em teste temporal intocado. Definir metas com stakeholders para limite de falsos positivos, recall mínimo, latência e taxa de revisão. Monitorar drift de score/entrada, atraso e mudança na taxa de confirmação. A revisão humana e o processo de contestação são parte do desenho operacional.

## Entregáveis para completar

Pipeline de treinamento reprodutível; relatório com matriz de confusão em vários limiares, PR curve e captura de fraude em top-K; simulador de fila de revisão; documentação de decisão; plano de monitoramento. Testes contra leakage temporal e resampling fora da partição de treino.

## Riscos e limitações

Dois dias não medem ciclos de longo prazo. Variáveis PCA reduzem interpretabilidade operacional e não preservam todas as relações necessárias a uma implantação. Definições de fraude variam entre operações e fontes. A taxa da amostra não deve ser transportada para outro negócio ou interpretada como prevalência universal.

## Fontes

- [Dataset ULB/Worldline no Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [OpenML — dataset 42175](https://www.openml.org/d/42175)
- [Fraud Detection Handbook — base real](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/Baseline_RealWorldData.html)
- [Scikit-learn — validação cruzada](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Banco Central Europeu — estatísticas de fraude](https://www.ecb.europa.eu/press/cardfraud/html/ecb.cardfraudreport202110~cac4c418e8.en.html)
