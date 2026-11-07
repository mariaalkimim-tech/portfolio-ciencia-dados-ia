# 01 — Previsão de demanda e política de reposição

## Resumo executivo

**Pergunta de negócio:** para cada produto e período, qual distribuição de demanda é plausível e qual decisão de reposição equilibra ruptura, excesso e custo de estoque? O projeto diferencia previsão (estimativa do futuro) de decisão (ação sujeita a custos e restrições).

**Dataset inicial:** UCI Online Retail, transações de varejista britânica entre 01/12/2010 e 09/12/2011, 541.909 registros e seis atributos reportados pelo repositório. É transacional, não contém níveis de estoque, lead time ou custo de falta. Licença indicada: CC BY 4.0. Use-o para demonstrar agregação, previsão de vendas e backtesting; não alegue que resolve reposição real sem enriquecer dados.

## Escopo e método

1. Auditar duplicatas, cancelamentos (InvoiceNo começando por `C`), devoluções, valores inválidos e granularidade. Definir se quantidade líquida ou pedidos brutos representam a demanda desejada. Documentar regras antes de treinar.
2. Agregar a série em SKU-dia ou SKU-semana. Identificar produtos intermitentes e mudanças de sortimento; avaliar cobertura e zeros em calendário.
3. Criar baselines sazonais ingênuos e médias móveis. Comparar modelos de suavização exponencial e modelos globais de machine learning apenas se a quantidade de séries justificar. Variáveis exógenas entram somente se conhecidas no instante da previsão.
4. Usar backtesting de origem móvel, splits cronológicos e horizonte alinhado ao ciclo operacional. Nunca embaralhar o tempo. Separar tuning do período final de teste.
5. Produzir previsões pontuais e quantis. Se houver dados de lead time e custo, otimizar ponto de reposição sob restrições e medir serviço/custo; sem esses dados, reportar somente previsão de vendas.

## Avaliação

Reportar MAE e WAPE agregados e por segmento; MASE como comparação normalizada quando definida corretamente. Incluir viés de previsão e desempenho por horizonte/volume/intermitência. Avaliar quantis por pinball loss e cobertura dos intervalos. Não usar MAPE indiscriminadamente em séries com zeros. Para reposição, exigir dados de custo, estoque e lead time e então reportar fill rate, ruptura e estoque médio em simulação retrospectiva.

## Critério de sucesso

O baseline e o candidato são comparados no mesmo backtest e no mesmo conjunto de produtos. A promoção requer melhoria material acordada com a área, ausência de piora inaceitável em segmentos críticos, intervalos calibrados e resultado estável em janelas temporais. O limiar financeiro fica pendente de custos reais; não há alegação prévia de economia.

## Entregáveis para completar

Notebook de auditoria e agregação; pipeline parametrizado; relatório de backtesting; dashboard com erro, viés e cobertura por segmento; nota executiva com decisão, limites e plano de implantação. Incluir testes para cancelamentos, datas, ausência de períodos e prevenção de vazamento.

## Riscos e limitações

A base é antiga, de uma única varejista e sem estoque ou lead time. O processo de limpeza pode mudar a definição de demanda. Mudanças de preço, promoção, sortimento e mercado não são inferíveis de forma confiável sem covariáveis. A evidência desta base não representa uma rede moderna.

## Fontes

- [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- [Forecasting: Principles and Practice — introdução](https://otexts.com/fpp3/intro.html)
- [Forecasting: Principles and Practice — suavização exponencial](https://otexts.com/fpp3/expsmooth.html)
