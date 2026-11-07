# 04 — Assistente corporativo com RAG, evidências e segurança

## Resumo executivo

**Pergunta de negócio:** como responder perguntas a partir de documentos autorizados, citando evidências recuperadas e recusando respostas não suportadas?

**Benchmark público:** FiQA-2018 no BEIR pode servir para iniciar comparação de recuperação no domínio financeiro. BEIR foi proposto como benchmark heterogêneo de recuperação zero-shot; não é coleção de documentos internos de uma empresa. Use documentos autorizados/sanitizados e consultas avaliadas por especialistas para validação empresarial posterior.

## Arquitetura proposta

Ingestão com controle de acesso e versionamento → extração/OCR validada → segmentação com metadados e ACL → índice BM25 baseline e índice vetorial → busca híbrida → reranking → geração condicionada apenas ao contexto permitido → citações com identificadores de documento/trecho → validação da resposta → logs sem dados sensíveis. Separar identidade do usuário e autorização do mecanismo de recuperação; filtros de ACL devem ser aplicados antes de expor trechos ao modelo.

## Avaliação em camadas

**Retrieval:** NDCG@10, Recall@k, MRR e precisão do contexto com relevâncias julgadas; comparar BM25, vetorial, híbrido e reranker no mesmo conjunto de consultas. **Resposta:** factualidade/groundedness, relevância, completude e qualidade das citações. **Operação:** latência p50/p95, custo por pergunta, taxa de abstinência e taxa de falha. Avaliações automáticas por LLM devem ser calibradas contra revisão humana e versionadas junto com prompt/modelo.

Testar consultas sem resposta, contradições e documentos desatualizados. Incluir testes adversariais para prompt injection em conteúdo recuperado, tentativas de recuperar documentos sem permissão, exfiltração, instruções conflitantes, PII e conteúdo malicioso. RAG não elimina prompt injection. Não conectar o protótipo a ferramentas de escrita/ação; qualquer ação de alto risco exigiria aprovação humana e controles separados.

## Critério de sucesso

Definir um conjunto de perguntas aceito por especialistas, critérios de suporte/citação e limites por risco. Só promover uma versão se superar baseline em retrieval e qualidade de resposta sem violar controle de acesso, segurança ou latência acordada. Publicar resultados com intervalos e erros categorizados, não uma pontuação única.

## Entregáveis para completar

Pipeline de ingestão e indexação; benchmark versionado; avaliação retrieval/resposta; painel de erros e latência; threat model; guia de operação; demonstração com fontes visíveis e opção de abstenção. Armazenar prompts/configuração e hash das fontes para reprodutibilidade.

## Riscos e limitações

FiQA é benchmark público financeiro, não corpus corporativo. Métricas de LLM-as-judge dependem do modelo e prompt avaliador. Citações podem parecer válidas sem sustentar a alegação, portanto requerem verificação. ACL, retenção, residência dos dados e privacidade dependem da política empresarial; não presumir autorização de documentos públicos para treino.

## Fontes

- [BEIR — paper](https://arxiv.org/abs/2104.08663)
- [BEIR — repositório](https://github.com/beir-cellar/beir)
- [Datasets BEIR e FiQA](https://github.com/beir-cellar/beir/wiki/Datasets-available)
- [Microsoft — avaliadores RAG](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/rag-evaluators)
- [Ragas — métricas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [OWASP — prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)
