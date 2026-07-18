# Three-dataset full-corpus retrieval measurements

This bundle records six full-corpus BEIR retrieval cells: SciFact, FiQA, and NFCorpus, each measured with `Full Stack` and `Embedding + Reranking`. It is limited to retrieval quality. It does not establish end-to-end answer quality, broad platform support, or business outcomes.

## Profile identity

The display names do not replace the immutable source identities:

| Source identity | Public display label | Runner profile |
| --- | --- | --- |
| `full_llm` | Full Stack | `bge_m3_rf_pool_listwise` |
| `no_llm` | Embedding + Reranking | `bge_m3_rerank_baseline_v2` |

Embedding + Reranking is not embedding-only or vector-only. It keeps BGE-M3 embedding and `bge-reranker-v2-m3` while disabling the LLM-dependent ingestion and query stages defined by its profile contract.

## Full-corpus results

All official test-qrels queries completed with zero query failures. Scores below are raw means. Each latency pair is comparable only within the same dataset and hardware setup.

`qrels_rows` means positive test-qrels rows. SciFact has 339 rows across 300 query IDs and 283 unique relevant document IDs; FiQA has 1,706 rows across 648 query IDs and 1,706 unique relevant document IDs; NFCorpus has 12,334 rows across 323 query IDs and 3,128 unique relevant document IDs.

| Dataset | Profile | Docs | Queries | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | p50 ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | Full Stack | 5,183 | 300 | 0.8698 | 0.9303 | 0.7716 | 0.7860 | 34,232 | 39,545 |
| SciFact | Embedding + Reranking | 5,183 | 300 | 0.7794 | 0.9193 | 0.6009 | 0.6345 | 7,311 | 8,766 |
| FiQA | Full Stack | 57,638 | 648 | 0.5205 | 0.6640 | 0.4663 | 0.5062 | 106,172 | 135,121 |
| FiQA | Embedding + Reranking | 57,638 | 648 | 0.3635 | 0.6343 | 0.2526 | 0.3324 | 26,521 | 33,303 |
| NFCorpus | Full Stack | 3,633 | 323 | 0.1717 | 0.2545 | 0.3721 | 0.3175 | 29,504 | 32,229 |
| NFCorpus | Embedding + Reranking | 3,633 | 323 | 0.1482 | 0.2361 | 0.2982 | 0.2659 | 5,905 | 6,964 |

The unweighted three-dataset macro means are:

| Profile | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 |
| --- | ---: | ---: | ---: | ---: |
| Full Stack | 0.5207 | 0.6163 | 0.5366 | 0.5366 |
| Embedding + Reranking | 0.4304 | 0.5966 | 0.3839 | 0.4109 |

The two bounded FiQA comparisons derived from the raw cells are:

- **Up to 84.6% higher NDCG@10.** Full Stack vs Embedding + Reranking on FiQA.
- **Up to 43.2% higher Recall@10.** Full Stack vs Embedding + Reranking on FiQA.

Exact source values, formulas, and unrounded macro means are in [`summaries.json`](summaries.json).

## Scientific and strict status

Scientific validity and strict feature completeness are separate dimensions:

| Dataset | Profile | Scientific | Strict feature completeness |
| --- | --- | --- | --- |
| SciFact | Full Stack | VALID | FAIL |
| SciFact | Embedding + Reranking | VALID | PASS |
| FiQA | Full Stack | VALID | NOT_EVALUATED |
| FiQA | Embedding + Reranking | VALID | NOT_EVALUATED |
| NFCorpus | Full Stack | VALID | FAIL |
| NFCorpus | Embedding + Reranking | VALID | PASS |

The NFCorpus and SciFact Full Stack runs had partial coverage in LLM-dependent feature verification; the scores shown are the complete measured retrieval results. The retained FiQA result predates the terminal-status v1.0 contract, so this bundle does not infer a formal strict result for FiQA.

## Hardware boundary

- SciFact and NFCorpus were paired on AWS `g5.2xlarge` with NVIDIA A10G CUDA execution for embedding and reranking.
- FiQA was paired on AWS `c7i.8xlarge` with CPU execution for embedding and reranking.
- Latency is not compared across datasets or hardware classes.

## Reproduction order

1. Read [`manifest.json`](manifest.json) for source locks, archive identities, and protocol.
2. Inspect the six files under [`scorecards/`](scorecards/) for exact metrics, latency, status, and provenance.
3. Inspect [`profiles/`](profiles/) for the source-to-display mapping and query/server contracts.
4. Follow [`reproduction.md`](reproduction.md) for the pinned full-corpus procedure.
5. Run the repo-owned result-bundle validator and verify [`checksums.sha256`](checksums.sha256).

When citing these measurements, pin this directory to a release tag or immutable commit rather than a mutable branch.
