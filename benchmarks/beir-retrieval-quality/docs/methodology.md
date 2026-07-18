# BEIR Retrieval Quality Methodology

The BEIR retrieval quality benchmark evaluates retrieval quality using BEIR corpora, queries, and relevance judgments (qrels).

## Measurement modes

- **Full corpus:** evaluate every official test query with qrels against the complete corpus for that dataset.
- **Capped validation:** use a fixed seed plus `--max-queries` and `--max-corpus-docs` to create an auditable local sample.

Capped validation checks the runner and product pipeline. It is not a smaller published benchmark and its scores are not comparable to a full-corpus result.

## Core steps

1. Download or reuse a pinned BEIR dataset archive.
2. Load corpus, queries, and qrels.
3. Select full-corpus or deterministic capped scope and write its manifest.
4. Import the selected corpus documents into Cortrix.
5. Wait for backend processing to reach a query-ready state.
6. Query Cortrix with the selected official qrels queries.
7. Map returned results back to original BEIR corpus document IDs.
8. Compute Recall and NDCG at the declared cutoffs.
9. Record latency and resource snapshots.
10. Write run and result manifests.

## Result labels

- `local diagnostic`: useful for debugging the runner, environment, or product readiness.
- `published measured result`: curated result approved for external publication.

Measurement completeness is not publication identity. A complete local run remains a local diagnostic until it is curated into a versioned bundle with checksums, source locks, limitations, and an immutable citation.
