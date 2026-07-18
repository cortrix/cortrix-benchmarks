# Result Interpretation

A benchmark result must be interpreted together with its run manifest, profile manifest, resource snapshots, and server state.

## Minimum Evidence

- Dataset name and source archive.
- Sample seed and sample manifest.
- Cortrix server version or commit.
- Profile name and actual request body.
- Model and fallback state.
- Query count and qrels coverage.
- Recall@10 and NDCG@10.
- Latency summary.
- Resource snapshots.

## Scientific validity and strict feature completeness

Scientific validity and strict feature completeness are independent result
dimensions. A run can complete every scored query with zero query failures
while a required enrichment or post-processing feature still has pending,
permanent, orphaned, or otherwise mismatched evidence.

When strict feature proof is required, generate `terminal-status.json` with
`runner/terminal_status.py` and retain all of the following:

- `scientific_result.status`: `VALID`, `INVALID`, or `NOT_EVALUATED`.
- Imported-document and evaluated-query totals used by the scientific gate.
- `strict_feature_completeness.status`: `PASS`, `FAIL`, or `NOT_EVALUATED`.
- `lane_classification`, including
  `SCIENTIFIC_COMPLETE_WITH_FEATURE_DEBT` when scores are valid but strict
  proof is not clean.
- Pending retry, failed permanent, orphan source, and other mismatch counts.
- The original wrapper return code, without promoting it after classification.

The classifier accepts existing `summary.json` artifacts without changing
their schema. Missing query-failure evidence or missing strict feature proof is
reported as `NOT_EVALUATED`; it is never inferred as passing. The terminal
artifact is written atomically before `--propagate-wrapper-rc` returns the
preserved wrapper code.

If the server uses stub embeddings, fallback rerankers, disabled enrichment, or an unconfirmed profile, the result cannot be treated as a published measured result.
