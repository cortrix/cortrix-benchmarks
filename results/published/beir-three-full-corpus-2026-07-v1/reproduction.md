# Exact full-corpus reproduction

These commands describe how to reproduce the six measured cells. Run them only in an environment where you control the required Cortrix services, models, and compute resources.

The bundle can be inspected and checksummed independently. A complete rerun also requires the exact Cortrix source snapshot cited alongside the bundle. Until that snapshot is available from the future public Cortrix repository, this document describes an authenticated maintainer procedure rather than claiming end-to-end public reproducibility.

This procedure reproduces one isolated cell at a time. It uses the current runner CLI without invented options. With `--top-k 50`, the pinned runner requests 250 candidates, collapses chunk or child hits to source documents, retains the first 50 unique document IDs, and records them in `queries.jsonl`. Recall and NDCG at 10 and 50 are then computed from that same cold-query log.

## 1. Pin source and verify the runner

Use clean, commit-pinned checkouts. Set `BENCHMARK_PUBLICATION_COMMIT` to the full 40-character commit in the canonical bundle URL. Set `CORE_PUBLICATION_COMMIT` to the full commit in the exact Cortrix source URL cited alongside the bundle.

The historical measurement identities remain `6defa0a50d696a7e4049413522fc0daca68f882e` for the runner and `ba725c892ece3d0ce4e315c2d0155d82031f6d77` for the terminal-status v1.0 contract. Those internal engineering commits are not required to exist in the clean repository history. Their public file-level equivalents and hashes are recorded in `provenance/source-equivalence.json`.

```bash
export CORE_DIR=/path/to/cortrix
export BENCH_DIR=/path/to/cortrix-benchmarks
export CORE_PUBLICATION_COMMIT="${CORE_PUBLICATION_COMMIT:-}"
export BENCHMARK_PUBLICATION_COMMIT="${BENCHMARK_PUBLICATION_COMMIT:-}"

: "${CORE_PUBLICATION_COMMIT:?Set this to the 40-character commit from the cited Cortrix source URL}"
: "${BENCHMARK_PUBLICATION_COMMIT:?Set this to the 40-character commit from the canonical bundle URL}"

git -C "$CORE_DIR" checkout --detach "$CORE_PUBLICATION_COMMIT"
git -C "$BENCH_DIR" checkout --detach "$BENCHMARK_PUBLICATION_COMMIT"

cd "$BENCH_DIR"
python3 benchmarks/beir-retrieval-quality/runner/run_benchmark.py selftest
python3 benchmarks/beir-retrieval-quality/runner/run_benchmark.py run --help

sha256sum benchmarks/beir-retrieval-quality/runner/profiles.py
sha256sum benchmarks/beir-retrieval-quality/runner/run_benchmark.py
```

Expected source hashes:

```text
e86f6fab488a70183b1b8540cee548fdcf3033b640d9a57e7e26c336ef79c012  profiles.py
030ad59186003b6efb0ce24594f76652d82a82ccfcfe748bf6ee75a67c8d7b46  run_benchmark.py
```

`profiles.py` is the documented clean-repository overlay. Its executable fields for the two published profiles are tested against the bundle contracts. The frozen source version has SHA-256 `2dd762cbd22a57388c25ed16fee1c44eed6edf724aef6a55583f1eb1b6203aaf`; the equivalence record preserves both identities.

The terminal-status contract is a separate historical source lock. Use the commit-pinned clean snapshot and its equivalence record when regenerating the bundle, and do not infer a formal strict result for retained FiQA cells that predate that contract.

## 2. Download and verify official BEIR archives

```bash
export CORTRIX_BENCH_WORKDIR="${CORTRIX_BENCH_WORKDIR:-/tmp/cortrix-benchmarks/beir-retrieval-quality}"
mkdir -p "$CORTRIX_BENCH_WORKDIR/datasets"

curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -o "$CORTRIX_BENCH_WORKDIR/datasets/scifact.zip"
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip -o "$CORTRIX_BENCH_WORKDIR/datasets/fiqa.zip"
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip -o "$CORTRIX_BENCH_WORKDIR/datasets/nfcorpus.zip"

printf '%s  %s\n' \
  536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165 "$CORTRIX_BENCH_WORKDIR/datasets/scifact.zip" \
  32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02 "$CORTRIX_BENCH_WORKDIR/datasets/fiqa.zip" \
  efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b "$CORTRIX_BENCH_WORKDIR/datasets/nfcorpus.zip" \
  | sha256sum -c -
```

Verify the archive contents before starting Cortrix:

```bash
python3 - "$CORTRIX_BENCH_WORKDIR/datasets" <<'PY'
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

expected = {
    "scifact": (5183, 300, 339),
    "fiqa": (57638, 648, 1706),
    "nfcorpus": (3633, 323, 12334),
}
root = Path(sys.argv[1])
for dataset, wanted in expected.items():
    with zipfile.ZipFile(root / f"{dataset}.zip") as archive:
        def member(suffix):
            matches = [name for name in archive.namelist() if name.endswith(suffix)]
            if len(matches) != 1:
                raise SystemExit(f"{dataset}: expected one {suffix}, found {matches}")
            return matches[0]

        corpus_rows = sum(1 for line in archive.open(member("corpus.jsonl")) if line.strip())
        qrels_text = io.TextIOWrapper(archive.open(member("qrels/test.tsv")), encoding="utf-8")
        rows = list(csv.DictReader(qrels_text, delimiter="\t"))
        positive_rows = [row for row in rows if float(row["score"]) > 0]
        qrel_queries = {row["query-id"] for row in positive_rows}
        observed = (corpus_rows, len(qrel_queries), len(positive_rows))
        if observed != wanted:
            raise SystemExit(f"{dataset}: observed {observed}, expected {wanted}")
        print(json.dumps({"dataset": dataset, "counts": observed}))
PY
```

## 3. Prepare one isolated Cortrix runtime per cell

Use `profiles/full-stack.json` or `profiles/embedding-reranking.json` as the sanitized contract. Do not reuse a namespace across profiles or datasets.

For FiQA, use an AWS `c7i.8xlarge` server with the embedding and reranker execution providers set to CPU. For SciFact and NFCorpus, use AWS `g5.2xlarge` with NVIDIA A10G CUDA execution for embedding and reranking. Prepare Cortrix according to the exact source and deployment documentation cited with the release. Keep model identities pinned to BGE-M3 and `bge-reranker-v2-m3`.

Full Stack additionally requires the configured semantic, document-summary, and enrichment LLM roles plus query-time RAG Fusion and listwise reranking. Supply credentials only through the supported local secret mechanism; do not place them in run artifacts. Generate a contract-smoke report before the benchmark and export its path:

```bash
export BASE_URL=http://127.0.0.1:8080
export METRICS_URL=http://127.0.0.1:9091/metrics
export CORTRIX_DATA_DIR=/path/to/the/isolated/runtime-data
export LLM_CONTRACT_SMOKE_REPORT=/path/to/validated/llm-contract-smoke.json
```

The runtime must be dedicated to one cell. No other importer, worker configuration change, or corpus mutation may run after the runner finishes task drain and enters `cold_query`. The `stage_events.jsonl` ordering is the freeze boundary.

## 4. Run the six full-corpus cells

The helper functions below expand only to CLI options present in the pinned `run --help` output. They run every official test query with at least one positive qrel once, request 50 unique source documents from a 250-candidate response, disable warm passes, and fail closed on stalled ingestion.

```bash
run_embedding_reranking() {
  dataset="$1"
  run_id="$2"
  python3 benchmarks/beir-retrieval-quality/runner/run_benchmark.py run \
    --base-url "$BASE_URL" \
    --metrics-url "$METRICS_URL" \
    --work-dir "$CORTRIX_BENCH_WORKDIR" \
    --datasets "$dataset" \
    --profiles bge_m3_rerank_baseline_v2 \
    --seed 44044 \
    --top-k 50 \
    --batch-size 100 \
    --poll-timeout-seconds 86400 \
    --query-timeout-seconds 900 \
    --query-cooldown-seconds 0 \
    --query-transient-retries 2 \
    --query-transient-retry-backoff-seconds 60 \
    --min-queries 1 \
    --task-monitor-interval-seconds 60 \
    --no-progress-seconds 7200 \
    --fail-on-no-progress \
    --warm-query-repeats 0 \
    --cortrix-data-dir "$CORTRIX_DATA_DIR" \
    --run-id "$run_id"
}

run_full_stack() {
  dataset="$1"
  run_id="$2"
  python3 benchmarks/beir-retrieval-quality/runner/run_benchmark.py run \
    --base-url "$BASE_URL" \
    --metrics-url "$METRICS_URL" \
    --work-dir "$CORTRIX_BENCH_WORKDIR" \
    --datasets "$dataset" \
    --profiles bge_m3_rf_pool_listwise \
    --seed 44044 \
    --top-k 50 \
    --batch-size 100 \
    --poll-timeout-seconds 86400 \
    --query-timeout-seconds 900 \
    --query-cooldown-seconds 0 \
    --query-transient-retries 2 \
    --query-transient-retry-backoff-seconds 60 \
    --min-queries 1 \
    --task-monitor-interval-seconds 60 \
    --no-progress-seconds 7200 \
    --fail-on-no-progress \
    --warm-query-repeats 0 \
    --llm-contract-smoke-report "$LLM_CONTRACT_SMOKE_REPORT" \
    --require-llm-contract-smoke \
    --cortrix-data-dir "$CORTRIX_DATA_DIR" \
    --require-doc-summary-drain \
    --doc-summary-task-type 3 \
    --doc-summary-drain-timeout-seconds 86400 \
    --doc-summary-drain-interval-seconds 5 \
    --doc-summary-no-progress-seconds 7200 \
    --run-id "$run_id"
}
```

Start from a fresh runtime and set a fresh `CORTRIX_DATA_DIR` before each invocation:

```bash
run_embedding_reranking scifact candidate-scifact-er
run_full_stack scifact candidate-scifact-fs
run_embedding_reranking fiqa candidate-fiqa-er
run_full_stack fiqa candidate-fiqa-fs
run_embedding_reranking nfcorpus candidate-nfcorpus-er
run_full_stack nfcorpus candidate-nfcorpus-fs
```

Do not execute the six lines against one shared runtime. The displayed sequence is an invocation matrix; each line requires its own isolated server lifecycle and runtime directory.

## 5. Verify freeze ordering and source-document mapping

For each run, confirm `task_drain` completed before `cold_query`. Full Stack must also show `doc_summary_drain` completed before `cold_query`. The runner stores the original BEIR corpus ID in document metadata, resolves child IDs through a read-only immutable SQLite connection, collapses duplicate chunks to the first source-document rank, and writes `child_id_aliases_summary.json`.

```bash
python3 - "$CORTRIX_BENCH_WORKDIR/runs/candidate-fiqa-fs/stage_events.jsonl" <<'PY'
import json
import sys

events = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
ordered = [(row.get("stage"), row.get("status")) for row in events]
for required in (("task_drain", "completed"), ("doc_summary_drain", "completed"), ("cold_query", "start"), ("cold_query", "completed")):
    if required not in ordered:
        raise SystemExit(f"missing stage event: {required}")
if ordered.index(("task_drain", "completed")) > ordered.index(("cold_query", "start")):
    raise SystemExit("task drain did not precede measurement")
if ordered.index(("doc_summary_drain", "completed")) > ordered.index(("cold_query", "start")):
    raise SystemExit("document-summary drain did not precede measurement")
print("freeze-order: PASS")
PY
```

Run the analogous check without `doc_summary_drain` for Embedding + Reranking. Reject any cell with missing aliases when the API response does not already expose the source ID, any corpus write during `cold_query`, any query failure, or a count mismatch.

## 6. Recompute Recall and NDCG at both cutoffs

Set `RUN_ID`, `PROFILE`, and `DATASET` for a cell, then run:

```bash
export RUN_ID=candidate-fiqa-fs
export PROFILE=bge_m3_rf_pool_listwise
export DATASET=fiqa
export CELL_DIR="$CORTRIX_BENCH_WORKDIR/runs/$RUN_ID/$PROFILE/$DATASET"

PYTHONPATH=benchmarks/beir-retrieval-quality/runner \
python3 - "$CELL_DIR/queries.jsonl" "$CELL_DIR/sample/sampled_qrels.tsv" <<'PY'
import csv
import json
import sys
from metrics import ndcg_at_k, recall_at_k, summarize

runs = {}
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        runs[str(row["query_id"])] = [str(value) for value in row.get("retrieved_doc_ids", [])]

qrels = {}
with open(sys.argv[2], encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        score = float(row["score"])
        if score > 0:
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = score

for cutoff in (10, 50):
    recall = summarize(recall_at_k(qrels, runs, k=cutoff).values())
    ndcg = summarize(ndcg_at_k(qrels, runs, k=cutoff).values())
    print(json.dumps({"cutoff": cutoff, "recall": recall, "ndcg": ndcg}, sort_keys=True))
PY
```

The mean fields must match the corresponding scorecard within ordinary floating-point serialization. Latency comes only from the cold rows and must have the official-query count for that cell.

## 7. Classify scientific and strict status separately

Run the terminal-status classifier against the runner's `summary.json`. Supply a feature-proof file only when an independent run-specific check has produced one; otherwise omit `--feature-proof` and retain `NOT_EVALUATED`.

```bash
python3 benchmarks/beir-retrieval-quality/runner/terminal_status.py \
  --summary "$CORTRIX_BENCH_WORKDIR/runs/$RUN_ID/summary.json" \
  --wrapper-rc 0 \
  --output "$CORTRIX_BENCH_WORKDIR/runs/$RUN_ID/terminal-status.json"
```

Never infer strict PASS from a successful process, a complete query set, or a requested profile. A strict FAIL does not erase a scientifically complete retrieval measurement; it must remain a separate field.

## 8. Validate a regenerated result bundle

After copying only sanitized scorecards and summaries into a new bundle directory, run:

```bash
python3 benchmarks/beir-retrieval-quality/tools/validate_release_candidate.py \
  results/published/beir-three-full-corpus-2026-07-v1

git diff --check
```

The validator checks the six-cell identity, exact measured values, display mapping, completeness, checksum manifest, status separation, and public-safety boundary. Do not copy raw queries, documents, logs, local paths, credentials, endpoints, cloud identifiers, or private operational material into the public bundle.
