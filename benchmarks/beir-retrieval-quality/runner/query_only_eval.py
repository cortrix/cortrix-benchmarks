#!/usr/bin/env python3
"""Query-only recall/nDCG eval against an ALREADY-INGESTED Cortrix namespace.

Why this exists: full-corpus BEIR ingest on CPU bge-m3 is ~8h, far longer than the
runner's poll-timeout. The runner has no skip-ingest/reuse-namespace mode, so this
script reuses the runner's own loaders + metrics + doc-id extraction to score an
existing namespace without re-ingesting. Same doc-collapse logic as run_benchmark
(extract_result_doc_ids + dedup) so the numbers are comparable.

Usage:
  python3 query_only_eval.py <namespace> [--dataset scifact] [--max-queries 50] \
      [--base-url http://127.0.0.1:8420] [--top-k 10] [--timeout-seconds 300] \
      [--split test] [--profile vector_full]

  # Several namespaces in one request, comma-separated. The server merges across
  # them and the merged list is scored as one result set.
  python3 query_only_eval.py ns-shard1,ns-shard2,ns-shard3 --dataset quora \
      --max-queries 2000 --top-k 10 --profile vector_full

Query subset contract: queries are the qrels ids for the chosen split that carry
at least one relevance judgement, sorted lexicographically, then truncated to
--max-queries. The selection depends only on the dataset and the split, never on
wall-clock time, iteration order or the server. The sha256 of the selected id
list is printed as `query_subset` so a published result can be checked against
the exact queries that produced it.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import beir_loader
import metrics
from cortrix_client import CortrixClient, CortrixClientError, extract_result_doc_ids
from profiles import PROFILES

# A request timeout long enough for the slowest arm we measure. A CPU cross-encoder
# rerank over a large fan-out runs tens of seconds per query, and an LLM listwise
# stage on top of it runs longer still; the client default of 30s cuts those off
# mid-flight and loses the run.
DEFAULT_TIMEOUT_SECONDS = 300.0


def parse_namespaces(raw: str) -> List[str]:
    """Split the namespace argument into a list, preserving order.

    A single namespace stays a one-element list. Empty segments are dropped so a
    trailing comma is harmless, and surrounding whitespace is stripped so a
    shell-wrapped list does not silently produce a namespace named " ns".
    """
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_query_body(
    profile: object,
    query: str,
    namespaces: Sequence[str],
    top_k: int,
) -> Dict[str, object]:
    """Build the request body for one query over one or more namespaces.

    The profile owns every retrieval knob; this function only decides how the
    namespaces are addressed. `namespaces` is always sent as a list, including for
    a single namespace, so that a one-namespace run and a many-namespace run take
    the same server path and stay comparable.
    """
    body = dict(profile.query_body(query=query, namespace=namespaces[0], top_k=top_k))
    body["namespaces"] = list(namespaces)
    return body


def dedupe_doc_ids(response: Mapping[str, object], top_k: int) -> List[str]:
    """Collapse a response to at most top_k distinct document ids, in rank order.

    Identical to run_benchmark's collapse so the two report the same thing: a
    document that appears as several chunks counts once, at its best rank.
    """
    seen: set[str] = set()
    doc_ids: List[str] = []
    for doc_id in extract_result_doc_ids(response):
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids[:top_k]


def select_query_ids(qrels: Mapping[str, Mapping[str, float]], max_queries: int) -> List[str]:
    """The deterministic query subset: judged ids, sorted, then capped.

    Sorting before truncating is what makes --max-queries reproducible. Without
    it the subset would depend on dict ordering and two runs of the same command
    could score different queries.
    """
    qids = [qid for qid in sorted(qrels.keys()) if qrels[qid]]
    if max_queries > 0:
        qids = qids[:max_queries]
    return qids


def query_subset_digest(qids: Sequence[str]) -> str:
    """sha256 over the selected query ids, newline-joined, in selection order.

    Published results carry this so a reader can prove they are scoring the same
    queries without having to ship the id list itself.
    """
    joined = "\n".join(qids).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def nearest_rank(ordered: Sequence[float], p: float) -> float:
    """The p-th percentile by nearest rank: the ceil(p*n)-th smallest observation.

    Nearest rank returns an observation that is actually in the sample, so there
    is no interpolation convention for a reader to agree with. It is stated as a
    named function because the obvious spelling, `ordered[int(p * n)]`, is wrong
    whenever p*n is an integer -- it returns the next observation up. At n=2000
    that puts p95 on observation 1901 rather than 1900, and p50 on 1001 rather
    than 1000. The two spellings agree everywhere else, which is what makes the
    error survive casual reading.
    """
    if not ordered:
        raise ValueError("nearest_rank requires a non-empty sample")
    rank = max(1, math.ceil(p * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarise_latency(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """mean/p50/p95/min/max/n over per-query wall time.

    Percentiles are nearest rank; see nearest_rank above for why that is spelled
    out rather than inlined.
    """
    if not latencies_ms:
        return {}
    ordered = sorted(latencies_ms)
    return {
        "mean": statistics.mean(ordered),
        "p50": nearest_rank(ordered, 0.50),
        "p95": nearest_rank(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
        "n": float(len(ordered)),
    }


def build_parser() -> argparse.ArgumentParser:
    """The command-line surface, as a value main() and the tests both use.

    Exposed rather than built inline so a test exercises the parser this script
    actually runs. A test that rebuilds an equivalent parser passes while the
    production defaults or wiring drift away from it, which is the failure mode
    it was supposed to catch.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Score an already-ingested Cortrix namespace against a BEIR dataset. "
            "Pass several namespaces comma-separated to query them as one merged "
            "result set."
        ),
        epilog=(
            "examples:\n"
            "  # single namespace, full SciFact test split\n"
            "  python3 query_only_eval.py my-scifact-ns --dataset scifact "
            "--max-queries 300 --top-k 10 --profile vector_full\n"
            "\n"
            "  # eight namespaces merged, 2000 Quora queries, long timeout for a\n"
            "  # CPU cross-encoder arm\n"
            "  python3 query_only_eval.py ns1,ns2,ns3,ns4,ns5,ns6,ns7,ns8 "
            "--dataset quora \\\n"
            "      --max-queries 2000 --top-k 10 --profile bge_m3_rerank_full "
            "--timeout-seconds 300\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "namespace",
        help="namespace to score; comma-separate several to merge them in one request",
    )
    ap.add_argument("--dataset", default="scifact", choices=sorted(beir_loader.DATASETS))
    ap.add_argument("--split", default="test", help="qrels split to score against")
    ap.add_argument(
        "--max-queries",
        type=int,
        default=50,
        help="cap the judged, sorted query list; 0 or less means no cap",
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8420")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--profile", default="vector_full", choices=sorted(PROFILES))
    ap.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "per-request timeout in seconds (default: %(default)s). Reranking and "
            "LLM arms on CPU exceed the client default of 30s."
        ),
    )
    ap.add_argument("--work-dir", default="/tmp/cortrix-benchmarks/beir")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    namespaces = parse_namespaces(args.namespace)
    if not namespaces:
        ap.error("namespace must name at least one namespace")

    root = Path(args.work_dir) / "datasets" / args.dataset / args.dataset
    qrels = beir_loader.load_qrels(root, args.split)
    qids = select_query_ids(qrels, args.max_queries)
    queries = beir_loader.load_queries(root, set(qids))

    profile = PROFILES[args.profile]
    top_k = args.top_k
    retrieval_k = max(top_k, top_k * 5)
    client = CortrixClient(args.base_url, timeout_seconds=args.timeout_seconds)

    runs: Dict[str, List[str]] = {}
    latencies_ms: List[float] = []
    missing = 0
    failed = 0
    for qid in qids:
        qtext = queries.get(qid)
        if not qtext:
            missing += 1
            continue
        body = build_query_body(profile, qtext, namespaces, retrieval_k)
        try:
            response, latency_ms = client.query(body)
        except CortrixClientError:
            # A failed request is counted and excluded rather than scored as an
            # empty result. Scoring it as empty would quietly depress the metric
            # and make a broken run look like a bad one.
            failed += 1
            continue
        latencies_ms.append(latency_ms)
        runs[qid] = dedupe_doc_ids(response, top_k)

    scored_qrels = {qid: qrels[qid] for qid in qids if qid in runs}
    rec = metrics.recall_at_k(scored_qrels, runs, top_k)
    ndcg = metrics.ndcg_at_k(scored_qrels, runs, top_k)

    def mean(values: Mapping[str, float]) -> float:
        return statistics.mean(values.values()) if values else 0.0

    print(f"namespace      : {','.join(namespaces)}")
    print(f"namespace_count: {len(namespaces)}")
    print(f"dataset        : {args.dataset} (split={args.split} profile={args.profile})")
    print(
        f"queries scored : {len(rec)} (requested {len(qids)}, "
        f"missing_text={missing}, failed={failed})"
    )
    print(f"query_subset   : sha256={query_subset_digest(qids)}")
    print(f"recall@{top_k}      : {mean(rec):.4f}")
    print(f"ndcg@{top_k}        : {mean(ndcg):.4f}")
    latency = summarise_latency(latencies_ms)
    if latency:
        print(
            f"latency_ms     : mean={latency['mean']:.0f} p50={latency['p50']:.0f} "
            f"p95={latency['p95']:.0f} min={latency['min']:.0f} "
            f"max={latency['max']:.0f} n={int(latency['n'])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
