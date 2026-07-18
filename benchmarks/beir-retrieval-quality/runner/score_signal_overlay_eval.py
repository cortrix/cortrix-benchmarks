#!/usr/bin/env python3
"""Evaluate a fixed query log after overlaying SQLite score signals.

This is a local diagnostic helper for already-generated Cortrix query logs. It
does not query Cortrix, ingest documents, or modify the SQLite store. It keeps
the returned candidate set fixed, reads optional F03/F07 score columns from the
`blocks` table by `child_id`, recomputes a candidate ordering score, then reports
BEIR recall/nDCG using the same document-level collapse convention as the main
runner.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import metrics
from cortrix_client import extract_result_doc_ids


ScoreSignals = Mapping[str, Optional[float]]
QueryRuns = Dict[str, Sequence[str]]


def load_sampled_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            query_id, doc_id, score_text = parts[:3]
            try:
                score = float(score_text)
            except ValueError:
                continue
            qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def iter_query_rows(path: Path) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item


def response_results(row: Mapping[str, object]) -> list[Mapping[str, object]]:
    response = row.get("response")
    if not isinstance(response, dict):
        return []
    results = response.get("results")
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)]


def collapse_doc_ids(items: Sequence[Mapping[str, object]]) -> list[str]:
    doc_ids = extract_result_doc_ids({"results": list(items)})
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in doc_ids:
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def load_signals(store_db: Path, child_ids: Iterable[str]) -> Dict[str, ScoreSignals]:
    conn = sqlite3.connect(f"file:{store_db}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    signals: Dict[str, ScoreSignals] = {}
    ids = sorted(set(child_id for child_id in child_ids if child_id))
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        sql = (
            "select child_id, enriched_score, semantic_score "
            f"from blocks where child_id in ({placeholders})"
        )
        for row in conn.execute(sql, batch):
            signals[str(row["child_id"])] = {
                "enriched_score": row["enriched_score"],
                "semantic_score": row["semantic_score"],
            }
    conn.close()
    return signals


def fixed_candidate_score(item: Mapping[str, object], signals_by_child: Mapping[str, ScoreSignals]) -> tuple[float, float, bool]:
    visible_score = float(item.get("score") or 0.0)
    rerank_score = float(item.get("rerank_score") or 0.0)
    base_score = 0.7 * rerank_score + 0.3 * visible_score
    child_id = item.get("child_id")
    sig = signals_by_child.get(child_id) if isinstance(child_id, str) else None
    if not sig:
        return base_score, 1.0, False
    enriched_score = sig.get("enriched_score")
    semantic_score = sig.get("semantic_score")
    if enriched_score is None and semantic_score is None:
        return base_score, 1.0, False
    effective_semantic = float(enriched_score if enriched_score is not None else semantic_score)
    multiplier = 1.0 + 0.2 * (effective_semantic - 0.5)
    return base_score * multiplier, multiplier, True


def build_runs(rows: Sequence[Mapping[str, object]], signals_by_child: Mapping[str, ScoreSignals], top_k: int) -> tuple[Mapping[str, QueryRuns], Mapping[str, object]]:
    runs: MutableMapping[str, QueryRuns] = {
        "official_retrieved_doc_ids": {},
        "response_order": {},
        "visible_score_desc": {},
        "rerank_score_desc": {},
        "fixed_candidate_score_desc": {},
    }
    multipliers: list[float] = []
    returned_results = 0
    results_with_child_id = 0
    results_matched_in_db = 0
    results_with_any_signal = 0
    results_with_enriched_score = 0
    results_with_semantic_score = 0
    queries_top10_changed = 0

    for row in rows:
        query_id = str(row.get("query_id", ""))
        items = response_results(row)
        returned_results += len(items)
        scored_items: list[dict[str, object]] = []

        for index, item in enumerate(items):
            child_id = item.get("child_id")
            if isinstance(child_id, str) and child_id:
                results_with_child_id += 1
            sig = signals_by_child.get(child_id) if isinstance(child_id, str) else None
            if sig is not None:
                results_matched_in_db += 1
                if sig.get("enriched_score") is not None:
                    results_with_enriched_score += 1
                if sig.get("semantic_score") is not None:
                    results_with_semantic_score += 1
                if sig.get("enriched_score") is not None or sig.get("semantic_score") is not None:
                    results_with_any_signal += 1
            score, multiplier, used_signal = fixed_candidate_score(item, signals_by_child)
            if used_signal:
                multipliers.append(multiplier)
            scored = dict(item)
            scored["_fixed_candidate_score"] = score
            scored["_original_index"] = index
            scored_items.append(scored)

        official = row.get("retrieved_doc_ids")
        runs["official_retrieved_doc_ids"][query_id] = list(official if isinstance(official, list) else [])[:top_k]
        runs["response_order"][query_id] = collapse_doc_ids(items)[:top_k]
        runs["visible_score_desc"][query_id] = collapse_doc_ids(
            sorted(items, key=lambda item: float(item.get("score") or 0.0), reverse=True)
        )[:top_k]
        runs["rerank_score_desc"][query_id] = collapse_doc_ids(
            sorted(items, key=lambda item: float(item.get("rerank_score") or 0.0), reverse=True)
        )[:top_k]
        fixed_docs = collapse_doc_ids(
            sorted(
                scored_items,
                key=lambda item: (float(item.get("_fixed_candidate_score") or 0.0), -int(item.get("_original_index") or 0)),
                reverse=True,
            )
        )[:top_k]
        runs["fixed_candidate_score_desc"][query_id] = fixed_docs
        if fixed_docs != runs["response_order"][query_id]:
            queries_top10_changed += 1

    coverage = {
        "queries": len(rows),
        "returned_results": returned_results,
        "results_with_child_id": results_with_child_id,
        "results_matched_in_db": results_matched_in_db,
        "results_with_any_signal": results_with_any_signal,
        "results_with_enriched_score": results_with_enriched_score,
        "results_with_semantic_score": results_with_semantic_score,
        "queries_top10_changed_vs_response": queries_top10_changed,
        "semantic_multiplier": summarize(multipliers),
    }
    return runs, coverage


def summarize(values: Iterable[float]) -> Mapping[str, float | int]:
    values_list = sorted(float(value) for value in values)
    if not values_list:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values_list),
        "mean": statistics.mean(values_list),
        "p50": percentile(values_list, 50),
        "p95": percentile(values_list, 95),
        "min": values_list[0],
        "max": values_list[-1],
    }


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def metric_summary(qrels: Mapping[str, Mapping[str, float]], runs: Mapping[str, Sequence[str]], top_k: int) -> Mapping[str, float | int]:
    recall = metrics.recall_at_k(qrels, runs, top_k)
    ndcg = metrics.ndcg_at_k(qrels, runs, top_k)
    return {
        "queries": len(recall),
        f"recall@{top_k}": statistics.mean(recall.values()) if recall else 0.0,
        f"ndcg@{top_k}": statistics.mean(ndcg.values()) if ndcg else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--store-db", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    rows = list(iter_query_rows(args.queries))
    child_ids = [
        str(item["child_id"])
        for row in rows
        for item in response_results(row)
        if isinstance(item.get("child_id"), str) and item.get("child_id")
    ]
    signals_by_child = load_signals(args.store_db, child_ids)
    qrels = load_sampled_qrels(args.qrels)
    runs, coverage = build_runs(rows, signals_by_child, args.top_k)

    output = {
        "inputs": {
            "queries": str(args.queries),
            "qrels": str(args.qrels),
            "store_db": str(args.store_db),
            "top_k": args.top_k,
        },
        "coverage": coverage,
        "metrics": {
            name: metric_summary(qrels, run, args.top_k)
            for name, run in runs.items()
        },
        "label": "local diagnostic",
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
