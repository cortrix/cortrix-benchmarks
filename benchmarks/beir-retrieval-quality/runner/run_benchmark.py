#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from beir_loader import DATASETS, DatasetSpec, ensure_dataset, file_size_bytes
from cortrix_client import (
    CortrixClient,
    CortrixClientError,
    collect_query_latency_metrics,
    diff_query_latency_metric_snapshots,
    extract_result_doc_ids,
    extract_task_ids,
)
from diagnostics import (
    TaskDrainDiagnosticError,
    append_stage_event,
    capture_log_tail,
    poll_sqlite_task_type_with_diagnostics,
    poll_tasks_with_diagnostics,
    write_latency_csv,
)
from llm_contract_smoke import validate_contract_smoke_report
from manifest import file_manifest, write_json
from metrics import aggregate, recall_at_k, summarize
from profiles import PROFILES, BenchmarkProfile
from resource_probe import append_jsonl, snapshot
from sample_plan import iter_sampled_corpus, load_sampled_queries, make_sample


DEFAULT_WORK_DIR = Path(os.environ.get("CORTRIX_BENCH_WORKDIR", "/tmp/cortrix-benchmarks/beir-retrieval-quality"))


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _pace_query(query_index: int, cooldown_seconds: float) -> None:
    if query_index > 0 and cooldown_seconds > 0:
        time.sleep(cooldown_seconds)


_TRANSIENT_DEGRADE_REASONS = {
    "circuit_open",
    "llm_timeout",
    "llm_transport",
    "quota_exceeded",
}


def _transient_query_exception_reason(exc: Exception) -> str:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return type(exc).__name__
    if not isinstance(exc, CortrixClientError):
        return ""
    message = str(exc).lower()
    if " failed: " in message or "timed out" in message:
        return "transport"
    if "http 429" in message:
        return "http_429"
    if any(f"http {status}" in message for status in range(500, 600)):
        return "http_5xx"
    return ""


class QueryRetriesExhausted(RuntimeError):
    def __init__(self, cause: Exception, attempts: Sequence[Mapping[str, object]]) -> None:
        super().__init__(str(cause))
        self.attempts = list(attempts)


def _llm_degrade_reason(response: Mapping[str, object]) -> str:
    explain = response.get("explain")
    if not isinstance(explain, Mapping):
        return ""
    features = explain.get("llm_dependent_features")
    if not isinstance(features, Mapping):
        return ""
    for feature_name, raw_state in features.items():
        if not isinstance(raw_state, Mapping) or not raw_state.get("degraded"):
            continue
        reason = str(raw_state.get("degrade_reason") or "")
        detail = str(raw_state.get("degrade_detail") or "")
        return f"{feature_name}:{reason}" + (f":{detail}" if detail else "")
    return ""


def _transient_degrade_reason(response: Mapping[str, object]) -> str:
    degrade = _llm_degrade_reason(response)
    if not degrade:
        return ""
    parts = degrade.split(":", 2)
    reason = parts[1] if len(parts) > 1 else ""
    detail = parts[2] if len(parts) > 2 else ""
    if reason in _TRANSIENT_DEGRADE_REASONS:
        return degrade
    if reason == "llm_http" and detail.startswith("http_status=5"):
        return degrade
    return ""


def _query_with_transient_retries(
    client: CortrixClient,
    body: Mapping[str, object],
    max_retries: int,
    backoff_seconds: float,
) -> tuple[Mapping[str, object], float, List[Mapping[str, object]]]:
    attempts: List[Mapping[str, object]] = []
    total_latency_ms = 0.0
    for attempt_index in range(max_retries + 1):
        attempt_started = time.perf_counter()
        try:
            response, latency_ms = client.query(body)
        except Exception as exc:
            latency_ms = (time.perf_counter() - attempt_started) * 1000.0
            total_latency_ms += latency_ms
            retry_reason = _transient_query_exception_reason(exc)
            retry_wait_seconds = (
                backoff_seconds if retry_reason and attempt_index < max_retries else 0.0
            )
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "exception",
                    "error": repr(exc),
                    "latency_ms": latency_ms,
                    "retry_reason": retry_reason,
                    "backoff_seconds": retry_wait_seconds,
                }
            )
            if not retry_reason:
                raise QueryRetriesExhausted(exc, attempts) from exc
            if attempt_index >= max_retries:
                raise QueryRetriesExhausted(exc, attempts) from exc
            time.sleep(backoff_seconds)
            total_latency_ms += backoff_seconds * 1000.0
            continue

        total_latency_ms += latency_ms
        degrade_reason = _llm_degrade_reason(response)
        retry_reason = _transient_degrade_reason(response)
        retry_wait_seconds = (
            backoff_seconds if retry_reason and attempt_index < max_retries else 0.0
        )
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "status": (
                    "transient_degrade"
                    if retry_reason
                    else ("degraded" if degrade_reason else "success")
                ),
                "latency_ms": latency_ms,
                "degrade_reason": degrade_reason,
                "retry_reason": retry_reason,
                "backoff_seconds": retry_wait_seconds,
            }
        )
        if not retry_reason or attempt_index >= max_retries:
            return response, total_latency_ms, attempts
        time.sleep(backoff_seconds)
        total_latency_ms += backoff_seconds * 1000.0
    raise AssertionError("unreachable query retry loop")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Cortrix BEIR retrieval quality sampled benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("selftest", help="Run local metric and manifest self-tests.")

    run = sub.add_parser("run", help="Run sampled BEIR benchmark against a Cortrix API server.")
    run.add_argument("--base-url", default="http://127.0.0.1:8080")
    run.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    run.add_argument("--datasets", default="scifact,nfcorpus,fiqa,webis-touche2020,hotpotqa")
    run.add_argument("--profiles", default="vector_only,full_stack")
    run.add_argument("--seed", type=int, default=44044)
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--batch-size", type=int, default=100)
    run.add_argument("--poll-timeout-seconds", type=float, default=1800.0)
    # Per-request HTTP timeout. The 30s client default aborts a whole cell on the
    # first slow query (cold reranker load, CPU rerank, or LLM-arm calls run
    # 30-90s+); LLM isolation arms should raise this to >= 300.
    run.add_argument("--query-timeout-seconds", type=float, default=30.0)
    # Optional benchmark-side pacing between completed query calls. This is
    # outside the measured per-query HTTP latency and is useful for LLM profiles
    # that would otherwise exceed an external provider's request/token window.
    run.add_argument("--query-cooldown-seconds", type=_nonnegative_float, default=0.0)
    run.add_argument("--query-transient-retries", type=int, default=0)
    run.add_argument(
        "--query-transient-retry-backoff-seconds",
        type=_nonnegative_float,
        default=60.0,
    )
    # Reuse an already-ingested namespace: skip create + document batch + task
    # drain + doc-summary drain, and query the given namespace directly. Lets the
    # ingest-identical LLM arms (B/C/D) share ONE GLM ingest instead of paying for
    # three. Only valid with a single dataset + single profile per invocation.
    run.add_argument("--reuse-namespace", default="")
    run.add_argument("--min-queries", type=int, default=20)
    run.add_argument("--max-queries", type=int, default=0)
    run.add_argument("--max-corpus-docs", type=int, default=0)
    run.add_argument("--task-monitor-interval-seconds", type=float, default=60.0)
    run.add_argument("--task-poll-interval-seconds", type=float, default=2.0)
    run.add_argument("--task-poll-batch-size", type=int, default=500)
    run.add_argument("--task-sample-size", type=int, default=20)
    run.add_argument("--no-progress-seconds", type=float, default=1200.0)
    run.add_argument("--fail-on-no-progress", action="store_true")
    run.add_argument("--log-tail-lines", type=int, default=300)
    run.add_argument("--metrics-url", default="http://127.0.0.1:9091/metrics")
    run.add_argument("--warm-query-repeats", type=int, default=1)
    run.add_argument("--run-id", default="")
    run.add_argument("--container-name", default="cortrix")
    run.add_argument("--llm-contract-smoke-report", type=Path, default=None)
    run.add_argument("--require-llm-contract-smoke", action="store_true")
    run.add_argument("--cortrix-data-dir", type=Path, default=None)
    run.add_argument("--require-doc-summary-drain", action="store_true")
    run.add_argument("--doc-summary-task-type", type=int, default=3)
    run.add_argument("--doc-summary-drain-timeout-seconds", type=float, default=0.0)
    run.add_argument("--doc-summary-drain-interval-seconds", type=float, default=5.0)
    run.add_argument("--doc-summary-no-progress-seconds", type=float, default=0.0)

    args = parser.parse_args(argv)
    if getattr(args, "query_transient_retries", 0) < 0:
        parser.error("--query-transient-retries must be >= 0")
    if args.command == "selftest":
        return selftest()
    if args.command == "run":
        return run_benchmark(args)
    return 2


def selftest() -> int:
    from cortrix_client import extract_result_doc_ids
    from metrics import ndcg_at_k, recall_at_k

    qrels = {"q1": {"d1": 1.0, "d2": 1.0}, "q2": {"d3": 2.0}}
    runs = {"q1": ["d2", "d9"], "q2": ["d4", "d3"]}
    recall = recall_at_k(qrels, runs, k=2)
    ndcg = ndcg_at_k(qrels, runs, k=2)
    assert math.isclose(recall["q1"], 0.5)
    assert math.isclose(recall["q2"], 1.0)
    assert 0.0 < ndcg["q1"] <= 1.0
    assert 0.0 < ndcg["q2"] <= 1.0
    mapped = extract_result_doc_ids(
        {
            "results": [
                {"metadata": {"metadata_json": "{\"filename\":\"scifact-12345.txt\"}"}},
                {"metadata": {"metadata_json": "{\"filename\":\"67890.txt\"}"}},
            ]
        }
    )
    assert mapped == ["12345", "67890"]
    print(json.dumps({"status": "ok", "selftest": "metrics"}, indent=2))
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    work_dir: Path = args.work_dir
    run_dir = work_dir / "runs" / run_id
    dataset_cache = work_dir / "datasets"
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets = _parse_names(args.datasets, DATASETS, "dataset")
    profiles = _parse_names(args.profiles, PROFILES, "profile")
    client = CortrixClient(args.base_url, timeout_seconds=args.query_timeout_seconds)
    resource_log = run_dir / "resource_snapshots.jsonl"
    import_batch_log = run_dir / "import_batches.jsonl"
    task_drain_log = run_dir / "task_drain_snapshots.jsonl"
    doc_summary_drain_log = run_dir / "doc_summary_drain_snapshots.jsonl"
    stage_log = run_dir / "stage_events.jsonl"
    log_tail_dir = run_dir / "log_tail_snapshots"

    manifest = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "datasets": datasets,
        "profiles": profiles,
        "seed": args.seed,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "max_queries": args.max_queries or None,
        "max_corpus_docs": args.max_corpus_docs or None,
        "task_monitor_interval_seconds": args.task_monitor_interval_seconds,
        "task_poll_interval_seconds": args.task_poll_interval_seconds,
        "task_poll_batch_size": args.task_poll_batch_size,
        "task_sample_size": args.task_sample_size,
        "no_progress_seconds": args.no_progress_seconds,
        "fail_on_no_progress": args.fail_on_no_progress,
        "warm_query_repeats": args.warm_query_repeats,
        "query_cooldown_seconds": args.query_cooldown_seconds,
        "query_transient_retries": args.query_transient_retries,
        "query_transient_retry_backoff_seconds": args.query_transient_retry_backoff_seconds,
        "llm_contract_smoke_report": str(args.llm_contract_smoke_report) if args.llm_contract_smoke_report else None,
        "require_llm_contract_smoke": args.require_llm_contract_smoke,
        "cortrix_data_dir": str(args.cortrix_data_dir) if args.cortrix_data_dir else None,
        "require_doc_summary_drain": args.require_doc_summary_drain,
        "doc_summary_task_type": args.doc_summary_task_type,
        "doc_summary_drain_timeout_seconds": args.doc_summary_drain_timeout_seconds,
        "doc_summary_drain_interval_seconds": args.doc_summary_drain_interval_seconds,
        "doc_summary_no_progress_seconds": args.doc_summary_no_progress_seconds,
        "work_dir": str(work_dir),
        "diagnostic_artifacts": {
            "import_batches": str(import_batch_log),
            "task_drain_snapshots": str(task_drain_log),
            "doc_summary_drain_snapshots": str(doc_summary_drain_log),
            "resource_snapshots": str(resource_log),
            "stage_events": str(stage_log),
            "log_tail_snapshots": str(log_tail_dir),
        },
        "api_boundary": {
            "query_endpoint": "POST /api/v1/query",
            "document_batch_endpoint": "POST /api/v1/documents/batch",
            "namespace_endpoint": "POST /api/v1/namespaces",
            "note": (
                "The runner records requested profile semantics separately from "
                "the actual API body so external claims do not overstate API routing."
            ),
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "profile_manifest.json", {name: PROFILES[name].to_json() for name in profiles})

    if args.require_llm_contract_smoke:
        append_stage_event(stage_log, "llm_contract_smoke_gate", "start")
        if args.llm_contract_smoke_report is None:
            error = "missing --llm-contract-smoke-report"
            write_json(run_dir / "run_error.json", {"stage": "llm_contract_smoke_gate", "error": error})
            append_stage_event(stage_log, "llm_contract_smoke_gate", "failed", {"error": error})
            print(f"[error] {error}", file=sys.stderr)
            return 1
        try:
            smoke_report = validate_contract_smoke_report(args.llm_contract_smoke_report)
        except Exception as exc:  # noqa: BLE001
            write_json(run_dir / "run_error.json", {"stage": "llm_contract_smoke_gate", "error": str(exc)})
            append_stage_event(stage_log, "llm_contract_smoke_gate", "failed", {"error": str(exc)})
            print(f"[error] llm contract smoke gate failed: {exc}", file=sys.stderr)
            return 1
        write_json(run_dir / "llm_contract_smoke_gate.json", dict(smoke_report))
        append_stage_event(stage_log, "llm_contract_smoke_gate", "completed", {"report": str(args.llm_contract_smoke_report)})

    append_stage_event(stage_log, "health_check", "start")
    append_jsonl(resource_log, snapshot("before_health_check", {"run_dir": run_dir, "dataset_cache": dataset_cache}, args.container_name))
    try:
        health = client.health()
        readiness = client.readiness()
    except CortrixClientError as exc:
        write_json(run_dir / "run_error.json", {"stage": "health_check", "error": str(exc)})
        append_stage_event(stage_log, "health_check", "failed", {"error": str(exc)})
        print(f"[error] health check failed: {exc}", file=sys.stderr)
        return 1
    write_json(run_dir / "server_health.json", {"health": dict(health), "readiness": dict(readiness)})
    append_stage_event(stage_log, "health_check", "completed")
    append_jsonl(resource_log, snapshot("after_health_check", {"run_dir": run_dir, "dataset_cache": dataset_cache}, args.container_name))

    all_results: List[Mapping[str, object]] = []
    for profile_name in profiles:
        profile = PROFILES[profile_name]
        for dataset_name in datasets:
            spec = DATASETS[dataset_name]
            try:
                result = run_cell(
                    run_dir=run_dir,
                    dataset_cache=dataset_cache,
                    spec=spec,
                    profile=profile,
                    client=client,
                    run_id=run_id,
                    seed=args.seed,
                    top_k=args.top_k,
                    batch_size=args.batch_size,
                    poll_timeout_seconds=args.poll_timeout_seconds,
                    min_queries=args.min_queries,
                    max_queries=args.max_queries or None,
                    max_corpus_docs=args.max_corpus_docs or None,
                    task_monitor_interval_seconds=args.task_monitor_interval_seconds,
                    task_poll_interval_seconds=args.task_poll_interval_seconds,
                    task_poll_batch_size=args.task_poll_batch_size,
                    task_sample_size=args.task_sample_size,
                    no_progress_seconds=args.no_progress_seconds,
                    fail_on_no_progress=args.fail_on_no_progress,
                    log_tail_lines=args.log_tail_lines,
                    metrics_url=args.metrics_url,
                    warm_query_repeats=args.warm_query_repeats,
                    query_cooldown_seconds=args.query_cooldown_seconds,
                    query_transient_retries=args.query_transient_retries,
                    query_transient_retry_backoff_seconds=args.query_transient_retry_backoff_seconds,
                    resource_log=resource_log,
                    import_batch_log=import_batch_log,
                    task_drain_log=task_drain_log,
                    doc_summary_drain_log=doc_summary_drain_log,
                    stage_log=stage_log,
                    run_log_tail_dir=log_tail_dir,
                    container_name=args.container_name,
                    cortrix_data_dir=args.cortrix_data_dir,
                    require_doc_summary_drain=args.require_doc_summary_drain,
                    doc_summary_task_type=args.doc_summary_task_type,
                    doc_summary_drain_timeout_seconds=args.doc_summary_drain_timeout_seconds or args.poll_timeout_seconds,
                    doc_summary_drain_interval_seconds=args.doc_summary_drain_interval_seconds,
                    reuse_namespace=args.reuse_namespace,
                    doc_summary_no_progress_seconds=args.doc_summary_no_progress_seconds or args.no_progress_seconds,
                )
                all_results.append(result)
            except Exception as exc:  # noqa: BLE001
                cell_error = {
                    "dataset": dataset_name,
                    "profile": profile_name,
                    "status": "failed",
                    "error": repr(exc),
                }
                if isinstance(exc, TaskDrainDiagnosticError):
                    cell_error["diagnosis"] = exc.summary.get("diagnosis")
                    cell_error["task_drain_summary"] = exc.summary
                all_results.append(cell_error)
                cell_dir = run_dir / profile_name / dataset_name
                write_json(cell_dir / "cell_error.json", cell_error)
                print(f"[error] {profile_name}/{dataset_name}: {exc}", file=sys.stderr)

    summary = {"run_id": run_id, "results": all_results}
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "sample_manifest.json",
        {
            "run_id": run_id,
            "samples": [
                item["sample"]
                for item in all_results
                if isinstance(item, dict) and isinstance(item.get("sample"), dict)
            ],
        },
    )
    append_jsonl(resource_log, snapshot("after_run", {"run_dir": run_dir, "dataset_cache": dataset_cache}, args.container_name))
    print(json.dumps(summary, indent=2))
    return 0 if all(item.get("status") == "completed" for item in all_results) else 1


def run_cell(
    run_dir: Path,
    dataset_cache: Path,
    spec: DatasetSpec,
    profile: BenchmarkProfile,
    client: CortrixClient,
    run_id: str,
    seed: int,
    top_k: int,
    batch_size: int,
    poll_timeout_seconds: float,
    min_queries: int,
    max_queries: int | None,
    max_corpus_docs: int | None,
    task_monitor_interval_seconds: float,
    task_poll_interval_seconds: float,
    task_poll_batch_size: int,
    task_sample_size: int,
    no_progress_seconds: float,
    fail_on_no_progress: bool,
    log_tail_lines: int,
    metrics_url: str,
    warm_query_repeats: int,
    query_cooldown_seconds: float,
    query_transient_retries: int,
    query_transient_retry_backoff_seconds: float,
    resource_log: Path,
    import_batch_log: Path,
    task_drain_log: Path,
    doc_summary_drain_log: Path,
    stage_log: Path,
    run_log_tail_dir: Path,
    container_name: str,
    cortrix_data_dir: Path | None,
    require_doc_summary_drain: bool,
    doc_summary_task_type: int,
    doc_summary_drain_timeout_seconds: float,
    doc_summary_drain_interval_seconds: float,
    doc_summary_no_progress_seconds: float,
    reuse_namespace: str = "",
) -> Mapping[str, object]:
    cell_dir = run_dir / profile.name / spec.name
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell_import_batch_log = cell_dir / "import_batches.jsonl"
    cell_task_drain_log = cell_dir / "task_drain_snapshots.jsonl"
    cell_doc_summary_drain_log = cell_dir / "doc_summary_drain_snapshots.jsonl"
    cell_log_tail_dir = cell_dir / "log_tail_snapshots"
    cell_label = f"{profile.name}/{spec.name}"
    print(f"[cell] start {profile.name}/{spec.name}")

    append_stage_event(stage_log, "dataset_load", "start", {"profile": profile.name, "dataset": spec.name})
    append_jsonl(resource_log, snapshot(f"{profile.name}/{spec.name}/before_dataset", {"cell_dir": cell_dir}, container_name))
    dataset_root = ensure_dataset(spec, dataset_cache)
    append_stage_event(stage_log, "dataset_load", "completed", {"profile": profile.name, "dataset": spec.name, "dataset_root": str(dataset_root)})
    append_jsonl(resource_log, snapshot(f"{profile.name}/{spec.name}/after_dataset", {"dataset_root": dataset_root, "cell_dir": cell_dir}, container_name))

    append_stage_event(stage_log, "sample_materialization", "start", {"profile": profile.name, "dataset": spec.name})
    sample_dir = cell_dir / "sample"
    sample = make_sample(
        dataset=spec.name,
        root=dataset_root,
        output_dir=sample_dir,
        fraction=profile.query_fraction,
        seed=seed,
        split=spec.split,
        min_queries=min_queries,
        max_queries=max_queries,
        max_corpus_rows=max_corpus_docs,
    )
    write_json(cell_dir / "sample_manifest.json", sample.to_json())
    append_stage_event(stage_log, "sample_materialization", "completed", {"profile": profile.name, "dataset": spec.name, "sample": sample.to_json()})
    append_jsonl(resource_log, snapshot(f"{profile.name}/{spec.name}/after_sample", {"sample_dir": sample_dir, "dataset_root": dataset_root}, container_name))

    namespace = reuse_namespace or _namespace(run_id, profile.name, spec.name)
    if not reuse_namespace:
        client.create_namespace(namespace)
    task_ids: List[str] = []
    imported_docs = 0
    if reuse_namespace:
        append_stage_event(stage_log, "batch_submit", "skipped", {"profile": profile.name, "dataset": spec.name, "namespace": namespace, "reason": "reuse_namespace"})
    else:
        append_stage_event(stage_log, "batch_submit", "start", {"profile": profile.name, "dataset": spec.name, "namespace": namespace})
    corpus_batches = [] if reuse_namespace else _batched(iter_sampled_corpus(sample.sampled_corpus_path), batch_size)
    for batch_no, batch in enumerate(corpus_batches, start=1):
        docs = [_corpus_row_to_document(spec.name, row) for row in batch]
        response = client.submit_documents_batch_with_meta(namespace, docs)
        batch_task_ids = extract_task_ids(response.body)
        task_ids.extend(batch_task_ids)
        imported_docs += len(docs)
        trace = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": run_id,
            "profile": profile.name,
            "dataset": spec.name,
            "namespace": namespace,
            "batch_index": batch_no,
            "docs_submitted": len(docs),
            "http_status": response.status,
            "task_ids_count": len(batch_task_ids),
            "request_latency_ms": response.latency_ms,
            "response_size_bytes": response.response_bytes,
            "response_keys": sorted(response.body.keys()),
            "response_meta": response.body.get("meta"),
        }
        append_jsonl(import_batch_log, trace)
        append_jsonl(cell_import_batch_log, trace)
        if batch_no % 50 == 0:
            print(f"[import] {profile.name}/{spec.name}: batches={batch_no} docs={imported_docs}")
    append_stage_event(
        stage_log,
        "batch_submit",
        "completed",
        {"profile": profile.name, "dataset": spec.name, "imported_docs": imported_docs, "task_ids": len(task_ids)},
    )
    if task_ids:
        append_stage_event(stage_log, "task_drain", "start", {"profile": profile.name, "dataset": spec.name, "task_ids": len(task_ids)})
        try:
            task_drain = poll_tasks_with_diagnostics(
                client,
                task_ids,
                timeout_seconds=poll_timeout_seconds,
                interval_seconds=task_poll_interval_seconds,
                monitor_interval_seconds=task_monitor_interval_seconds,
                no_progress_seconds=no_progress_seconds,
                fail_on_no_progress=fail_on_no_progress,
                poll_batch_size=task_poll_batch_size,
                sample_size=task_sample_size,
                log_tail_lines=log_tail_lines,
                task_log=task_drain_log,
                resource_log=resource_log,
                resource_paths={"sample_dir": sample_dir, "cell_dir": cell_dir, "run_dir": run_dir, "dataset_cache": dataset_cache},
                log_tail_dir=run_log_tail_dir,
                container_name=container_name,
                metrics_url=metrics_url,
                label=cell_label,
            )
        except TaskDrainDiagnosticError as exc:
            append_jsonl(cell_task_drain_log, exc.summary)
            append_stage_event(stage_log, "task_drain", "failed", {"profile": profile.name, "dataset": spec.name, "summary": exc.summary})
            raise
        append_jsonl(cell_task_drain_log, task_drain["summary"])
        append_stage_event(stage_log, "task_drain", "completed", {"profile": profile.name, "dataset": spec.name, "summary": task_drain["summary"]})
        # A drain where every task terminated but none succeeded is an ingest
        # wipeout (e.g. server-side CX_ERR_PARSE_FAILED on every document when
        # docling_bridge.py is unreachable because CORTRIX_SCRIPTS_DIR is not
        # set for the server). Without this guard the cell continues into
        # alias building against an empty store.db and fails with a misleading
        # "no such table: blocks" sqlite error. Observed live 2026-07-08.
        drain_summary = task_drain["summary"]
        drain_completed = int(drain_summary.get("completed") or 0)
        drain_failed = int(drain_summary.get("failed") or 0)
        if drain_completed == 0 and drain_failed > 0:
            raise RuntimeError(
                f"ingest wipeout: 0 of {drain_summary.get('total')} tasks succeeded "
                f"({drain_failed} failed); check the server log for per-task errors "
                f"(first suspect: parser/LLM config such as CORTRIX_SCRIPTS_DIR)"
            )
        if drain_failed:
            print(
                f"[warn] {cell_label}: {drain_failed}/{drain_summary.get('total')} "
                f"ingest tasks failed; metrics will run on the partial corpus"
            )
    append_jsonl(resource_log, snapshot(f"{profile.name}/{spec.name}/after_import", {"sample_dir": sample_dir, "cell_dir": cell_dir}, container_name))

    doc_summary_drain: Mapping[str, object] | None = None
    if require_doc_summary_drain and not reuse_namespace:
        if cortrix_data_dir is None:
            raise TaskDrainDiagnosticError(
                "doc-summary drain requires --cortrix-data-dir",
                {
                    "diagnosis": "doc_summary_drain_missing_data_dir",
                    "dataset": spec.name,
                    "profile": profile.name,
                    "namespace": namespace,
                    "expected_total": imported_docs,
                },
            )
        db_path = cortrix_data_dir / "tasks.db"
        append_stage_event(
            stage_log,
            "doc_summary_drain",
            "start",
            {
                "profile": profile.name,
                "dataset": spec.name,
                "namespace": namespace,
                "task_type": doc_summary_task_type,
                "expected_total": imported_docs,
                "db_path": str(db_path),
            },
        )
        try:
            doc_summary_drain = poll_sqlite_task_type_with_diagnostics(
                db_path=db_path,
                namespace=namespace,
                task_type=doc_summary_task_type,
                expected_total=imported_docs,
                timeout_seconds=doc_summary_drain_timeout_seconds,
                interval_seconds=doc_summary_drain_interval_seconds,
                monitor_interval_seconds=task_monitor_interval_seconds,
                no_progress_seconds=doc_summary_no_progress_seconds,
                fail_on_no_progress=fail_on_no_progress,
                require_all_success=True,
                sample_size=task_sample_size,
                log_tail_lines=log_tail_lines,
                task_log=doc_summary_drain_log,
                resource_log=resource_log,
                resource_paths={
                    "sample_dir": sample_dir,
                    "cell_dir": cell_dir,
                    "run_dir": run_dir,
                    "dataset_cache": dataset_cache,
                    "cortrix_data_dir": cortrix_data_dir,
                },
                log_tail_dir=run_log_tail_dir,
                container_name=container_name,
                metrics_url=metrics_url,
                label=f"{cell_label}/doc_summary",
            )
        except TaskDrainDiagnosticError as exc:
            append_jsonl(cell_doc_summary_drain_log, exc.summary)
            append_stage_event(stage_log, "doc_summary_drain", "failed", {"profile": profile.name, "dataset": spec.name, "summary": exc.summary})
            raise
        append_jsonl(cell_doc_summary_drain_log, doc_summary_drain["summary"])
        append_stage_event(stage_log, "doc_summary_drain", "completed", {"profile": profile.name, "dataset": spec.name, "summary": doc_summary_drain["summary"]})
        append_jsonl(
            resource_log,
            snapshot(
                f"{profile.name}/{spec.name}/after_doc_summary_drain",
                {"sample_dir": sample_dir, "cell_dir": cell_dir, "cortrix_data_dir": cortrix_data_dir},
                container_name,
            ),
        )

    append_stage_event(stage_log, "index_ready", "start", {"profile": profile.name, "dataset": spec.name})
    try:
        readiness = client.readiness()
        write_json(cell_dir / "index_ready.json", {"readiness": dict(readiness)})
        append_stage_event(stage_log, "index_ready", "completed", {"profile": profile.name, "dataset": spec.name, "readiness": dict(readiness)})
    except CortrixClientError as exc:
        write_json(cell_dir / "index_ready.json", {"error": str(exc)})
        append_stage_event(stage_log, "index_ready", "warning", {"profile": profile.name, "dataset": spec.name, "error": str(exc)})

    queries = load_sampled_queries(sample.sampled_queries_path)
    qrels = _load_sampled_qrels(sample.sampled_qrels_path)
    child_id_aliases = _build_child_id_aliases(cortrix_data_dir, namespace)
    write_json(
        cell_dir / "child_id_aliases_summary.json",
        {
            "alias_count": len(child_id_aliases),
            "source": "runtime_store_db" if child_id_aliases else "not_available",
            "namespace": namespace,
        },
    )
    runs: Dict[str, List[str]] = {}
    # Full deduplicated document list per query (bounded by retrieval_k). Used
    # for the pool-survival recall@retrieval_k readout: a relevant document
    # displaced from the candidate pool disappears here too, while one that is
    # merely ranked below top_k survives. (RD isolation-matrix requirement.)
    runs_pool: Dict[str, List[str]] = {}
    # One failed query must not abort the whole cell (LLM arms can time out or
    # 5xx transiently). Failed queries score as empty runs and are counted.
    query_failures: List[Dict[str, str]] = []
    cold_latencies: List[float] = []
    latency_rows: List[Mapping[str, object]] = []
    query_log = cell_dir / "queries.jsonl"
    # BEIR metrics are DOCUMENT-level, but Cortrix returns chunk/child-level hits — one
    # document can surface as several chunks. Collapsing them to unique documents (best
    # rank kept) is required, else a single relevant doc's chunks fill multiple ranks:
    # DCG climbs above IDCG (nDCG > 1) and the duplicates crowd real docs out of recall@k.
    # Oversample retrieval so top_k UNIQUE docs survive the collapse.
    retrieval_k = max(top_k, top_k * 5)

    def _dedup_docs(ids: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for doc_id in ids:
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)
        return out

    append_stage_event(stage_log, "cold_query", "start", {"profile": profile.name, "dataset": spec.name, "queries": len(queries)})
    with query_log.open("w", encoding="utf-8") as out:
        for query_index, (query_id, query_text) in enumerate(queries.items()):
            _pace_query(query_index, query_cooldown_seconds)
            body = profile.query_body(query=query_text, namespace=namespace, top_k=retrieval_k)
            metrics_before = collect_query_latency_metrics(metrics_url)
            try:
                response, latency_ms, query_attempts = _query_with_transient_retries(
                    client,
                    body,
                    query_transient_retries,
                    query_transient_retry_backoff_seconds,
                )
            except Exception as exc:
                query_failures.append({"query_id": query_id, "error": repr(exc)})
                runs[query_id] = []
                runs_pool[query_id] = []
                out.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "query_error": repr(exc),
                            "query_attempts": getattr(exc, "attempts", []),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            metrics_after = collect_query_latency_metrics(metrics_url)
            query_latency_delta = diff_query_latency_metric_snapshots(metrics_before, metrics_after)
            derived_latency = query_latency_delta.get("derived") if isinstance(query_latency_delta.get("derived"), dict) else {}
            doc_ids_pool = _dedup_docs(extract_result_doc_ids(response, child_id_aliases=child_id_aliases))
            doc_ids = doc_ids_pool[:top_k]
            runs[query_id] = doc_ids
            runs_pool[query_id] = doc_ids_pool
            cold_latencies.append(latency_ms)
            latency_rows.append(_latency_row("cold", query_id, latency_ms, len(doc_ids), query_latency_delta))
            out.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "latency_ms": latency_ms,
                        "retrieved_doc_ids": doc_ids,
                        "retrieved_doc_ids_pool": doc_ids_pool,
                        "request_body": body,
                        "query_attempts": query_attempts,
                        "response": response,
                        "query_latency_metrics_delta": query_latency_delta,
                        "latency_breakdown": {
                            "http_wall_ms": latency_ms,
                            "retrieval_calls": derived_latency.get("retrieval_calls"),
                            "retrieval_wall_seconds": derived_latency.get("retrieval_wall_seconds"),
                            "rerank_calls": derived_latency.get("rerank_calls"),
                            "rerank_wall_seconds": derived_latency.get("rerank_wall_seconds"),
                            "rag_llm_wall_seconds": derived_latency.get("llm_call_wall_time_seconds"),
                            "rag_llm_calls": derived_latency.get("llm_call_count"),
                            "query_variants": derived_latency.get("query_variants"),
                            "rrf_fusion_seconds": derived_latency.get("rrf_fusion_seconds"),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    append_stage_event(stage_log, "cold_query", "completed", {"profile": profile.name, "dataset": spec.name, "queries": len(queries)})

    for repeat in range(max(0, warm_query_repeats)):
        warm_log = cell_dir / f"warm_queries_{repeat + 1}.jsonl"
        append_stage_event(stage_log, "warm_query", "start", {"profile": profile.name, "dataset": spec.name, "repeat": repeat + 1, "queries": len(queries)})
        with warm_log.open("w", encoding="utf-8") as out:
            for query_index, (query_id, query_text) in enumerate(queries.items()):
                _pace_query(query_index, query_cooldown_seconds)
                body = profile.query_body(query=query_text, namespace=namespace, top_k=retrieval_k)
                metrics_before = collect_query_latency_metrics(metrics_url)
                try:
                    response, latency_ms, query_attempts = _query_with_transient_retries(
                        client,
                        body,
                        query_transient_retries,
                        query_transient_retry_backoff_seconds,
                    )
                except Exception as exc:
                    out.write(
                        json.dumps(
                            {
                                "query_id": query_id,
                                "query_error": repr(exc),
                                "query_attempts": getattr(exc, "attempts", []),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                metrics_after = collect_query_latency_metrics(metrics_url)
                query_latency_delta = diff_query_latency_metric_snapshots(metrics_before, metrics_after)
                derived_latency = query_latency_delta.get("derived") if isinstance(query_latency_delta.get("derived"), dict) else {}
                doc_ids = _dedup_docs(extract_result_doc_ids(response, child_id_aliases=child_id_aliases))[:top_k]
                latency_rows.append(_latency_row(f"warm_{repeat + 1}", query_id, latency_ms, len(doc_ids), query_latency_delta))
                out.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "latency_ms": latency_ms,
                            "retrieved_doc_ids": doc_ids,
                            "request_body": body,
                            "query_attempts": query_attempts,
                            "response": response,
                            "query_latency_metrics_delta": query_latency_delta,
                            "latency_breakdown": {
                                "http_wall_ms": latency_ms,
                                "retrieval_calls": derived_latency.get("retrieval_calls"),
                                "retrieval_wall_seconds": derived_latency.get("retrieval_wall_seconds"),
                                "rerank_calls": derived_latency.get("rerank_calls"),
                                "rerank_wall_seconds": derived_latency.get("rerank_wall_seconds"),
                                "rag_llm_wall_seconds": derived_latency.get("llm_call_wall_time_seconds"),
                                "rag_llm_calls": derived_latency.get("llm_call_count"),
                                "query_variants": derived_latency.get("query_variants"),
                                "rrf_fusion_seconds": derived_latency.get("rrf_fusion_seconds"),
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        append_stage_event(stage_log, "warm_query", "completed", {"profile": profile.name, "dataset": spec.name, "repeat": repeat + 1, "queries": len(queries)})
    append_jsonl(resource_log, snapshot(f"{profile.name}/{spec.name}/after_queries", {"cell_dir": cell_dir}, container_name))

    append_stage_event(stage_log, "metric_compute", "start", {"profile": profile.name, "dataset": spec.name})
    metric_summary = aggregate(qrels, runs, cold_latencies, k=top_k)
    if retrieval_k > top_k:
        # Pool-survival recall: relevant docs anywhere in the deduplicated
        # response (up to retrieval_k), regardless of top_k rank.
        pool_recall = recall_at_k(qrels, runs_pool, k=retrieval_k)
        metric_summary[f"recall@{retrieval_k}"] = summarize(pool_recall.values())
    metric_summary["query_failures"] = len(query_failures)
    if query_failures:
        metric_summary["query_failure_ids"] = [row["query_id"] for row in query_failures]
    write_json(cell_dir / "metric_summary.json", metric_summary)
    write_json(cell_dir / "metrics.json", metric_summary)
    write_latency_csv(cell_dir / "latency.csv", latency_rows)
    append_stage_event(stage_log, "metric_compute", "completed", {"profile": profile.name, "dataset": spec.name, "metrics": metric_summary})
    capture_log_tail(cell_log_tail_dir, f"{cell_label}-after-query", container_name, log_tail_lines)
    artifact_manifest = file_manifest(
        [
            cell_dir / "sample_manifest.json",
            sample.sampled_corpus_path,
            sample.sampled_queries_path,
            sample.sampled_qrels_path,
            sample.manifest_path,
            cell_import_batch_log,
            cell_task_drain_log,
            cell_doc_summary_drain_log,
            query_log,
            cell_dir / "metric_summary.json",
            cell_dir / "metrics.json",
            cell_dir / "latency.csv",
        ]
    )
    result = {
        "status": "completed",
        "diagnosis": "completed_diagnostic_flow",
        "dataset": spec.name,
        "profile": profile.name,
        "namespace": namespace,
        "sample": sample.to_json(),
        "imported_documents": imported_docs,
        "metrics": metric_summary,
        "doc_summary_drain": doc_summary_drain["summary"] if doc_summary_drain else None,
        "artifact_manifest": artifact_manifest,
        "dataset_root": str(dataset_root),
        "dataset_bytes_on_disk": file_size_bytes(dataset_root),
    }
    write_json(cell_dir / "cell_result.json", result)
    print(f"[cell] done {profile.name}/{spec.name}")
    return result


def _latency_row(
    phase: str,
    query_id: str,
    latency_ms: float,
    retrieved_count: int,
    query_latency_delta: Mapping[str, object],
) -> Mapping[str, object]:
    derived = query_latency_delta.get("derived") if isinstance(query_latency_delta.get("derived"), dict) else {}
    return {
        "phase": phase,
        "query_id": query_id,
        "latency_ms": latency_ms,
        "retrieved_count": retrieved_count,
        "metrics_status": query_latency_delta.get("status"),
        "retrieval_calls": derived.get("retrieval_calls"),
        "retrieval_wall_seconds": derived.get("retrieval_wall_seconds"),
        "rerank_calls": derived.get("rerank_calls"),
        "rerank_wall_seconds": derived.get("rerank_wall_seconds"),
        "rag_llm_wall_seconds": derived.get("llm_call_wall_time_seconds"),
        "rag_llm_calls": derived.get("llm_call_count"),
        "query_variants": derived.get("query_variants"),
        "rrf_fusion_seconds": derived.get("rrf_fusion_seconds"),
    }


def _corpus_row_to_document(dataset: str, row: Mapping[str, str]) -> Mapping[str, object]:
    doc_id = str(row.get("_id", ""))
    title = str(row.get("title", "") or "")
    text = str(row.get("text", "") or "")
    content = f"{title}\n\n{text}" if title else text
    return {
        "doc_id": doc_id,
        "filename": f"{dataset}-{_safe_filename_component(doc_id)}.txt",
        "content": content,
        "metadata": {
            "beir_dataset": dataset,
            "beir_corpus_id": doc_id,
            "beir_title": title,
        },
    }


def _build_child_id_aliases(cortrix_data_dir: Path | None, namespace: str) -> Dict[str, str]:
    if cortrix_data_dir is None:
        return {}
    store_db = cortrix_data_dir / "units" / f"unit-{namespace}" / "store.db"
    if not store_db.exists():
        return {}
    aliases: Dict[str, str] = {}
    uri = f"file:{store_db}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT b.child_id, d.doc_id, d.metadata_json, d.source_path
                FROM blocks b
                JOIN documents d ON b.doc_id = d.doc_id
                WHERE b.child_id IS NOT NULL
                """
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # store.db exists but has no schema yet (namespace created, zero
            # blocks ever written — i.e. the ingest never landed anything).
            # Aliases are best-effort; the ingest-wipeout guard upstream is
            # responsible for failing the cell in that case.
            print(f"[warn] child-id alias build skipped ({store_db}): {exc}")
            return {}
    for row in rows:
        child_id = str(row["child_id"] or "")
        stable_id = _stable_doc_id_from_document_row(row)
        if child_id and stable_id:
            aliases[child_id] = stable_id
            aliases.setdefault(stable_id, stable_id)
    return aliases


def _stable_doc_id_from_document_row(row: Mapping[str, object]) -> str:
    metadata_json = _row_value(row, "metadata_json")
    if isinstance(metadata_json, str) and metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata = {}
        if isinstance(metadata, dict):
            for key in ["beir_corpus_id", "beir_doc_id", "corpus_id", "corpus_doc_id", "original_doc_id", "original_id"]:
                value = metadata.get(key)
                if isinstance(value, str) and value and not _looks_internal_id(value):
                    return value
    doc_id = _row_value(row, "doc_id")
    if isinstance(doc_id, str) and doc_id and not _looks_internal_id(doc_id):
        return doc_id
    source_path = _row_value(row, "source_path")
    if isinstance(source_path, str) and source_path:
        name = source_path.rsplit("/", 1)[-1]
        if name.endswith(".txt"):
            name = name[:-4]
        doc_id = _strip_known_dataset_prefix(name)
        if doc_id:
            return doc_id
    return ""


def _strip_known_dataset_prefix(name: str) -> str:
    for dataset in sorted(
        {"scifact", "nfcorpus", "fiqa", "fiqa-mini-120", "fiqa-mini-600", "hotpotqa", "webis_touche2020", "webis-touche2020"},
        key=len,
        reverse=True,
    ):
        prefix = f"{dataset}-"
        if name.startswith(prefix):
            return name[len(prefix) :]
    return ""


def _row_value(row: Mapping[str, object], key: str) -> object:
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _looks_internal_id(value: str) -> bool:
    if len(value) != 26:
        return False
    return all(ch.isdigit() or ("A" <= ch <= "Z" and ch not in {"I", "L", "O", "U"}) for ch in value)


def _load_sampled_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no == 1 and line.startswith("query-id\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qrels.setdefault(parts[0], {})[parts[1]] = float(parts[2])
    return qrels


def _batched(items: Iterable[Mapping[str, str]], size: int) -> Iterable[List[Mapping[str, str]]]:
    batch: List[Mapping[str, str]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _safe_filename_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe[:160] or "doc"


def _namespace(run_id: str, profile: str, dataset: str) -> str:
    safe = f"beir-{profile}-{dataset}-{run_id}".lower()
    safe = "".join(ch if ch.isalnum() else "-" for ch in safe)
    return safe[:63].strip("-")


def _parse_names(raw: str, known: Mapping[str, object], kind: str) -> List[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SystemExit(f"unknown {kind}: {', '.join(unknown)}")
    return names


if __name__ == "__main__":
    raise SystemExit(main())
