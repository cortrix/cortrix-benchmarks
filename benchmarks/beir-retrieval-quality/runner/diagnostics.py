from __future__ import annotations

import csv
import json
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from cortrix_client import CortrixClient, CortrixClientError
from resource_probe import append_jsonl, run_cmd, snapshot


TERMINAL_STATUSES = {"completed", "complete", "ready", "failed", "error", "cancelled", "canceled", "timeout"}
SUCCESS_STATUSES = {"completed", "complete", "ready"}
FAILED_STATUSES = {"failed", "error", "timeout"}
LOG_KEYWORDS = ["panic", "SIGSEGV", "OOM", "error", "failed", "timeout", "retry", "queue", "worker", "task"]
METRICS_PATTERNS = [
    "cortrix_tasks_submitted_total",
    "cortrix_tasks_completed_total",
    "cortrix_tasks_queue_depth",
    "cortrix_tasks_zombie_cleaned_total",
    "cortrix_f16a_tasks_queue_depth",
    "cortrix_doc_summary_llm_calls_total",
    "cortrix_doc_summary_summaries_generated_total",
    "cortrix_doc_summary_llm_duration_seconds",
    "cortrix_doc_summary_fallback_triggered_total",
    "cortrix_fts5_fallback_failed_total",
]


class TaskDrainDiagnosticError(CortrixClientError):
    def __init__(self, message: str, summary: Mapping[str, object]) -> None:
        super().__init__(message)
        self.summary = dict(summary)


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_stage_event(path: Path, stage: str, status: str, extra: Optional[Mapping[str, object]] = None) -> None:
    item: Dict[str, object] = {"ts": utc_ts(), "stage": stage, "status": status}
    if extra:
        item.update(extra)
    append_jsonl(path, item)


def poll_tasks_with_diagnostics(
    client: CortrixClient,
    task_ids: Sequence[str],
    *,
    timeout_seconds: float,
    interval_seconds: float,
    monitor_interval_seconds: float,
    no_progress_seconds: float,
    fail_on_no_progress: bool,
    poll_batch_size: int,
    sample_size: int,
    log_tail_lines: int,
    task_log: Path,
    resource_log: Path,
    resource_paths: Mapping[str, Path],
    log_tail_dir: Path,
    container_name: str,
    metrics_url: str,
    label: str,
    log: Callable[[str], None] = print,
) -> Mapping[str, object]:
    total = len(task_ids)
    if total == 0:
        summary = _task_summary(total=0, status_by_task={}, start_time=time.time(), last_progress_time=time.time())
        append_jsonl(task_log, {"ts": utc_ts(), "label": label, "event": "no_tasks", **summary})
        return summary

    status_by_task: Dict[str, Mapping[str, object]] = {}
    pending = list(dict.fromkeys(task_ids))
    final: Dict[str, Mapping[str, object]] = {}
    cursor = 0
    start_time = time.time()
    deadline = start_time + timeout_seconds
    last_snapshot_time = 0.0
    last_progress_time = start_time
    last_terminal_count = 0
    no_progress_dumped = False
    poll_batch_size = max(1, poll_batch_size)

    while pending:
        now = time.time()
        if now > deadline:
            summary = _task_summary(total=total, status_by_task=status_by_task, start_time=start_time, last_progress_time=last_progress_time)
            summary["diagnosis"] = _diagnosis_for_timeout(summary, no_progress_seconds)
            _write_monitor_snapshot(
                task_log=task_log,
                resource_log=resource_log,
                resource_paths=resource_paths,
                log_tail_dir=log_tail_dir,
                container_name=container_name,
                metrics_url=metrics_url,
                label=label,
                event="timeout",
                summary=summary,
                status_by_task=status_by_task,
                sample_size=sample_size,
                log_tail_lines=log_tail_lines,
                capture_logs=True,
            )
            raise TaskDrainDiagnosticError(f"timed out waiting for {summary['pending']} of {total} tasks", summary)

        if cursor >= len(pending):
            cursor = 0
        batch = pending[cursor : cursor + poll_batch_size]
        if not batch:
            batch = pending[:poll_batch_size]
            cursor = 0
        terminal_ids: set[str] = set()
        for task_id in batch:
            try:
                progress = client.task_progress(task_id)
            except CortrixClientError as exc:
                progress = {"task_id": task_id, "status": "poll_error", "error": str(exc)}
            status_by_task[task_id] = progress
            status = _status(progress)
            if status in TERMINAL_STATUSES:
                final[task_id] = progress
                terminal_ids.add(task_id)

        if terminal_ids:
            pending = [task_id for task_id in pending if task_id not in terminal_ids]
            if cursor >= len(pending):
                cursor = 0
        else:
            cursor += len(batch)

        terminal_count = _terminal_count(status_by_task)
        if terminal_count > last_terminal_count:
            last_terminal_count = terminal_count
            last_progress_time = time.time()

        now = time.time()
        should_snapshot = last_snapshot_time == 0.0 or now - last_snapshot_time >= monitor_interval_seconds
        no_progress_duration = now - last_progress_time
        should_dump_no_progress = no_progress_duration >= no_progress_seconds and not no_progress_dumped
        if should_snapshot or should_dump_no_progress:
            summary = _task_summary(total=total, status_by_task=status_by_task, start_time=start_time, last_progress_time=last_progress_time)
            summary["diagnosis"] = _diagnosis_for_timeout(summary, no_progress_seconds) if should_dump_no_progress else "monitoring"
            _write_monitor_snapshot(
                task_log=task_log,
                resource_log=resource_log,
                resource_paths=resource_paths,
                log_tail_dir=log_tail_dir,
                container_name=container_name,
                metrics_url=metrics_url,
                label=label,
                event="no_progress_checkpoint" if should_dump_no_progress else "monitor",
                summary=summary,
                status_by_task=status_by_task,
                sample_size=sample_size,
                log_tail_lines=log_tail_lines,
                capture_logs=should_dump_no_progress,
            )
            last_snapshot_time = now
            if should_dump_no_progress:
                no_progress_dumped = True
                log(f"[diagnostic] {label}: no task completion progress for {int(no_progress_duration)}s")
                if fail_on_no_progress:
                    raise TaskDrainDiagnosticError(
                        f"no task completion progress for {int(no_progress_duration)}s",
                        summary,
                    )

        if pending:
            time.sleep(max(0.1, interval_seconds))

    summary = _task_summary(total=total, status_by_task=status_by_task, start_time=start_time, last_progress_time=last_progress_time)
    summary["diagnosis"] = "completed_task_drain"
    _write_monitor_snapshot(
        task_log=task_log,
        resource_log=resource_log,
        resource_paths=resource_paths,
        log_tail_dir=log_tail_dir,
        container_name=container_name,
        metrics_url=metrics_url,
        label=label,
        event="completed",
        summary=summary,
        status_by_task=status_by_task,
        sample_size=sample_size,
        log_tail_lines=log_tail_lines,
        capture_logs=False,
    )
    return {"summary": summary, "final": final}


def poll_sqlite_task_type_with_diagnostics(
    *,
    db_path: Path,
    namespace: str,
    task_type: int,
    expected_total: int,
    timeout_seconds: float,
    interval_seconds: float,
    monitor_interval_seconds: float,
    no_progress_seconds: float,
    fail_on_no_progress: bool,
    require_all_success: bool,
    sample_size: int,
    log_tail_lines: int,
    task_log: Path,
    resource_log: Path,
    resource_paths: Mapping[str, Path],
    log_tail_dir: Path,
    container_name: str,
    metrics_url: str,
    label: str,
    log: Callable[[str], None] = print,
) -> Mapping[str, object]:
    """Poll a local Cortrix tasks.db aggregate until a task type is drained.

    This is a benchmark harness diagnostic gate, not a Cortrix public API. It is
    used when a downstream async task type, such as kTaskDocSummary, is not
    returned by the batch-submit API and therefore cannot be waited on by task ID.
    """
    expected_total = max(0, expected_total)
    start_time = time.time()
    deadline = start_time + timeout_seconds
    last_snapshot_time = 0.0
    last_progress_time = start_time
    last_terminal_count = -1
    no_progress_dumped = False

    while True:
        now = time.time()
        try:
            summary, samples = _sqlite_task_type_snapshot(
                db_path=db_path,
                namespace=namespace,
                task_type=task_type,
                expected_total=expected_total,
                start_time=start_time,
                last_progress_time=last_progress_time,
                sample_size=sample_size,
            )
            query_error = None
        except sqlite3.Error as exc:
            summary = _sqlite_unavailable_summary(
                db_path=db_path,
                namespace=namespace,
                task_type=task_type,
                expected_total=expected_total,
                start_time=start_time,
                last_progress_time=last_progress_time,
                error=repr(exc),
            )
            samples = {}
            query_error = repr(exc)

        terminal_count = int(summary.get("terminal", 0) or 0)
        if terminal_count > last_terminal_count:
            last_terminal_count = terminal_count
            last_progress_time = time.time()
            summary["last_progress_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_progress_time))

        row_total = int(summary.get("row_total", 0) or 0)
        completed = int(summary.get("completed", 0) or 0)
        nonterminal = int(summary.get("nonterminal", 0) or 0)
        failed = int(summary.get("failed", 0) or 0)
        cancelled = int(summary.get("cancelled", 0) or 0)
        success_ready = (
            row_total >= expected_total
            and completed >= expected_total
            and nonterminal == 0
            and failed == 0
            and cancelled == 0
        )
        terminal_ready = (
            row_total >= expected_total
            and terminal_count >= expected_total
            and nonterminal == 0
        )
        if success_ready or (terminal_ready and not require_all_success):
            summary["diagnosis"] = "completed_sqlite_task_type_drain"
            _write_monitor_snapshot(
                task_log=task_log,
                resource_log=resource_log,
                resource_paths=resource_paths,
                log_tail_dir=log_tail_dir,
                container_name=container_name,
                metrics_url=metrics_url,
                label=label,
                event="completed",
                summary=summary,
                status_by_task=samples,
                sample_size=sample_size,
                log_tail_lines=log_tail_lines,
                capture_logs=False,
            )
            return {"summary": summary, "samples": samples}

        if require_all_success and (failed > 0 or cancelled > 0):
            summary["diagnosis"] = "sqlite_task_type_failed"
            _write_monitor_snapshot(
                task_log=task_log,
                resource_log=resource_log,
                resource_paths=resource_paths,
                log_tail_dir=log_tail_dir,
                container_name=container_name,
                metrics_url=metrics_url,
                label=label,
                event="failed",
                summary=summary,
                status_by_task=samples,
                sample_size=sample_size,
                log_tail_lines=log_tail_lines,
                capture_logs=True,
            )
            raise TaskDrainDiagnosticError(
                f"{label}: task_type={task_type} has failed/cancelled tasks",
                summary,
            )

        if now > deadline:
            summary["diagnosis"] = "sqlite_task_type_timeout"
            _write_monitor_snapshot(
                task_log=task_log,
                resource_log=resource_log,
                resource_paths=resource_paths,
                log_tail_dir=log_tail_dir,
                container_name=container_name,
                metrics_url=metrics_url,
                label=label,
                event="timeout",
                summary=summary,
                status_by_task=samples,
                sample_size=sample_size,
                log_tail_lines=log_tail_lines,
                capture_logs=True,
            )
            raise TaskDrainDiagnosticError(
                f"{label}: timed out waiting for task_type={task_type} aggregate drain",
                summary,
            )

        now = time.time()
        should_snapshot = last_snapshot_time == 0.0 or now - last_snapshot_time >= monitor_interval_seconds
        no_progress_duration = now - last_progress_time
        should_dump_no_progress = no_progress_duration >= no_progress_seconds and not no_progress_dumped
        if should_snapshot or should_dump_no_progress:
            summary["diagnosis"] = "sqlite_task_type_query_error" if query_error else "monitoring"
            _write_monitor_snapshot(
                task_log=task_log,
                resource_log=resource_log,
                resource_paths=resource_paths,
                log_tail_dir=log_tail_dir,
                container_name=container_name,
                metrics_url=metrics_url,
                label=label,
                event="no_progress_checkpoint" if should_dump_no_progress else "monitor",
                summary=summary,
                status_by_task=samples,
                sample_size=sample_size,
                log_tail_lines=log_tail_lines,
                capture_logs=should_dump_no_progress,
            )
            last_snapshot_time = now
            if should_dump_no_progress:
                no_progress_dumped = True
                log(f"[diagnostic] {label}: no aggregate task progress for {int(no_progress_duration)}s")
                if fail_on_no_progress:
                    raise TaskDrainDiagnosticError(
                        f"{label}: no aggregate task progress for {int(no_progress_duration)}s",
                        summary,
                    )

        time.sleep(max(0.5, interval_seconds))


def capture_log_tail(log_tail_dir: Path, label: str, container_name: str, lines: int) -> Mapping[str, object]:
    log_tail_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_label(label)
    log_path = log_tail_dir / f"{stem}.log"
    summary_path = log_tail_dir / f"{stem}.json"
    result = run_cmd(["docker", "logs", "--tail", str(lines), container_name], timeout_seconds=30.0)
    text = _redact_sensitive(f"{result.get('stdout', '')}{result.get('stderr', '')}")
    log_path.write_text(text, encoding="utf-8")
    matches = _scan_log_keywords(text)
    summary = {
        "ts": utc_ts(),
        "label": label,
        "container": container_name,
        "tail_lines": lines,
        "returncode": result.get("returncode"),
        "log_path": str(log_path),
        "keyword_matches": matches,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def fetch_metrics_snapshot(metrics_url: str, timeout_seconds: float = 5.0) -> Mapping[str, object]:
    if not metrics_url:
        return {"available": False, "reason": "metrics_url_not_configured"}
    try:
        with urllib.request.urlopen(metrics_url, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"available": False, "error": repr(exc), "url": metrics_url}
    lines = [
        line
        for line in payload.splitlines()
        if line and not line.startswith("#") and any(pattern in line for pattern in METRICS_PATTERNS)
    ]
    return {"available": True, "url": metrics_url, "lines": lines[:200]}


def write_latency_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "query_id",
        "latency_ms",
        "retrieved_count",
        "metrics_status",
        "retrieval_calls",
        "retrieval_wall_seconds",
        "rerank_calls",
        "rerank_wall_seconds",
        "rag_llm_wall_seconds",
        "rag_llm_calls",
        "query_variants",
        "rrf_fusion_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_monitor_snapshot(
    *,
    task_log: Path,
    resource_log: Path,
    resource_paths: Mapping[str, Path],
    log_tail_dir: Path,
    container_name: str,
    metrics_url: str,
    label: str,
    event: str,
    summary: Mapping[str, object],
    status_by_task: Mapping[str, Mapping[str, object]],
    sample_size: int,
    log_tail_lines: int,
    capture_logs: bool,
) -> None:
    item = {
        "ts": utc_ts(),
        "label": label,
        "event": event,
        **summary,
        "status_samples": _status_samples(status_by_task, sample_size),
        "metrics_snapshot": fetch_metrics_snapshot(metrics_url),
    }
    if capture_logs:
        item["log_tail_snapshot"] = capture_log_tail(log_tail_dir, f"{label}-{event}", container_name, log_tail_lines)
    append_jsonl(task_log, item)
    append_jsonl(resource_log, snapshot(f"{label}/task_drain/{event}", resource_paths, container_name))


def _sqlite_task_type_snapshot(
    *,
    db_path: Path,
    namespace: str,
    task_type: int,
    expected_total: int,
    start_time: float,
    last_progress_time: float,
    sample_size: int,
) -> tuple[Dict[str, object], Dict[str, Mapping[str, object]]]:
    if not db_path.exists():
        raise sqlite3.OperationalError(f"tasks db not found: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT status, COALESCE(error_code, '') AS error_code, COUNT(*) AS n
            FROM tasks
            WHERE namespace_id = ? AND task_type = ?
            GROUP BY status, COALESCE(error_code, '')
            ORDER BY status, error_code
            """,
            (namespace, task_type),
        ).fetchall()
        samples = conn.execute(
            """
            SELECT task_id, doc_id, filename, status, error_code, error_msg,
                   structured_data, created_at, updated_at, started_at, completed_at
            FROM tasks
            WHERE namespace_id = ? AND task_type = ? AND status != 'completed'
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (namespace, task_type, max(0, sample_size)),
        ).fetchall()

    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    status_error_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row["status"] or "unknown").lower()
        error_code = str(row["error_code"] or "")
        n = int(row["n"])
        status_counts[status] += n
        if error_code:
            error_counts[error_code] += n
            status_error_counts[f"{status}:{error_code}"] = n

    completed = sum(status_counts[status] for status in SUCCESS_STATUSES)
    failed = sum(status_counts[status] for status in FAILED_STATUSES)
    cancelled = status_counts["cancelled"] + status_counts["canceled"]
    terminal = completed + failed + cancelled
    row_total = sum(status_counts.values())
    nonterminal = max(0, row_total - terminal)
    now = time.time()
    summary: Dict[str, object] = {
        "db_path": str(db_path),
        "namespace": namespace,
        "task_type": task_type,
        "expected_total": expected_total,
        "row_total": row_total,
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "terminal": terminal,
        "nonterminal": nonterminal,
        "pending": max(0, expected_total - terminal),
        "status_counts": dict(status_counts),
        "error_counts": dict(error_counts),
        "status_error_counts": status_error_counts,
        "elapsed_seconds": round(now - start_time, 3),
        "last_progress_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_progress_time)),
        "no_progress_seconds": round(now - last_progress_time, 3),
        "api_observability_limit": (
            "This aggregate comes from local read-only tasks.db access in the benchmark "
            "harness because batch-submit API task IDs do not include downstream task types."
        ),
    }
    sample_map: Dict[str, Mapping[str, object]] = {}
    for row in samples:
        task_id = str(row["task_id"])
        sample_map[task_id] = {
            "task_id": task_id,
            "doc_id": row["doc_id"],
            "filename": row["filename"],
            "status": row["status"],
            "error_code": row["error_code"],
            "error_msg": _truncate_nullable(row["error_msg"], 800),
            "structured_data_preview": _truncate_nullable(row["structured_data"], 1200),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }
    return summary, sample_map


def _sqlite_unavailable_summary(
    *,
    db_path: Path,
    namespace: str,
    task_type: int,
    expected_total: int,
    start_time: float,
    last_progress_time: float,
    error: str,
) -> Dict[str, object]:
    now = time.time()
    return {
        "db_path": str(db_path),
        "namespace": namespace,
        "task_type": task_type,
        "expected_total": expected_total,
        "row_total": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "terminal": 0,
        "nonterminal": 0,
        "pending": expected_total,
        "status_counts": {},
        "error_counts": {},
        "status_error_counts": {},
        "elapsed_seconds": round(now - start_time, 3),
        "last_progress_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_progress_time)),
        "no_progress_seconds": round(now - last_progress_time, 3),
        "sqlite_error": error,
        "diagnosis": "sqlite_task_type_query_error",
    }


def _truncate_nullable(value: object, max_bytes: int) -> object:
    if value is None:
        return None
    text = str(value)
    if len(text.encode("utf-8", errors="ignore")) <= max_bytes:
        return text
    return text[:max_bytes] + "...[truncated]"


def _task_summary(
    *,
    total: int,
    status_by_task: Mapping[str, Mapping[str, object]],
    start_time: float,
    last_progress_time: float,
) -> Dict[str, object]:
    counts = Counter(_status(progress) for progress in status_by_task.values())
    completed = sum(counts[status] for status in SUCCESS_STATUSES)
    failed = sum(counts[status] for status in FAILED_STATUSES)
    cancelled = counts["cancelled"] + counts["canceled"]
    terminal = completed + failed + cancelled
    known = len(status_by_task)
    now = time.time()
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "pending": max(0, total - terminal),
        "known_nonterminal": max(0, known - terminal),
        "unknown": max(0, total - known),
        "status_counts": dict(counts),
        "elapsed_seconds": round(now - start_time, 3),
        "last_progress_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_progress_time)),
        "no_progress_seconds": round(now - last_progress_time, 3),
        "api_observability_limit": (
            "No aggregate document task status endpoint was found; counts combine exact "
            "terminal statuses for polled task IDs with unknown tasks that have not yet been sampled."
        ),
    }


def _diagnosis_for_timeout(summary: Mapping[str, object], no_progress_seconds: float) -> str:
    completed = int(summary.get("completed", 0) or 0)
    pending = int(summary.get("pending", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    no_progress = float(summary.get("no_progress_seconds", 0.0) or 0.0)
    if failed > 0:
        return "server_error"
    if pending > 0 and no_progress >= no_progress_seconds:
        return "task_drain_no_terminal_progress" if completed == 0 else "slow_task_drain_progressing"
    if pending > 0:
        return "slow_task_drain_progressing"
    return "completed_task_drain"


def _terminal_count(status_by_task: Mapping[str, Mapping[str, object]]) -> int:
    return sum(1 for progress in status_by_task.values() if _status(progress) in TERMINAL_STATUSES)


def _status(progress: Mapping[str, object]) -> str:
    status = progress.get("status")
    return str(status).lower() if status is not None else "unknown"


def _status_samples(status_by_task: Mapping[str, Mapping[str, object]], sample_size: int) -> List[Mapping[str, object]]:
    rows: List[Mapping[str, object]] = []
    for task_id in sorted(status_by_task)[: max(0, sample_size)]:
        progress = status_by_task[task_id]
        rows.append(
            {
                "task_id": task_id,
                "status": progress.get("status", "unknown"),
                "progress": progress.get("progress", progress.get("progress_pct")),
                "error": progress.get("error"),
                "error_code": progress.get("error_code"),
                "error_msg": progress.get("error_msg"),
                "structured_data_preview": progress.get("structured_data_preview"),
            }
        )
    return rows


def _scan_log_keywords(text: str) -> Dict[str, List[str]]:
    matches: Dict[str, List[str]] = {}
    for keyword in LOG_KEYWORDS:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        rows = []
        for line in text.splitlines():
            if pattern.search(line):
                rows.append(line[:500])
            if len(rows) >= 20:
                break
        if rows:
            matches[keyword] = rows
    return matches


def _redact_sensitive(text: str) -> str:
    redacted = re.sub(r"ctx_bootstrap_[A-Za-z0-9]+", "ctx_bootstrap_[REDACTED]", text)
    redacted = re.sub(r"sk-[A-Za-z0-9_-]{16,}", "sk-[REDACTED]", redacted)
    return redacted


def _safe_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in label.lower())
    safe = re.sub("-+", "-", safe).strip("-")
    return safe[:180] or "snapshot"
