from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

RUNNER_DIR = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER_DIR))

import run_benchmark  # noqa: E402
from diagnostics import (  # noqa: E402
    TaskDrainDiagnosticError,
    poll_sqlite_task_type_with_diagnostics,
)


def _init_tasks_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                namespace_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                doc_id TEXT,
                content_hash TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                task_type INTEGER NOT NULL DEFAULT 1,
                error_code TEXT,
                error_msg TEXT,
                structured_data TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            )
            """
        )


def _insert_task(path: Path, task_id: str, status: str, error_code: str = "") -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO tasks (
                task_id, namespace_id, filename, filepath, doc_id, status, task_type,
                error_code, error_msg, structured_data, created_at, updated_at,
                started_at, completed_at
            )
            VALUES (?, 'ns-fiqa', ?, '/tmp/doc.txt', ?, ?, 3, ?, '', '{}',
                    '2026-06-28T00:00:00Z', '2026-06-28T00:00:01Z',
                    '2026-06-28T00:00:00Z', '2026-06-28T00:00:01Z')
            """,
            (task_id, f"{task_id}.txt", f"doc-{task_id}", status, error_code),
        )


def _poll(path: Path, tmp_path: Path, expected_total: int = 2):
    return poll_sqlite_task_type_with_diagnostics(
        db_path=path,
        namespace="ns-fiqa",
        task_type=3,
        expected_total=expected_total,
        timeout_seconds=0.1,
        interval_seconds=0.01,
        monitor_interval_seconds=0.01,
        no_progress_seconds=999,
        fail_on_no_progress=False,
        require_all_success=True,
        sample_size=10,
        log_tail_lines=1,
        task_log=tmp_path / "tasks.jsonl",
        resource_log=tmp_path / "resources.jsonl",
        resource_paths={"tmp": tmp_path},
        log_tail_dir=tmp_path / "logs",
        container_name="cortrix-test",
        metrics_url="",
        label="test/doc_summary",
        log=lambda _msg: None,
    )


def test_doc_summary_drain_completes_when_expected_rows_completed(tmp_path: Path) -> None:
    db = tmp_path / "tasks.db"
    _init_tasks_db(db)
    _insert_task(db, "t1", "completed")
    _insert_task(db, "t2", "completed")

    result = _poll(db, tmp_path)

    summary = result["summary"]
    assert summary["diagnosis"] == "completed_sqlite_task_type_drain"
    assert summary["completed"] == 2
    assert summary["failed"] == 0
    assert summary["pending"] == 0


def test_doc_summary_drain_rejects_observed_row_shortfall(tmp_path: Path) -> None:
    db = tmp_path / "tasks.db"
    _init_tasks_db(db)
    _insert_task(db, "t1", "completed")

    try:
        _poll(db, tmp_path, expected_total=2)
    except TaskDrainDiagnosticError as exc:
        assert exc.summary["diagnosis"] == "sqlite_task_type_timeout"
        assert exc.summary["completed"] == 1
        assert exc.summary["pending"] == 1
    else:
        raise AssertionError("expected observed-row shortfall to reject the drain gate")


def test_doc_summary_drain_rejects_large_observed_row_shortfall(tmp_path: Path) -> None:
    db = tmp_path / "tasks.db"
    _init_tasks_db(db)
    _insert_task(db, "t1", "completed")

    try:
        _poll(db, tmp_path, expected_total=200)
    except TaskDrainDiagnosticError as exc:
        assert exc.summary["diagnosis"] == "sqlite_task_type_timeout"
        assert exc.summary["completed"] == 1
        assert exc.summary["pending"] == 199
    else:
        raise AssertionError("expected large observed-row shortfall to reject the drain gate")


def test_doc_summary_drain_rejects_failed_rows(tmp_path: Path) -> None:
    db = tmp_path / "tasks.db"
    _init_tasks_db(db)
    _insert_task(db, "t1", "completed")
    _insert_task(db, "t2", "failed", "CX_ERR_F41_LLM_INVALID_OUTPUT")

    try:
        _poll(db, tmp_path)
    except TaskDrainDiagnosticError as exc:
        assert exc.summary["diagnosis"] == "sqlite_task_type_failed"
        assert exc.summary["error_counts"] == {"CX_ERR_F41_LLM_INVALID_OUTPUT": 1}
    else:
        raise AssertionError("expected failed doc-summary task to reject the drain gate")


def test_query_pacing_skips_first_query_and_default_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(run_benchmark.time, "sleep", sleeps.append)

    run_benchmark._pace_query(0, 30.0)
    run_benchmark._pace_query(1, 0.0)

    assert sleeps == []


def test_query_pacing_waits_between_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(run_benchmark.time, "sleep", sleeps.append)

    run_benchmark._pace_query(1, 30.0)
    run_benchmark._pace_query(2, 7.5)

    assert sleeps == [30.0, 7.5]


def test_query_pacing_rejects_negative_cli_value() -> None:
    with pytest.raises(SystemExit) as exc:
        run_benchmark.main(["run", "--query-cooldown-seconds", "-1"])

    assert exc.value.code == 2


class _SequenceClient:
    def __init__(self, responses: list[tuple[dict[str, object], float] | Exception]) -> None:
        self.responses = list(responses)

    def query(self, _body: object) -> tuple[dict[str, object], float]:
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _llm_rerank_response(degraded: bool, reason: str = "") -> dict[str, object]:
    return {
        "explain": {
            "llm_dependent_features": {
                "llm_rerank": {
                    "degraded": degraded,
                    "degrade_reason": reason,
                }
            }
        }
    }


def test_query_retry_recovers_transient_degrade(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(run_benchmark.time, "sleep", sleeps.append)
    client = _SequenceClient(
        [
            (_llm_rerank_response(True, "llm_transport"), 100.0),
            (_llm_rerank_response(False), 200.0),
        ]
    )

    response, latency_ms, attempts = run_benchmark._query_with_transient_retries(
        client, {}, max_retries=1, backoff_seconds=5.0
    )

    assert run_benchmark._transient_degrade_reason(response) == ""
    assert latency_ms == 5300.0
    assert sleeps == [5.0]
    assert [attempt["status"] for attempt in attempts] == ["transient_degrade", "success"]
    assert [attempt["backoff_seconds"] for attempt in attempts] == [5.0, 0.0]


def test_query_retry_does_not_retry_permanent_degrade(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(run_benchmark.time, "sleep", sleeps.append)
    client = _SequenceClient([(_llm_rerank_response(True, "invalid_response"), 100.0)])

    response, latency_ms, attempts = run_benchmark._query_with_transient_retries(
        client, {}, max_retries=3, backoff_seconds=5.0
    )

    assert run_benchmark._transient_degrade_reason(response) == ""
    assert latency_ms == 100.0
    assert sleeps == []
    assert len(attempts) == 1
    assert attempts[0]["status"] == "degraded"
    assert attempts[0]["backoff_seconds"] == 0.0


def test_query_retry_recovers_transient_transport_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(run_benchmark.time, "sleep", sleeps.append)
    perf_counter = iter([10.0, 10.25, 20.0])
    monkeypatch.setattr(run_benchmark.time, "perf_counter", lambda: next(perf_counter))
    client = _SequenceClient(
        [
            run_benchmark.CortrixClientError("POST /api/v1/query failed: connection reset"),
            (_llm_rerank_response(False), 200.0),
        ]
    )

    response, latency_ms, attempts = run_benchmark._query_with_transient_retries(
        client, {}, max_retries=1, backoff_seconds=5.0
    )

    assert run_benchmark._transient_degrade_reason(response) == ""
    assert latency_ms == 5450.0
    assert sleeps == [5.0]
    assert attempts[0]["retry_reason"] == "transport"
    assert attempts[0]["latency_ms"] == 250.0
    assert attempts[0]["backoff_seconds"] == 5.0
    assert attempts[1]["status"] == "success"


def test_query_retry_does_not_retry_permanent_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(run_benchmark.time, "sleep", sleeps.append)
    perf_counter = iter([10.0, 10.1])
    monkeypatch.setattr(run_benchmark.time, "perf_counter", lambda: next(perf_counter))
    client = _SequenceClient(
        [run_benchmark.CortrixClientError("POST /api/v1/query failed with HTTP 400: bad request")]
    )

    with pytest.raises(run_benchmark.QueryRetriesExhausted) as exc:
        run_benchmark._query_with_transient_retries(
            client, {}, max_retries=3, backoff_seconds=5.0
        )

    assert sleeps == []
    assert len(exc.value.attempts) == 1
    assert exc.value.attempts[0]["retry_reason"] == ""
    assert exc.value.attempts[0]["latency_ms"] == pytest.approx(100.0)
    assert exc.value.attempts[0]["backoff_seconds"] == 0.0


def test_query_retry_rejects_negative_retry_count() -> None:
    with pytest.raises(SystemExit) as exc:
        run_benchmark.main(["run", "--query-transient-retries", "-1"])

    assert exc.value.code == 2
