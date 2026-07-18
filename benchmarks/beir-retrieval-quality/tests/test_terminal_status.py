from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RUNNER_DIR = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER_DIR))

from terminal_status import (  # noqa: E402
    build_terminal_status,
    main,
    validate_terminal_status,
)


def _summary(*query_failures: int, statuses: tuple[str, ...] | None = None) -> dict[str, object]:
    if statuses is None:
        statuses = tuple("completed" for _ in query_failures)
    return {
        "run_id": "synthetic-run",
        "results": [
            {
                "status": status,
                "imported_documents": 10,
                "metrics": {
                    "query_failures": failures,
                    "evaluated_queries": 5,
                    "recall@10": {"count": 5},
                    "ndcg@10": {"count": 5},
                    "latency_ms": {"count": 5},
                },
            }
            for status, failures in zip(statuses, query_failures, strict=True)
        ],
    }


def test_scientific_valid_and_feature_clean_are_separate_dimensions() -> None:
    status = build_terminal_status(
        _summary(0, 0),
        {"status": "PASS", "reason": "all required feature proof is clean"},
        wrapper_return_code=0,
    )

    assert status["scientific_result"]["status"] == "VALID"
    assert status["scientific_result"]["imported_documents"] == 20
    assert status["scientific_result"]["evaluated_queries"] == 10
    assert status["strict_feature_completeness"]["status"] == "PASS"
    assert status["lane_classification"] == "SCIENTIFIC_COMPLETE"
    assert status["wrapper_process"] == {"return_code": 0, "preserved": True}


def test_scientific_valid_with_feature_debt_preserves_counts_and_wrapper_rc() -> None:
    status = build_terminal_status(
        _summary(0),
        {
            "status": "FAIL",
            "pending_retry": 2,
            "failed_permanent": 3,
            "orphan_sources": 1,
            "other_mismatches": 4,
            "reason": "synthetic strict proof contains debt",
        },
        wrapper_return_code=1,
    )

    assert status["scientific_result"]["status"] == "VALID"
    assert status["strict_feature_completeness"]["total_debt"] == 10
    assert status["lane_classification"] == "SCIENTIFIC_COMPLETE_WITH_FEATURE_DEBT"
    assert status["wrapper_process"] == {"return_code": 1, "preserved": True}


def test_query_failure_makes_scientific_result_invalid() -> None:
    status = build_terminal_status(
        _summary(1),
        {"status": "NOT_EVALUATED"},
        wrapper_return_code=1,
    )

    assert status["scientific_result"]["status"] == "INVALID"
    assert status["scientific_result"]["query_failures"] == 1
    assert status["lane_classification"] == "SCIENTIFIC_INVALID"


def test_incomplete_cell_makes_scientific_result_invalid() -> None:
    status = build_terminal_status(
        _summary(0, statuses=("failed",)),
        None,
        wrapper_return_code=1,
    )

    assert status["scientific_result"]["status"] == "INVALID"
    assert status["lane_classification"] == "SCIENTIFIC_INVALID"


def test_existing_summary_without_feature_proof_remains_compatible() -> None:
    status = build_terminal_status(_summary(0), None, wrapper_return_code=0)

    assert status["scientific_result"]["status"] == "VALID"
    assert status["strict_feature_completeness"]["status"] == "NOT_EVALUATED"
    assert status["lane_classification"] == "SCIENTIFIC_COMPLETE_FEATURE_STATUS_UNKNOWN"


def test_legacy_summary_without_query_failure_evidence_is_not_promoted() -> None:
    status = build_terminal_status(
        {"run_id": "legacy-run", "results": [{"status": "completed", "metrics": {}}]},
        None,
        wrapper_return_code=0,
    )

    assert status["scientific_result"]["status"] == "NOT_EVALUATED"
    assert status["lane_classification"] == "SCIENTIFIC_STATUS_UNKNOWN"


def test_completed_summary_without_scorecard_evidence_is_not_promoted() -> None:
    status = build_terminal_status(
        {
            "run_id": "legacy-run",
            "results": [
                {
                    "status": "completed",
                    "metrics": {"query_failures": 0},
                }
            ],
        },
        None,
        wrapper_return_code=0,
    )

    assert status["scientific_result"]["status"] == "NOT_EVALUATED"
    assert status["scientific_result"]["reason"] == "scorecard_evidence_missing"
    assert status["lane_classification"] == "SCIENTIFIC_STATUS_UNKNOWN"


def test_inconsistent_feature_proof_is_rejected() -> None:
    with pytest.raises(ValueError, match="PASS requires all debt counts to be zero"):
        build_terminal_status(
            _summary(0),
            {"status": "PASS", "pending_retry": 1},
            wrapper_return_code=0,
        )


def test_terminal_artifact_is_written_before_wrapper_rc_is_propagated(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    proof_path = tmp_path / "feature-proof.json"
    output_path = tmp_path / "terminal-status.json"
    summary_path.write_text(json.dumps(_summary(0)), encoding="utf-8")
    proof_path.write_text(
        json.dumps({"status": "FAIL", "failed_permanent": 1}),
        encoding="utf-8",
    )

    return_code = main(
        [
            "--summary",
            str(summary_path),
            "--feature-proof",
            str(proof_path),
            "--wrapper-rc",
            "1",
            "--output",
            str(output_path),
            "--propagate-wrapper-rc",
        ]
    )

    assert return_code == 1
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["lane_classification"] == "SCIENTIFIC_COMPLETE_WITH_FEATURE_DEBT"
    assert payload["wrapper_process"]["return_code"] == 1
    assert not list(tmp_path.glob(".terminal-status.json.*.tmp"))


def test_generated_payload_matches_runtime_validator() -> None:
    status = build_terminal_status(
        _summary(0),
        {"status": "FAIL", "orphan_sources": 1},
        wrapper_return_code=1,
    )

    validate_terminal_status(status)


def test_terminal_status_schema_is_valid_json_and_matches_required_fields() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "terminal-status.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert set(schema["required"]) == {
        "schema_version",
        "run_id",
        "source_summary_format",
        "scientific_result",
        "strict_feature_completeness",
        "lane_classification",
        "wrapper_process",
    }
