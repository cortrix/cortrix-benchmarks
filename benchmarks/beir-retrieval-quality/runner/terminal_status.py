from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


SCIENTIFIC_STATUSES = {"VALID", "INVALID", "NOT_EVALUATED"}
STRICT_FEATURE_STATUSES = {"PASS", "FAIL", "NOT_EVALUATED"}
DEBT_FIELDS = (
    "pending_retry",
    "failed_permanent",
    "orphan_sources",
    "other_mismatches",
)


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _wrapper_return_code(value: object) -> int:
    return_code = _non_negative_int(value, "wrapper return code")
    if return_code > 255:
        raise ValueError("wrapper return code must be between 0 and 255")
    return return_code


def infer_scientific_result(summary: Mapping[str, object]) -> dict[str, object]:
    results = summary.get("results")
    if not isinstance(results, list) or not results:
        return {
            "status": "NOT_EVALUATED",
            "total_cells": 0,
            "completed_cells": 0,
            "imported_documents": 0,
            "evaluated_queries": 0,
            "query_failures": None,
            "reason": "result_summary_has_no_cells",
        }

    total_cells = len(results)
    completed_cells = sum(
        1
        for item in results
        if isinstance(item, Mapping) and item.get("status") == "completed"
    )
    if completed_cells != total_cells:
        return {
            "status": "INVALID",
            "total_cells": total_cells,
            "completed_cells": completed_cells,
            "imported_documents": None,
            "evaluated_queries": None,
            "query_failures": None,
            "reason": "one_or_more_cells_not_completed",
        }

    query_failures = 0
    for index, item in enumerate(results):
        if not isinstance(item, Mapping):
            raise ValueError(f"results[{index}] must be an object")
        metrics = item.get("metrics")
        if not isinstance(metrics, Mapping) or "query_failures" not in metrics:
            return {
                "status": "NOT_EVALUATED",
                "total_cells": total_cells,
                "completed_cells": completed_cells,
                "imported_documents": None,
                "evaluated_queries": None,
                "query_failures": None,
                "reason": "query_failure_evidence_missing",
            }
        query_failures += _non_negative_int(
            metrics["query_failures"], f"results[{index}].metrics.query_failures"
        )

    if query_failures:
        return {
            "status": "INVALID",
            "total_cells": total_cells,
            "completed_cells": completed_cells,
            "imported_documents": None,
            "evaluated_queries": None,
            "query_failures": query_failures,
            "reason": "one_or_more_queries_failed",
        }

    imported_documents = 0
    evaluated_queries = 0
    for index, item in enumerate(results):
        if not isinstance(item, Mapping):
            raise ValueError(f"results[{index}] must be an object")
        metrics = item.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"results[{index}].metrics must be an object")
        if "imported_documents" not in item or "evaluated_queries" not in metrics:
            return {
                "status": "NOT_EVALUATED",
                "total_cells": total_cells,
                "completed_cells": completed_cells,
                "imported_documents": None,
                "evaluated_queries": None,
                "query_failures": 0,
                "reason": "scorecard_evidence_missing",
            }
        imported = _non_negative_int(
            item["imported_documents"], f"results[{index}].imported_documents"
        )
        evaluated = _non_negative_int(
            metrics["evaluated_queries"], f"results[{index}].metrics.evaluated_queries"
        )
        recall_summaries = [
            value for key, value in metrics.items() if str(key).startswith("recall@")
        ]
        ndcg_summaries = [
            value for key, value in metrics.items() if str(key).startswith("ndcg@")
        ]
        latency_summary = metrics.get("latency_ms")
        score_summaries = [*recall_summaries, *ndcg_summaries, latency_summary]
        if (
            imported == 0
            or evaluated == 0
            or not recall_summaries
            or not ndcg_summaries
            or any(
                not isinstance(summary, Mapping)
                or summary.get("count") != evaluated
                for summary in score_summaries
            )
        ):
            return {
                "status": "NOT_EVALUATED",
                "total_cells": total_cells,
                "completed_cells": completed_cells,
                "imported_documents": None,
                "evaluated_queries": None,
                "query_failures": 0,
                "reason": "scorecard_evidence_incomplete",
            }
        imported_documents += imported
        evaluated_queries += evaluated

    return {
        "status": "VALID",
        "total_cells": total_cells,
        "completed_cells": completed_cells,
        "imported_documents": imported_documents,
        "evaluated_queries": evaluated_queries,
        "query_failures": 0,
        "reason": "all_cells_completed_with_zero_query_failures",
    }


def normalize_feature_proof(
    feature_proof: Mapping[str, object] | None,
) -> dict[str, object]:
    if feature_proof is None:
        return {
            "status": "NOT_EVALUATED",
            "pending_retry": 0,
            "failed_permanent": 0,
            "orphan_sources": 0,
            "other_mismatches": 0,
            "total_debt": 0,
            "reason": "feature_proof_not_provided",
        }

    status = feature_proof.get("status")
    if status not in STRICT_FEATURE_STATUSES:
        allowed = ", ".join(sorted(STRICT_FEATURE_STATUSES))
        raise ValueError(f"feature proof status must be one of: {allowed}")

    counts = {
        field: _non_negative_int(feature_proof.get(field, 0), f"feature proof {field}")
        for field in DEBT_FIELDS
    }
    total_debt = sum(counts.values())
    if status == "PASS" and total_debt:
        raise ValueError("feature proof PASS requires all debt counts to be zero")
    if status == "FAIL" and total_debt == 0:
        raise ValueError("feature proof FAIL requires at least one debt count")
    if status == "NOT_EVALUATED" and total_debt:
        raise ValueError("feature proof NOT_EVALUATED requires all debt counts to be zero")

    reason = feature_proof.get("reason")
    if reason is None:
        reason = {
            "PASS": "required_feature_proof_is_clean",
            "FAIL": "required_feature_proof_has_debt",
            "NOT_EVALUATED": "feature_proof_not_evaluated",
        }[str(status)]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("feature proof reason must be a non-empty string")

    return {
        "status": status,
        **counts,
        "total_debt": total_debt,
        "reason": reason,
    }


def lane_classification(scientific_status: str, strict_feature_status: str) -> str:
    if scientific_status == "INVALID":
        return "SCIENTIFIC_INVALID"
    if scientific_status == "NOT_EVALUATED":
        return "SCIENTIFIC_STATUS_UNKNOWN"
    if strict_feature_status == "PASS":
        return "SCIENTIFIC_COMPLETE"
    if strict_feature_status == "FAIL":
        return "SCIENTIFIC_COMPLETE_WITH_FEATURE_DEBT"
    return "SCIENTIFIC_COMPLETE_FEATURE_STATUS_UNKNOWN"


def build_terminal_status(
    summary: Mapping[str, object],
    feature_proof: Mapping[str, object] | None,
    wrapper_return_code: int,
) -> dict[str, object]:
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("result summary run_id must be a non-empty string")

    return_code = _wrapper_return_code(wrapper_return_code)
    scientific_result = infer_scientific_result(summary)
    strict_feature = normalize_feature_proof(feature_proof)
    terminal_status = {
        "schema_version": "1.0",
        "run_id": run_id,
        "source_summary_format": "result-summary-v1",
        "scientific_result": scientific_result,
        "strict_feature_completeness": strict_feature,
        "lane_classification": lane_classification(
            str(scientific_result["status"]), str(strict_feature["status"])
        ),
        "wrapper_process": {
            "return_code": return_code,
            "preserved": True,
        },
    }
    validate_terminal_status(terminal_status)
    return terminal_status


def validate_terminal_status(data: Mapping[str, object]) -> None:
    required = {
        "schema_version",
        "run_id",
        "source_summary_format",
        "scientific_result",
        "strict_feature_completeness",
        "lane_classification",
        "wrapper_process",
    }
    if set(data) != required:
        raise ValueError("terminal status fields do not match schema version 1.0")
    if data["schema_version"] != "1.0":
        raise ValueError("unsupported terminal status schema version")
    if data["source_summary_format"] != "result-summary-v1":
        raise ValueError("unsupported result summary format")
    if not isinstance(data["run_id"], str) or not str(data["run_id"]).strip():
        raise ValueError("terminal status run_id must be a non-empty string")

    scientific = data["scientific_result"]
    if not isinstance(scientific, Mapping):
        raise ValueError("scientific_result must be an object")
    if set(scientific) != {
        "status",
        "total_cells",
        "completed_cells",
        "imported_documents",
        "evaluated_queries",
        "query_failures",
        "reason",
    }:
        raise ValueError("scientific_result fields do not match schema version 1.0")
    scientific_status = scientific.get("status")
    if scientific_status not in SCIENTIFIC_STATUSES:
        raise ValueError("scientific_result has an unsupported status")
    for field in ("total_cells", "completed_cells"):
        _non_negative_int(scientific.get(field), f"scientific_result.{field}")
    for field in ("imported_documents", "evaluated_queries", "query_failures"):
        value = scientific.get(field)
        if value is not None:
            _non_negative_int(value, f"scientific_result.{field}")
    if not isinstance(scientific.get("reason"), str) or not str(scientific["reason"]).strip():
        raise ValueError("scientific_result.reason must be a non-empty string")

    strict = data["strict_feature_completeness"]
    if not isinstance(strict, Mapping):
        raise ValueError("strict_feature_completeness must be an object")
    if set(strict) != {
        "status",
        "pending_retry",
        "failed_permanent",
        "orphan_sources",
        "other_mismatches",
        "total_debt",
        "reason",
    }:
        raise ValueError(
            "strict_feature_completeness fields do not match schema version 1.0"
        )
    strict_status = strict.get("status")
    if strict_status not in STRICT_FEATURE_STATUSES:
        raise ValueError("strict_feature_completeness has an unsupported status")
    counts = [_non_negative_int(strict.get(field), f"strict_feature_completeness.{field}") for field in DEBT_FIELDS]
    total_debt = _non_negative_int(
        strict.get("total_debt"), "strict_feature_completeness.total_debt"
    )
    if total_debt != sum(counts):
        raise ValueError("strict_feature_completeness.total_debt does not match debt counts")
    if strict_status == "PASS" and total_debt:
        raise ValueError("strict feature PASS cannot contain debt")
    if strict_status == "FAIL" and total_debt == 0:
        raise ValueError("strict feature FAIL must contain debt")
    if strict_status == "NOT_EVALUATED" and total_debt:
        raise ValueError("strict feature NOT_EVALUATED cannot contain debt")
    if not isinstance(strict.get("reason"), str) or not str(strict["reason"]).strip():
        raise ValueError("strict_feature_completeness.reason must be a non-empty string")

    expected_lane = lane_classification(str(scientific_status), str(strict_status))
    if data["lane_classification"] != expected_lane:
        raise ValueError("lane_classification is inconsistent with component statuses")

    wrapper = data["wrapper_process"]
    if not isinstance(wrapper, Mapping):
        raise ValueError("wrapper_process must be an object")
    if set(wrapper) != {"return_code", "preserved"}:
        raise ValueError("wrapper_process fields do not match schema version 1.0")
    _wrapper_return_code(wrapper.get("return_code"))
    if wrapper.get("preserved") is not True:
        raise ValueError("wrapper_process.preserved must be true")


def write_json_atomic(path: Path, data: Mapping[str, object]) -> None:
    validate_terminal_status(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _read_json_object(path: Path, label: str) -> Mapping[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write a terminal benchmark status without conflating scientific validity "
            "with strict feature completeness."
        )
    )
    parser.add_argument("--summary", type=Path, required=True, help="Existing runner summary.json")
    parser.add_argument(
        "--feature-proof",
        type=Path,
        help="Optional generic strict feature proof JSON",
    )
    parser.add_argument("--wrapper-rc", type=int, required=True, help="Original wrapper return code")
    parser.add_argument("--output", type=Path, required=True, help="terminal-status.json output path")
    parser.add_argument(
        "--propagate-wrapper-rc",
        action="store_true",
        help="Return the preserved wrapper code after atomically writing the artifact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary = _read_json_object(args.summary, "result summary")
        feature_proof = (
            _read_json_object(args.feature_proof, "feature proof")
            if args.feature_proof is not None
            else None
        )
        terminal_status = build_terminal_status(summary, feature_proof, args.wrapper_rc)
        write_json_atomic(args.output, terminal_status)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"terminal status error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(terminal_status, ensure_ascii=False, indent=2))
    if args.propagate_wrapper_rc:
        return _wrapper_return_code(args.wrapper_rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
