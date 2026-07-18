from __future__ import annotations

import json
import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parents[1] / "runner"
sys.path.insert(0, str(RUNNER_DIR))

from llm_contract_smoke import (  # noqa: E402
    _sanitize_request,
    analyze_chat_response,
    unwrap_complete_json_fence,
    validate_contract_smoke_report,
)


def test_unwrap_complete_json_fence_only() -> None:
    raw = '{"summary_text":"ok"}'
    assert unwrap_complete_json_fence(f"```json\n{raw}\n```") == raw
    assert unwrap_complete_json_fence(f"```\n{raw}\n```") == raw
    assert unwrap_complete_json_fence(f"Here:\n```json\n{raw}\n```") != raw


def test_analyze_f03_accepts_reasoning_content_and_numeric_strings() -> None:
    body = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "",
                    "reasoning_content": json.dumps(
                        {
                            "0": {
                                "entities": [
                                    {
                                        "text": "Cortrix",
                                        "type": "PRODUCT",
                                        "start_offset": "0",
                                        "end_offset": "7",
                                    }
                                ],
                                "summary": "about Cortrix",
                                "score": 0.9,
                            }
                        }
                    ),
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    row = analyze_chat_response("glm-5.2", "f03", 200, body)
    assert row["contract_ok"] is True
    assert row["content_source"] == "message.reasoning_content"
    assert row["reasoning_len"] > 0
    assert row["selected_content_sha256"]


def test_analyze_f41_accepts_complete_fenced_content() -> None:
    content = "```json\n" + json.dumps({"summary_text": "ok", "keywords": [], "topics": []}) + "\n```"
    row = analyze_chat_response(
        "glm-4-flash",
        "f41",
        200,
        {"choices": [{"finish_reason": "stop", "message": {"content": content}}]},
    )
    assert row["contract_ok"] is True
    assert row["complete_fence_unwrapped"] is True


def test_analyze_f41_can_reject_reasoning_fallback() -> None:
    body = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": "",
                    "reasoning_content": json.dumps(
                        {"summary_text": "ok", "keywords": [], "topics": []}
                    ),
                },
            }
        ]
    }
    row = analyze_chat_response(
        "deepseek-v4-flash",
        "f41",
        200,
        body,
        allow_reasoning_fallback=False,
    )
    assert row["contract_ok"] is False
    assert row["content_source"] == "message.content"
    assert row["failure_reason"] == "empty_message_content_and_reasoning_content"


def test_analyze_rejects_length_finish_when_required() -> None:
    body = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": json.dumps(
                        {"summary_text": "ok", "keywords": [], "topics": []}
                    )
                },
            }
        ]
    }
    row = analyze_chat_response(
        "deepseek-v4-flash",
        "f41",
        200,
        body,
        require_stop_finish=True,
    )
    assert row["contract_ok"] is False
    assert row["json_parse_ok"] is True
    assert row["schema_ok"] is True
    assert row["failure_reason"] == "finish_reason:length"


def test_validate_contract_smoke_report_rejects_failure(tmp_path: Path) -> None:
    report = {
        "contract_summary": {
            "all_contracts_ok": False,
            "failed": [{"model": "glm-5.2", "task": "f41", "reason": "json_parse_error"}],
        }
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    try:
        validate_contract_smoke_report(path)
    except ValueError as exc:
        assert "LLM contract smoke failed" in str(exc)
    else:
        raise AssertionError("expected failing report to raise")


def test_sanitized_request_preserves_thinking_type() -> None:
    sanitized = _sanitize_request(
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Return JSON."}],
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }
    )
    assert sanitized["thinking"] == {"type": "disabled"}
