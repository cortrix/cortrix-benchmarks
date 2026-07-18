from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODELS = "glm-4-flash,glm-5.2"
TASKS = ("f03", "f41")
DEFAULT_RESPONSE_FORMAT = "json_object"

F03_PROMPT = """Return one JSON object only. Do not include markdown.
Schema:
{
  "0": {
    "entities": [{"text": "Cortrix", "type": "PRODUCT", "start_offset": 0, "end_offset": 7}],
    "summary": "A short factual summary.",
    "score": 0.9
  }
}
Text: Cortrix is a semantic storage layer for agent-native applications.
"""

F41_PROMPT = """Return one JSON object only. Do not include markdown.
Schema:
{
  "summary_text": "A concise document summary.",
  "keywords": ["semantic storage", "agents"],
  "topics": ["Technical Documentation"],
  "one_liner": "Semantic storage for agents"
}
Document title: Cortrix overview
Document content: Cortrix helps agent-native applications turn documents and supported inputs into source-linked semantic records.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OpenAI-compatible LLM contract smoke for Cortrix benchmark preflight.")
    parser.add_argument("--endpoint", default=os.environ.get("GLM_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--api-key-env", default="GLM_API_KEY")
    parser.add_argument("--models", default=os.environ.get("GLM_CONTRACT_MODELS", DEFAULT_MODELS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--response-format", choices=("json_object", "none"), default=DEFAULT_RESPONSE_FORMAT)
    parser.add_argument("--thinking-type", choices=("enabled", "disabled", "none"), default="none")
    parser.add_argument("--f41-no-reasoning-fallback", action="store_true")
    parser.add_argument("--require-stop-finish", action="store_true")
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env) or os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        raise SystemExit(f"missing API key env: {args.api_key_env} or ZHIPUAI_API_KEY")

    report = run_contract_smoke(
        endpoint=args.endpoint,
        api_key=api_key,
        models=[item.strip() for item in args.models.split(",") if item.strip()],
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        response_format=args.response_format,
        thinking_type=args.thinking_type,
        f41_allow_reasoning_fallback=not args.f41_no_reasoning_fallback,
        require_stop_finish=args.require_stop_finish,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["contract_summary"]["all_contracts_ok"] else 1


def run_contract_smoke(
    *,
    endpoint: str,
    api_key: str,
    models: Sequence[str],
    output_dir: Path,
    timeout_seconds: float,
    max_tokens: int,
    temperature: float,
    response_format: str,
    thinking_type: str,
    f41_allow_reasoning_fallback: bool,
    require_stop_finish: bool,
) -> Mapping[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_dir = output_dir / "sanitized"
    sanitized_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        for task in TASKS:
            prompt = F03_PROMPT if task == "f03" else F41_PROMPT
            request_body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format != "none":
                request_body["response_format"] = {"type": response_format}
            if thinking_type != "none":
                request_body["thinking"] = {"type": thinking_type}
            row = _call_and_analyze(
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                task=task,
                request_body=request_body,
                timeout_seconds=timeout_seconds,
                allow_reasoning_fallback=(task != "f41" or f41_allow_reasoning_fallback),
                require_stop_finish=require_stop_finish,
            )
            rows.append(row)
            stem = f"{_safe_name(model)}-{task}"
            (sanitized_dir / f"{stem}.request.json").write_text(
                json.dumps(_sanitize_request(request_body), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (sanitized_dir / f"{stem}.response-summary.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": endpoint,
        "models": list(models),
        "tasks": list(TASKS),
        "response_format": response_format,
        "thinking_type": thinking_type,
        "f41_allow_reasoning_fallback": f41_allow_reasoning_fallback,
        "require_stop_finish": require_stop_finish,
        "contract_results": rows,
        "contract_summary": {
            "all_contracts_ok": all(bool(row.get("contract_ok")) for row in rows),
            "failed": [
                {"model": row.get("model"), "task": row.get("task"), "reason": row.get("failure_reason")}
                for row in rows
                if not bool(row.get("contract_ok"))
            ],
        },
        "artifact_policy": {
            "api_key_recorded": False,
            "raw_reasoning_content_recorded": False,
            "raw_response_recorded": False,
            "note": "Stores lengths, hashes, field names, finish_reason, usage, and parse status. Does not store Authorization headers or raw reasoning_content.",
        },
    }
    (output_dir / "llm-contract-smoke-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def validate_contract_smoke_report(path: Path) -> Mapping[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("contract smoke report is not a JSON object")
    summary = report.get("contract_summary")
    if not isinstance(summary, dict):
        raise ValueError("contract smoke report missing contract_summary")
    if not bool(summary.get("all_contracts_ok")):
        raise ValueError(f"LLM contract smoke failed: {summary.get('failed')}")
    return report


def analyze_chat_response(
    model: str,
    task: str,
    http_status: int,
    body: Mapping[str, object],
    *,
    allow_reasoning_fallback: bool = True,
    require_stop_finish: bool = False,
) -> Mapping[str, object]:
    content = ""
    reasoning_content = ""
    finish_reason = ""
    message_keys = []
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        choice0 = choices[0]
        if isinstance(choice0, dict):
            finish = choice0.get("finish_reason")
            if isinstance(finish, str):
                finish_reason = finish
            message = choice0.get("message")
            if isinstance(message, dict):
                message_keys = sorted(str(key) for key in message.keys())
                raw_content = message.get("content")
                raw_reasoning = message.get("reasoning_content")
                if isinstance(raw_content, str):
                    content = raw_content
                if isinstance(raw_reasoning, str):
                    reasoning_content = raw_reasoning

    content_source = "message.content"
    selected_content = content
    if not selected_content and reasoning_content and allow_reasoning_fallback:
        selected_content = reasoning_content
        content_source = "message.reasoning_content"

    normalized = unwrap_complete_json_fence(selected_content)
    parse_ok = False
    schema_ok = False
    failure_reason = ""
    if not selected_content:
        parsed = None
        failure_reason = "empty_message_content_and_reasoning_content"
    else:
        try:
            parsed = json.loads(normalized)
            parse_ok = isinstance(parsed, dict)
        except json.JSONDecodeError as exc:
            parsed = None
            failure_reason = f"json_parse_error:{exc.msg}"
    if parse_ok and isinstance(parsed, dict):
        schema_ok, failure_reason = _validate_task_schema(task, parsed)

    usage = body.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    finish_ok = not require_stop_finish or finish_reason in {"", "stop"}
    contract_ok = http_status == 200 and parse_ok and schema_ok and finish_ok
    if http_status != 200 and not failure_reason:
        failure_reason = f"http_status:{http_status}"
    if parse_ok and schema_ok and not finish_ok and not failure_reason:
        failure_reason = f"finish_reason:{finish_reason}"
    if parse_ok and not schema_ok and not failure_reason:
        failure_reason = "schema_validation_failed"

    return {
        "model": model,
        "task": task,
        "http_status": http_status,
        "contract_ok": contract_ok,
        "allow_reasoning_fallback": allow_reasoning_fallback,
        "require_stop_finish": require_stop_finish,
        "content_source": content_source,
        "message_keys": message_keys,
        "finish_reason": finish_reason,
        "content_len": len(content),
        "reasoning_len": len(reasoning_content),
        "selected_content_len": len(selected_content),
        "selected_content_sha256": hashlib.sha256(selected_content.encode("utf-8")).hexdigest() if selected_content else "",
        "complete_fence_unwrapped": normalized != selected_content,
        "json_parse_ok": parse_ok,
        "schema_ok": schema_ok,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "failure_reason": failure_reason,
    }


def unwrap_complete_json_fence(value: str) -> str:
    trimmed = value.strip()
    if not trimmed.startswith("```"):
        return value
    first_newline = trimmed.find("\n")
    if first_newline < 0:
        return trimmed
    opener = trimmed[:first_newline].strip()
    if "`" in opener[3:]:
        return trimmed
    if not trimmed.endswith("```"):
        return trimmed
    inner = trimmed[first_newline + 1 : -3]
    return inner.rstrip("\r\n")


def _call_and_analyze(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    task: str,
    request_body: Mapping[str, object],
    timeout_seconds: float,
    allow_reasoning_fallback: bool,
    require_stop_finish: bool,
) -> Mapping[str, object]:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_payload = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        response_payload = exc.read()
        status = int(exc.code)
    except urllib.error.URLError as exc:
        return {
            "model": model,
            "task": task,
            "http_status": 0,
            "contract_ok": False,
            "failure_reason": f"transport_error:{exc}",
        }
    try:
        body = json.loads(response_payload)
    except json.JSONDecodeError as exc:
        return {
            "model": model,
            "task": task,
            "http_status": status,
            "contract_ok": False,
            "failure_reason": f"response_json_parse_error:{exc.msg}",
            "response_bytes": len(response_payload),
        }
    if not isinstance(body, dict):
        return {
            "model": model,
            "task": task,
            "http_status": status,
            "contract_ok": False,
            "failure_reason": "response_body_not_object",
            "response_bytes": len(response_payload),
        }
    analyzed = dict(
        analyze_chat_response(
            model,
            task,
            status,
            body,
            allow_reasoning_fallback=allow_reasoning_fallback,
            require_stop_finish=require_stop_finish,
        )
    )
    analyzed["response_bytes"] = len(response_payload)
    return analyzed


def _validate_task_schema(task: str, parsed: Mapping[str, object]) -> tuple[bool, str]:
    if task == "f03":
        chunk = parsed.get("0")
        if not isinstance(chunk, dict):
            return False, "f03_missing_chunk_0"
        if not isinstance(chunk.get("summary"), str):
            return False, "f03_missing_summary_string"
        entities = chunk.get("entities")
        if not isinstance(entities, list):
            return False, "f03_missing_entities_array"
        for entity in entities:
            if not isinstance(entity, dict):
                return False, "f03_entity_not_object"
            if not isinstance(entity.get("text"), str) or not isinstance(entity.get("type"), str):
                return False, "f03_entity_missing_text_or_type"
            if not _is_number_or_numeric_string(entity.get("start_offset")):
                return False, "f03_entity_start_offset_not_numeric"
            if not _is_number_or_numeric_string(entity.get("end_offset")):
                return False, "f03_entity_end_offset_not_numeric"
        score = chunk.get("score")
        if score is not None and not isinstance(score, (int, float)):
            return False, "f03_score_not_numeric"
        return True, ""
    if task == "f41":
        if not isinstance(parsed.get("summary_text"), str):
            return False, "f41_missing_summary_text_string"
        for key in ("keywords", "topics"):
            value = parsed.get(key)
            if value is not None and not isinstance(value, list):
                return False, f"f41_{key}_not_array"
        return True, ""
    return False, f"unknown_task:{task}"


def _is_number_or_numeric_string(value: object) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            int(value)
            return True
        except ValueError:
            return False
    return False


def _sanitize_request(request_body: Mapping[str, object]) -> Mapping[str, object]:
    messages = request_body.get("messages")
    first_content = ""
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        first_content = str(messages[0].get("content", ""))
    return {
        "model": request_body.get("model"),
        "temperature": request_body.get("temperature"),
        "max_tokens": request_body.get("max_tokens"),
        "response_format": request_body.get("response_format"),
        "thinking": request_body.get("thinking"),
        "messages": [
            {
                "role": "user",
                "content_sha256": hashlib.sha256(first_content.encode("utf-8")).hexdigest(),
                "content_len": len(first_content),
            }
        ],
    }


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120]


if __name__ == "__main__":
    raise SystemExit(main())
