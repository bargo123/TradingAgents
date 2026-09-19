"""Bounded, read-only diagnostics for the local Ollama native chat API."""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx


def build_native_payload(
    model: str,
    messages: Sequence[Mapping[str, str]],
    schema: Mapping[str, Any],
    *,
    context_tokens: int,
    max_tokens: int,
    keep_alive: str | int = "5m",
) -> dict[str, Any]:
    """Build a native ``/api/chat`` payload without tool calls or prose fallback."""

    if not model.strip():
        raise ValueError("model is required")
    if isinstance(context_tokens, bool) or context_tokens < 1:
        raise ValueError("context_tokens must be positive")
    if isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    return {
        "model": model,
        "messages": [dict(message) for message in messages],
        "stream": False,
        "think": False,
        "format": dict(schema),
        "options": {
            "temperature": 0.0,
            "num_ctx": int(context_tokens),
            "num_predict": int(max_tokens),
        },
        "keep_alive": keep_alive,
    }


def classify_diagnostic_error(error: BaseException) -> str:
    """Return a bounded class name without exposing provider error text."""

    name = type(error).__name__.casefold()
    module = type(error).__module__.casefold()
    if isinstance(error, TimeoutError) or "timeout" in name or "timeout" in module:
        return "PROVIDER_TIMEOUT"
    if isinstance(error, httpx.HTTPStatusError) or getattr(error, "status_code", None) is not None:
        return "HTTP_ERROR"
    if isinstance(error, (json.JSONDecodeError,)) or "jsondecode" in name:
        return "JSON_INVALID"
    if "connect" in name or "connection" in name:
        return "CONNECTION_ERROR"
    return "PROVIDER_ERROR"


def _rate(count: Any, duration_ns: Any) -> float | None:
    if not isinstance(count, int) or not isinstance(duration_ns, int) or duration_ns <= 0:
        return None
    return round(count / (duration_ns / 1_000_000_000), 3)


def _is_output_truncated(done_reason: Any) -> bool:
    return isinstance(done_reason, str) and done_reason.casefold() in {
        "length",
        "max_tokens",
    }


def summarise_native_response(
    response: Mapping[str, Any],
    *,
    test_name: str,
    model: str,
    request_chars: int,
    approximate_input_tokens: int,
    schema_valid: bool,
    elapsed_seconds: float,
    parse_error: str | None = None,
    status: str | None = None,
    source_chars: int = 0,
    instruction_chars: int = 0,
    schema_chars: int = 0,
    context_tokens: int | None = None,
    max_tokens: int | None = None,
    load_phase: str = "unknown",
) -> dict[str, Any]:
    """Convert an Ollama response to scalar-only diagnostics."""

    prompt_count = response.get("prompt_eval_count")
    eval_count = response.get("eval_count")
    prompt_duration = response.get("prompt_eval_duration")
    eval_duration = response.get("eval_duration")
    done_reason = response.get("done_reason")
    if not isinstance(done_reason, str):
        done_reason = "stop" if response.get("done") is True else None
    output_truncated = _is_output_truncated(done_reason)
    if status is None:
        if output_truncated and not schema_valid:
            status = "OUTPUT_TRUNCATED"
        elif schema_valid:
            status = "SUCCESS"
        elif parse_error == "JSONDecodeError":
            status = "JSON_INVALID"
        else:
            status = "SCHEMA_INVALID"
    return {
        "test": test_name,
        "model": model,
        "status": status,
        "output_truncated": output_truncated,
        "schema_valid": bool(schema_valid),
        "parse_error": parse_error,
        "request_chars": int(request_chars),
        "source_chars": int(source_chars),
        "instruction_chars": int(instruction_chars),
        "schema_chars": int(schema_chars),
        "approximate_input_tokens": int(approximate_input_tokens),
        "prompt_eval_count": prompt_count if isinstance(prompt_count, int) else None,
        "eval_count": eval_count if isinstance(eval_count, int) else None,
        "prompt_eval_duration_ns": prompt_duration if isinstance(prompt_duration, int) else None,
        "eval_duration_ns": eval_duration if isinstance(eval_duration, int) else None,
        "load_duration_ns": response.get("load_duration")
        if isinstance(response.get("load_duration"), int)
        else None,
        "total_duration_ns": response.get("total_duration")
        if isinstance(response.get("total_duration"), int)
        else None,
        "prompt_tokens_per_second": _rate(prompt_count, prompt_duration),
        "generation_tokens_per_second": _rate(eval_count, eval_duration),
        "done": response.get("done") if isinstance(response.get("done"), bool) else None,
        "done_reason": done_reason,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "context_tokens": context_tokens,
        "max_tokens": max_tokens,
        "load_phase": load_phase,
    }


def _message_chars(messages: Sequence[Mapping[str, str]]) -> tuple[int, int]:
    total = sum(len(str(message.get("content", ""))) for message in messages)
    return total, total


def run_native_call(
    client: httpx.Client,
    *,
    model: str,
    test_name: str,
    messages: Sequence[Mapping[str, str]],
    schema: Mapping[str, Any],
    validator: Callable[[Any], None],
    context_tokens: int,
    max_tokens: int,
    keep_alive: str | int,
    source_chars: int = 0,
    instruction_chars: int = 0,
    load_phase: str = "unknown",
) -> dict[str, Any]:
    """Execute one native call and discard all response text after validation."""

    request_chars, _ = _message_chars(messages)
    schema_chars = len(json.dumps(schema, sort_keys=True, separators=(",", ":")))
    approximate_input_tokens = max(1, math.ceil(request_chars / 4))
    payload = build_native_payload(
        model,
        messages,
        schema,
        context_tokens=context_tokens,
        max_tokens=max_tokens,
        keep_alive=keep_alive,
    )
    started = time.perf_counter()
    try:
        response = client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, Mapping):
            raise TypeError("native response must be an object")
        message = data.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            return summarise_native_response(
                data,
                test_name=test_name,
                model=model,
                request_chars=request_chars,
                approximate_input_tokens=approximate_input_tokens,
                schema_valid=False,
                elapsed_seconds=time.perf_counter() - started,
                parse_error="EMPTY_RESPONSE",
                status="EMPTY_RESPONSE",
                source_chars=source_chars,
                instruction_chars=instruction_chars,
                schema_chars=schema_chars,
                context_tokens=context_tokens,
                max_tokens=max_tokens,
                load_phase=load_phase,
            )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return summarise_native_response(
                data,
                test_name=test_name,
                model=model,
                request_chars=request_chars,
                approximate_input_tokens=approximate_input_tokens,
                schema_valid=False,
                elapsed_seconds=time.perf_counter() - started,
                parse_error="JSONDecodeError",
                source_chars=source_chars,
                instruction_chars=instruction_chars,
                schema_chars=schema_chars,
                context_tokens=context_tokens,
                max_tokens=max_tokens,
                load_phase=load_phase,
            )
        try:
            validation_status = validator(parsed)
        except Exception as error:
            return summarise_native_response(
                data,
                test_name=test_name,
                model=model,
                request_chars=request_chars,
                approximate_input_tokens=approximate_input_tokens,
                schema_valid=False,
                elapsed_seconds=time.perf_counter() - started,
                parse_error=type(error).__name__,
                source_chars=source_chars,
                instruction_chars=instruction_chars,
                schema_chars=schema_chars,
                context_tokens=context_tokens,
                max_tokens=max_tokens,
                load_phase=load_phase,
            )
        if validation_status in {"GROUNDING_FAILED", "QUALITY_FAILED", "SCHEMA_INVALID"}:
            return summarise_native_response(
                data,
                test_name=test_name,
                model=model,
                request_chars=request_chars,
                approximate_input_tokens=approximate_input_tokens,
                schema_valid=False,
                elapsed_seconds=time.perf_counter() - started,
                parse_error=str(validation_status),
                status=str(validation_status),
                source_chars=source_chars,
                instruction_chars=instruction_chars,
                schema_chars=schema_chars,
                context_tokens=context_tokens,
                max_tokens=max_tokens,
                load_phase=load_phase,
            )
        return summarise_native_response(
            data,
            test_name=test_name,
            model=model,
            request_chars=request_chars,
            approximate_input_tokens=approximate_input_tokens,
            schema_valid=True,
            elapsed_seconds=time.perf_counter() - started,
            source_chars=source_chars,
            instruction_chars=instruction_chars,
            schema_chars=schema_chars,
            context_tokens=context_tokens,
            max_tokens=max_tokens,
            load_phase=load_phase,
        )
    except httpx.HTTPStatusError as error:
        data = {}
        status_code = error.response.status_code if error.response is not None else None
        summary = summarise_native_response(
            data,
            test_name=test_name,
            model=model,
            request_chars=request_chars,
            approximate_input_tokens=approximate_input_tokens,
            schema_valid=False,
            elapsed_seconds=time.perf_counter() - started,
            parse_error=type(error).__name__,
            status="HTTP_ERROR",
            source_chars=source_chars,
            instruction_chars=instruction_chars,
            schema_chars=schema_chars,
            context_tokens=context_tokens,
            max_tokens=max_tokens,
            load_phase=load_phase,
        )
        summary["http_status"] = status_code
        return summary
    except Exception as error:
        return summarise_native_response(
            {},
            test_name=test_name,
            model=model,
            request_chars=request_chars,
            approximate_input_tokens=approximate_input_tokens,
            schema_valid=False,
            elapsed_seconds=time.perf_counter() - started,
            parse_error=type(error).__name__,
            status=classify_diagnostic_error(error),
            source_chars=source_chars,
            instruction_chars=instruction_chars,
            schema_chars=schema_chars,
            context_tokens=context_tokens,
            max_tokens=max_tokens,
            load_phase=load_phase,
        )


def stop_model(model: str, *, timeout_seconds: float = 30.0) -> None:
    """Best-effort unload of one local model; no output is retained."""

    try:
        subprocess.run(
            ["ollama", "stop", model],
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return


__all__ = [
    "build_native_payload",
    "classify_diagnostic_error",
    "run_native_call",
    "stop_model",
    "summarise_native_response",
]
