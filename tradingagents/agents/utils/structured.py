"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from tradingagents.llm_clients.openai_client import OllamaChatOpenAI

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputRequiredError(RuntimeError):
    """Raised when a structured-output binding is required but unavailable."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 1,
        previous_errors: tuple[BaseException, ...] = (),
    ) -> None:
        super().__init__(message)
        # Retry provenance remains in memory only.  Persisted callers expose
        # only bounded failure categories, never provider text or model output.
        self.attempts = attempts
        self.previous_errors = previous_errors


def structured_failure_category(exc: BaseException) -> str:
    """Return a safe type-only category for a structured-output failure.

    The underlying provider exception is chained for local diagnostics, but
    its message can contain request details.  Callers that persist a failure
    should expose only this bounded category, never the provider text or model
    output.
    """

    cause = exc.__cause__
    return type(cause if cause is not None else exc).__name__


def _safe_error_path(location: Any) -> str:
    """Render a Pydantic error location without retaining model data."""

    if not isinstance(location, (tuple, list)):
        return "$"
    path = ""
    for part in location:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            text = str(part).replace("\r", " ").replace("\n", " ")
            path += f".{text}" if path else text
    return path or "$"


def _safe_error_message(message: Any) -> str:
    """Bound and sanitize a validator message before it reaches logs/DB."""

    text = " ".join(str(message or "").split())
    # Validator messages can echo an arbitrary model-provided scalar.  Keep the
    # stable human-readable prefix, but never persist quoted/JSON payloads.
    text = re.sub(r"(['\"]).*?\1", "<redacted>", text)
    text = re.sub(r"\{.*\}|\[.*\]", "<redacted>", text)
    return text[:160] + ("..." if len(text) > 160 else "")


def _validation_error_details(exc: BaseException) -> list[dict[str, Any]]:
    """Extract field/type metadata from a Pydantic ValidationError only.

    ``input`` is inspected solely to record its Python type.  The value itself
    is deliberately never returned, logged, or persisted because it may contain
    arbitrary model prose or sensitive context.
    """

    errors_method = getattr(exc, "errors", None)
    if not callable(errors_method):
        return []
    try:
        entries = errors_method(include_url=False, include_context=False)
    except TypeError:  # pragma: no cover - compatibility with older Pydantic
        try:
            entries = errors_method()
        except Exception:  # pragma: no cover - defensive diagnostics boundary
            return []
    except Exception:  # pragma: no cover - defensive diagnostics boundary
        return []

    details: list[dict[str, Any]] = []
    for entry in entries if isinstance(entries, list) else ():
        if not isinstance(entry, dict):
            continue
        input_value = entry.get("input")
        detail = {
            "path": _safe_error_path(entry.get("loc")),
            "type": str(entry.get("type", "unknown")),
            "message": _safe_error_message(entry.get("msg", "")),
            "input_type": type(input_value).__name__ if "input" in entry else None,
        }
        details.append(detail)
    return details


def _structured_attempt_causes(exc: BaseException):
    """Yield ``(attempt, cause)`` pairs without exposing exception messages."""

    if isinstance(exc, StructuredOutputRequiredError):
        previous = list(exc.previous_errors)
        for index, previous_error in enumerate(previous, start=1):
            yield index, previous_error.__cause__ or previous_error
        if exc.__cause__ is not None:
            yield len(previous) + 1, exc.__cause__
        elif not previous:
            yield 1, exc
        return
    yield 1, exc.__cause__ or exc


def structured_failure_diagnostics(exc: BaseException) -> list[dict[str, Any]]:
    """Return bounded, content-free diagnostics for structured failures.

    The result contains only attempt number, exception class, validation field
    paths/types/messages, and input Python types.  It must be safe for logs and
    the shadow decision's ``normalization_error`` field.
    """

    diagnostics: list[dict[str, Any]] = []
    for attempt, cause in _structured_attempt_causes(exc):
        diagnostics.append(
            {
                "attempt": attempt,
                "exception_type": type(cause).__name__,
                "validation_errors": _validation_error_details(cause),
            }
        )
    return diagnostics


def invoke_structured_only(
    structured_llm: Any | None,
    prompt: Any,
    agent_name: str,
    *,
    max_attempts: int = 1,
) -> BaseModel:
    """Invoke a structured LLM binding and fail closed on any miss.

    The helper never falls back to plain-text generation. Callers use it only
    when structured output is mandatory and any missing binding or invocation
    failure should be surfaced explicitly.  ``max_attempts`` is intentionally
    capped at two so a caller can permit one bounded retry without creating a
    recursive or unbounded retry path.
    """
    if max_attempts not in (1, 2):
        raise ValueError("structured output max_attempts must be 1 or 2")
    if structured_llm is None:
        raise StructuredOutputRequiredError(
            f"{agent_name}: structured output binding is required"
        )

    failures: list[StructuredOutputRequiredError] = []
    for attempt in range(max_attempts):
        try:
            result = structured_llm.invoke(prompt)
            if result is None:
                raise StructuredOutputRequiredError(
                    f"{agent_name}: structured output returned no parsed result"
                )
            if not isinstance(result, BaseModel):
                raise StructuredOutputRequiredError(
                    f"{agent_name}: structured output did not return a BaseModel"
                )
            return result
        except StructuredOutputRequiredError as exc:
            failures.append(exc)
        except Exception as exc:  # pragma: no cover - exercised in tests
            wrapped = StructuredOutputRequiredError(
                f"{agent_name}: structured output invocation failed"
            )
            wrapped.__cause__ = exc
            failures.append(wrapped)

        if attempt + 1 < max_attempts:
            logger.warning(
                "%s: structured-output attempt %d/%d failed; diagnostics=%s; retrying once",
                agent_name,
                attempt + 1,
                max_attempts,
                json.dumps(
                    structured_failure_diagnostics(failures[-1]),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )

    final = failures[-1]
    if len(failures) == 1:
        raise final
    cause = final.__cause__ or final
    retry_error = StructuredOutputRequiredError(
        f"{agent_name}: structured output invocation failed after {len(failures)} attempts",
        attempts=len(failures),
        previous_errors=tuple(failures[:-1]),
    )
    retry_error.__cause__ = cause
    logger.warning(
        "%s: structured-output failed after %d attempts; diagnostics=%s",
        agent_name,
        len(failures),
        json.dumps(
            structured_failure_diagnostics(retry_error),
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    raise retry_error from cause

# Schema-only structured output binds exactly one tool (the schema itself), so a
# model that reaches for a search tool emits an unknown tool call and the whole
# structured attempt is discarded for a free-text retry. Agents on this path
# state the constraint explicitly rather than relying on the binding alone
# (#1130).
NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)


def bind_structured(
    llm: Any,
    schema: type[T],
    agent_name: str,
    *,
    method: str | None = None,
    **kwargs: Any,
) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        if method is None:
            return llm.with_structured_output(schema, **kwargs)
        return llm.with_structured_output(schema, method=method, **kwargs)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def is_ollama_chat_model(llm: Any) -> bool:
    """Return whether ``llm`` is the Ollama OpenAI-compatible client."""
    return isinstance(llm, OllamaChatOpenAI)


def bind_forex_ollama_structured(
    llm: Any,
    schema: type[T],
    agent_name: str,
) -> Any | None:
    """Bind a forex schema using Ollama's strict JSON-schema request path.

    Ollama's OpenAI-compatible endpoint is unreliable when Qwen must choose a
    schema tool on production-sized prompts.  Its ``response_format`` JSON
    schema path is deterministic when thinking is disabled with the documented
    ``reasoning_effort=none`` request field.  Other providers retain the
    existing capability-selected binding and invocation semantics.
    """
    if not is_ollama_chat_model(llm):
        return bind_structured(llm, schema, agent_name)
    return bind_structured(
        llm,
        schema,
        agent_name,
        method="json_schema",
        reasoning_effort="none",
        # A forex deep client may be constructed with its normal thinking
        # default (``think=true``).  Ollama's JSON-schema path is reliable
        # only when the structured request explicitly disables thinking; the
        # per-binding override leaves ordinary deep analysis unchanged.
        extra_body={"think": False},
    )


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            if result is None:
                # A thinking model can answer in plain text instead of calling
                # the tool, leaving the parser with nothing to return. Treat it
                # as a structured miss and fall back, with a clear reason.
                raise ValueError("structured output returned no parsed result")
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content
