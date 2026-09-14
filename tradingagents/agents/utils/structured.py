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

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from tradingagents.llm_clients.openai_client import OllamaChatOpenAI

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputRequiredError(RuntimeError):
    """Raised when a structured-output binding is required but unavailable."""


def structured_failure_category(exc: BaseException) -> str:
    """Return a safe type-only category for a structured-output failure.

    The underlying provider exception is chained for local diagnostics, but
    its message can contain request details.  Callers that persist a failure
    should expose only this bounded category, never the provider text or model
    output.
    """

    cause = exc.__cause__
    return type(cause if cause is not None else exc).__name__


def invoke_structured_only(
    structured_llm: Any | None,
    prompt: Any,
    agent_name: str,
) -> BaseModel:
    """Invoke a structured LLM binding and fail closed on any miss.

    The helper never falls back to plain-text generation. Callers use it only
    when structured output is mandatory and any missing binding or invocation
    failure should be surfaced explicitly.
    """
    if structured_llm is None:
        raise StructuredOutputRequiredError(
            f"{agent_name}: structured output binding is required"
        )

    try:
        result = structured_llm.invoke(prompt)
    except Exception as exc:  # pragma: no cover - exercised in tests
        raise StructuredOutputRequiredError(
            f"{agent_name}: structured output invocation failed"
        ) from exc

    if result is None:
        raise StructuredOutputRequiredError(
            f"{agent_name}: structured output returned no parsed result"
        )
    if not isinstance(result, BaseModel):
        raise StructuredOutputRequiredError(
            f"{agent_name}: structured output did not return a BaseModel"
        )
    return result

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
