from __future__ import annotations

import json

import pytest

from tradingagents.finetuning.formatting import SFTFormatter
from tradingagents.finetuning.sequence import SequenceTooLongError
from tradingagents.finetuning.tokenization import TokenizationPolicy, tokenize_example
from tests.test_phase11_formatting import row


class SpyTokenizer:
    model_max_length = 1000
    name_or_path = "spy"
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        assert kwargs.get("truncation") is not True
        value = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return self.encode(value, add_special_tokens=False, truncation=False) if tokenize else value

    def encode(self, value, **kwargs):
        assert kwargs.get("truncation") is not True
        return list(range(len(value)))


def test_native_template_and_assistant_only_labels() -> None:
    ex = SFTFormatter().format(row())
    result = tokenize_example(ex, TokenizationPolicy(tokenizer=SpyTokenizer(), max_length=1000))
    assert len(result.input_ids) == len(result.labels)
    assert any(x != -100 for x in result.labels)
    assert result.labels[0] == -100


def test_missing_template_requires_explicit_fallback() -> None:
    class NoTemplate(SpyTokenizer):
        chat_template = None

    ex = SFTFormatter().format(row())
    with pytest.raises(Exception):
        tokenize_example(ex, TokenizationPolicy(tokenizer=NoTemplate(), max_length=100))
    result = tokenize_example(ex, TokenizationPolicy(tokenizer=NoTemplate(), max_length=1000, template="fallback-v1"))
    assert result.input_ids


def test_over_capacity_rejected_without_truncation() -> None:
    ex = SFTFormatter().format(row())
    with pytest.raises(SequenceTooLongError):
        tokenize_example(ex, TokenizationPolicy(tokenizer=SpyTokenizer(), max_length=10))


def test_structural_bundles_are_preserved_and_split() -> None:
    ex = SFTFormatter().format(row())
    context = {"current_state": {"symbol": "EURUSD"}, "tables": [{"id": "t1", "header": ["a"], "caption": "c", "rows": [[str(i)] for i in range(20)]}], "equations": [{"id": "e1", "expression": "x=y", "definition": "x"}], "used_evidence": ["K1"], "rejected_evidence": ["K2"]}
    ex = ex.__class__(ex.example_id, ex.split, tuple((*ex.messages[:1], {"role": "user", "content": json.dumps(context)} , ex.messages[-1])), ex.target)
    result = tokenize_example(ex, TokenizationPolicy(tokenizer=SpyTokenizer(), max_length=500))
    assert result.reduction is not None
    assert result.reduction["current_state"] == context["current_state"]
    assert result.reduction["used_evidence"] == context["used_evidence"]
