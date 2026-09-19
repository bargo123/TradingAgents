import json
from types import SimpleNamespace

import httpx

from tests.fixtures.phase11a_knowledge import fixture_source
from tradingagents.distillation.models import TeacherConfig
from tradingagents.distillation.planning import PlannerConfig, SourcePacketPlanner
from tradingagents.distillation.teacher import (
    FakeTeacher,
    OllamaTeacher,
    TeacherLesson,
    teacher_from_environment,
)


def test_fake_teacher_is_deterministic_and_records_no_network_data():
    packet = SimpleNamespace(packet_id="p1")
    teacher = FakeTeacher({"lesson": "grounded"})
    result = teacher.generate(packet, SimpleNamespace(max_tokens=10))
    assert result.ok and result.candidate == {"lesson": "grounded"}
    assert teacher.calls == [packet]


def test_unconfigured_teacher_fails_closed():
    result = teacher_from_environment({}).generate(object(), object())
    assert result.error_code == "DISTILLATION_TEACHER_NOT_CONFIGURED"


def test_fake_teacher_reports_bounded_failure():
    result = FakeTeacher(error="timeout").generate(object(), object())
    assert not result.ok and result.error_code == "TEACHER_FAILED"


def test_fake_teacher_exposes_explicit_reproducibility_config():
    config = TeacherConfig(provider="fixture", model="teacher-v1", max_tokens=32)
    teacher = FakeTeacher({"lesson": "grounded"}, config=config)
    assert teacher.config == config
    assert teacher.generate(object(), config).provider == "fake"


class _Structured:
    def __init__(self, result):
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _FakeLLM:
    def __init__(self, result):
        self.result = result
        self.schema = None
        self.kwargs = None
        self.structured = _Structured(result)

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.kwargs = kwargs
        return self.structured


def _plan_one():
    return SourcePacketPlanner.plan(
        fixture_source(), (), PlannerConfig(max_blocks=1, max_chars=2_000, max_estimated_tokens=400)
    )


def _lesson(packet):
    ref = packet.refs[0]
    return TeacherLesson(
        lesson_type="DEFINITION",
        topic="order flow imbalance",
        difficulty="FOUNDATIONAL",
        system_instruction="Explain only the supplied source evidence.",
        user_instruction="What does this concept mean and what limitation is stated?",
        assistant_target="It describes a relationship in the supplied market-microstructure evidence.",
        source_refs=[ref.chunk_id],
        grounding_claims=[
            {
                "claim": "The supplied source describes the concept.",
                "source_refs": [ref.chunk_id],
            }
        ],
    )


def test_ollama_teacher_uses_strict_json_schema_and_exact_packet_refs():
    plan = _plan_one()
    packet = plan.packets[0]
    fake = _FakeLLM(_lesson(packet))
    teacher = OllamaTeacher(
        TeacherConfig(provider="ollama", model="qwen3.5:4b", max_tokens=512),
        client_factory=lambda config, endpoint: fake,
    )

    result = teacher.generate(packet, teacher.config)

    assert result.ok
    assert result.provider == "ollama"
    assert result.model == "qwen3.5:4b"
    assert result.candidate["source_refs"] == [packet.refs[0].chunk_id]
    assert fake.schema is TeacherLesson
    assert fake.kwargs["method"] == "json_schema"
    assert fake.kwargs["reasoning_effort"] == "none"
    assert fake.kwargs["extra_body"] == {"think": False}
    assert packet.refs[0].chunk_id in str(fake.structured.messages)
    assert "prompt" not in result.diagnostics
    assert "completion" not in result.diagnostics


def test_ollama_teacher_actual_client_sends_json_schema_payload():
    packet = _plan_one().packets[0]
    calls = []

    def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        return httpx.Response(
            200,
            json={
                "id": "phase11a-teacher-test",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.5:4b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": _lesson(packet).model_dump_json(),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
            },
            request=request,
        )

    def client_factory(config, endpoint):
        from tradingagents.llm_clients.openai_client import OllamaChatOpenAI

        return OllamaChatOpenAI(
            model=config.model,
            api_key="ollama",
            base_url=endpoint,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    teacher = OllamaTeacher(
        TeacherConfig(provider="ollama", model="qwen3.5:4b", max_tokens=512),
        client_factory=client_factory,
    )
    result = teacher.generate(packet, teacher.config)

    assert result.ok
    assert len(calls) == 1
    body = calls[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "TeacherLesson"
    assert body["reasoning_effort"] == "none"
    assert body["max_tokens"] == 512
    assert body["think"] is False
    assert "tools" not in body


def test_ollama_teacher_fails_closed_on_schema_invalid_response():
    packet = _plan_one().packets[0]
    fake = _FakeLLM({"unexpected": "field"})
    teacher = OllamaTeacher(
        TeacherConfig(provider="ollama", model="qwen3.5:4b"),
        client_factory=lambda config, endpoint: fake,
    )

    result = teacher.generate(packet, teacher.config)

    assert not result.ok
    assert result.error_code == "SCHEMA_INVALID"
    assert result.candidate is None


def test_ollama_teacher_rejects_unknown_source_reference():
    packet = _plan_one().packets[0]
    candidate = _lesson(packet).model_copy(
        update={"source_refs": ["not-a-packet-ref"], "grounding_claims": []}
    )
    fake = _FakeLLM(candidate)
    teacher = OllamaTeacher(
        TeacherConfig(provider="ollama", model="qwen3.5:4b"),
        client_factory=lambda config, endpoint: fake,
    )

    result = teacher.generate(packet, teacher.config)

    assert not result.ok
    assert result.error_code == "GROUNDING_FAILED"


def test_teacher_environment_is_opt_in_and_does_not_construct_ollama_without_provider():
    teacher = teacher_from_environment({})
    assert teacher.__class__.__name__ == "UnconfiguredTeacher"


def test_teacher_environment_rejects_non_ollama_provider():
    teacher = teacher_from_environment(
        {"PHASE11A_TEACHER_PROVIDER": "openai", "PHASE11A_TEACHER_MODEL": "gpt"}
    )
    assert teacher.__class__.__name__ == "UnconfiguredTeacher"
