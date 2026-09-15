from types import SimpleNamespace

from tradingagents.distillation.teacher import FakeTeacher, teacher_from_environment


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
    from tradingagents.distillation.models import TeacherConfig

    config = TeacherConfig(provider="fixture", model="teacher-v1", max_tokens=32)
    teacher = FakeTeacher({"lesson": "grounded"}, config=config)
    assert teacher.config == config
    assert teacher.generate(object(), config).provider == "fake"
