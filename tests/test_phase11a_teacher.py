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
