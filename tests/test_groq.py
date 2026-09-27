import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import support_eval.groq as groq_module
from support_eval.groq import GroqError, GroqEvaluator
from support_eval.ingestion import load_conversations, load_rubric
from support_eval.models import (
    Conversation,
    LLMEvaluationResponse,
    Message,
    Rubric,
    RubricCriterion,
)
from support_eval.scoring import EvaluationValidationError, validate_evaluation

PROJECT_ROOT = Path(__file__).parents[1]


class FakeHTTPResponse:
    def __init__(self, payload: object, *, raw: bool = False):
        self.payload = payload if raw else json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.payload


def _configure(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-secret")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")


def _conversation():
    return Conversation(
        id="one",
        messages=[
            Message(role="customer", content="My order is broken"),
            Message(role="agent", content="I understand and arranged a replacement"),
        ],
    )


def _rubric():
    return Rubric(
        name="QA",
        criteria=[
            RubricCriterion(
                id="empathy",
                name="Empathy",
                description="Acknowledges the customer's experience",
                weight=1,
                pass_threshold=1,
                keywords=["understand"],
            )
        ],
    )


def _model_content(**overrides):
    criterion = {
        "criterion_id": "empathy",
        "result": "pass",
        "explanation": "The agent acknowledged the issue.",
        "evidence": [{"message_id": "m0002"}],
        "confidence": 0.95,
    }
    criterion.update(overrides)
    return json.dumps({"conversation_id": "one", "criteria": [criterion]})


def _api_response(content=None):
    return {
        "choices": [{"message": {"content": content if content is not None else _model_content()}}]
    }


def _http_error(status, payload, headers=None):
    return HTTPError(
        "safe-url",
        status,
        "failure",
        hdrs=headers or {},
        fp=BytesIO(json.dumps(payload).encode("utf-8")),
    )


def _benchmark_inputs():
    conversation = load_conversations(PROJECT_ROOT / "examples" / "benchmark-conversations.json")[0]
    rubric = load_rubric(PROJECT_ROOT / "examples" / "benchmark-rubric.json")
    return conversation, rubric


def _benchmark_model_content(conversation_id, criterion_ids):
    return json.dumps(
        {
            "conversation_id": conversation_id,
            "criteria": [
                {
                    "criterion_id": criterion_id,
                    "result": "pass",
                    "explanation": f"Fixture evaluation for {criterion_id}.",
                    "evidence": [{"message_id": "m0003"}],
                    "confidence": 0.9,
                }
                for criterion_id in criterion_ids
            ],
        }
    )


def test_groq_exact_request_and_successful_response(monkeypatch):
    _configure(monkeypatch)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeHTTPResponse(_api_response())

    monkeypatch.setattr(groq_module, "urlopen", fake_urlopen)
    result = GroqEvaluator().evaluate(_conversation(), _rubric())

    request = captured["request"]
    body = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "https://api.groq.com/openai/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer test-groq-secret"
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("User-agent") == "support-eval/0.1.0"
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["temperature"] == 0
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    user_payload = json.loads(body["messages"][1]["content"])
    assert "One applicable violated rule is enough for fail" in user_payload["instructions"]
    assert "all materially relevant factual claims" in user_payload["instructions"]
    assert "customer's current issue" in user_payload["instructions"]
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "support_evaluation"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["properties"]["conversation_id"]["enum"] == ["one"]
    assert schema["properties"]["criteria"]["minItems"] == 1
    assert schema["properties"]["criteria"]["maxItems"] == 1
    assert schema["$defs"]["LLMCriterionEvaluation"]["properties"]["criterion_id"]["enum"] == [
        "empathy"
    ]
    assert schema["$defs"]["EvidenceReference"]["properties"]["message_id"]["enum"] == [
        "m0001",
        "m0002",
    ]
    assert [message["message_id"] for message in user_payload["conversation"]["messages"]] == [
        "m0001",
        "m0002",
    ]
    assert captured["timeout"] == 30
    assert result.conversation_id == "one"
    assert result.criteria[0].criterion_id == "empathy"
    assert result.criteria[0].status == "pass"
    assert result.criteria[0].rating is None
    assert result.criteria[0].evidence[0].message_id == "m0002"
    assert result.criteria[0].confidence == 0.95


def test_groq_strict_schema_closes_all_objects():
    schema = LLMEvaluationResponse.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["conversation_id", "criteria"]
    criterion_schema = schema["$defs"]["LLMCriterionEvaluation"]
    assert criterion_schema["additionalProperties"] is False
    assert set(criterion_schema["required"]) == {
        "criterion_id",
        "result",
        "explanation",
        "evidence",
        "confidence",
    }
    evidence_schema = schema["$defs"]["EvidenceReference"]
    assert evidence_schema["required"] == ["message_id"]
    assert set(evidence_schema["properties"]) == {"message_id"}
    evidence_list_schema = criterion_schema["properties"]["evidence"]
    assert evidence_list_schema["maxItems"] == 3
    assert "minItems" not in evidence_list_schema


def test_groq_benchmark_request_requires_all_six_exact_criterion_ids(monkeypatch):
    _configure(monkeypatch)
    conversation, rubric = _benchmark_inputs()
    criterion_ids = [criterion.id for criterion in rubric.criteria]
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return FakeHTTPResponse(
            _api_response(_benchmark_model_content(conversation.id, criterion_ids))
        )

    monkeypatch.setattr(groq_module, "urlopen", fake_urlopen)
    result = GroqEvaluator().evaluate(conversation, rubric)

    body = json.loads(captured["request"].data.decode("utf-8"))
    schema = body["response_format"]["json_schema"]["schema"]
    criteria_schema = schema["properties"]["criteria"]
    criterion_id_schema = schema["$defs"]["LLMCriterionEvaluation"]["properties"]["criterion_id"]
    prompt = " ".join(message["content"] for message in body["messages"])

    assert criterion_ids == [
        "empathy",
        "professionalism",
        "accuracy",
        "resolution",
        "policy_compliance",
        "closing",
    ]
    assert criteria_schema["minItems"] == 6
    assert criteria_schema["maxItems"] == 6
    assert criterion_id_schema["enum"] == criterion_ids
    assert schema["properties"]["conversation_id"]["enum"] == [conversation.id]
    assert "Evaluate ALL 6 supplied criteria" in prompt
    assert "exactly one per criterion" in prompt
    assert "Do not omit criteria" in prompt
    assert "invent additional criteria" in prompt
    assert [item.criterion_id for item in result.criteria] == criterion_ids
    validate_evaluation(result, conversation, rubric)


@pytest.mark.parametrize(
    ("case", "returned_ids", "error_match"),
    [
        (
            "missing",
            ["empathy", "professionalism", "accuracy", "resolution", "policy_compliance"],
            "missing: closing",
        ),
        (
            "extra",
            [
                "empathy",
                "professionalism",
                "accuracy",
                "resolution",
                "policy_compliance",
                "closing",
                "invented",
            ],
            "unexpected: invented",
        ),
        (
            "duplicate",
            [
                "empathy",
                "professionalism",
                "accuracy",
                "resolution",
                "policy_compliance",
                "empathy",
            ],
            "duplicate criterion results",
        ),
    ],
)
def test_groq_benchmark_output_keeps_mismatch_safety_check(
    monkeypatch, case, returned_ids, error_match
):
    _configure(monkeypatch)
    conversation, rubric = _benchmark_inputs()
    monkeypatch.setattr(
        groq_module,
        "urlopen",
        lambda _request, timeout: FakeHTTPResponse(
            _api_response(_benchmark_model_content(conversation.id, returned_ids))
        ),
    )

    result = GroqEvaluator().evaluate(conversation, rubric)

    with pytest.raises(EvaluationValidationError, match=error_match):
        validate_evaluation(result, conversation, rubric)


def test_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GroqError, match="GROQ_API_KEY is required"):
        GroqEvaluator()


def test_groq_rate_limit_uses_retry_after_with_safety_margin(monkeypatch):
    _configure(monkeypatch)
    calls = 0
    delays = []
    messages = []

    def rate_limit_then_succeed(_request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(
                429,
                {"error": {"message": "Token rate limit reached", "type": "tokens"}},
                {"Retry-After": "5.68"},
            )
        return FakeHTTPResponse(_api_response())

    monkeypatch.setattr(groq_module, "urlopen", rate_limit_then_succeed)
    result = GroqEvaluator(sleep=delays.append, notify=messages.append).evaluate(
        _conversation(), _rubric()
    )

    assert result.conversation_id == "one"
    assert calls == 2
    assert delays == [6.18]
    assert messages == ["Groq rate limit reached; retrying in 6.18 seconds..."]


def test_groq_rate_limit_uses_safe_body_delay_when_header_is_absent(monkeypatch):
    _configure(monkeypatch)
    calls = 0
    delays = []

    def rate_limit_then_succeed(_request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(
                429,
                {
                    "error": {
                        "message": "TPM exceeded. Please try again in 5.68 seconds.",
                        "type": "tokens",
                    }
                },
            )
        return FakeHTTPResponse(_api_response())

    monkeypatch.setattr(groq_module, "urlopen", rate_limit_then_succeed)
    GroqEvaluator(sleep=delays.append, notify=lambda _message: None).evaluate(
        _conversation(), _rubric()
    )

    assert calls == 2
    assert delays == [6.18]


def test_groq_rate_limit_falls_back_to_bounded_exponential_backoff(monkeypatch):
    _configure(monkeypatch)
    calls = 0
    delays = []

    def rate_limit_then_succeed(_request, timeout):
        nonlocal calls
        calls += 1
        if calls < 4:
            raise _http_error(
                429,
                {"error": {"message": "Rate limit reached", "type": "tokens"}},
            )
        return FakeHTTPResponse(_api_response())

    monkeypatch.setattr(groq_module, "urlopen", rate_limit_then_succeed)
    GroqEvaluator(sleep=delays.append, notify=lambda _message: None).evaluate(
        _conversation(), _rubric()
    )

    assert calls == 4
    assert delays == [1, 2, 4]


def test_groq_rate_limit_stops_after_maximum_attempts(monkeypatch):
    _configure(monkeypatch)
    calls = 0
    delays = []

    def always_rate_limited(_request, timeout):
        nonlocal calls
        calls += 1
        raise _http_error(
            429,
            {
                "error": {
                    "message": "Token quota is still exhausted",
                    "type": "tokens",
                    "code": "rate_limit_exceeded",
                }
            },
        )

    monkeypatch.setattr(groq_module, "urlopen", always_rate_limited)
    evaluator = GroqEvaluator(sleep=delays.append, notify=lambda _message: None)

    with pytest.raises(GroqError) as error:
        evaluator.evaluate(_conversation(), _rubric())

    assert calls == 4
    assert delays == [1, 2, 4]
    assert "HTTP 429" in str(error.value)
    assert "rate_limit_exceeded" in str(error.value)
    assert "Token quota is still exhausted" in str(error.value)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_groq_does_not_retry_non_retryable_client_errors(monkeypatch, status):
    _configure(monkeypatch)
    calls = 0
    delays = []

    def fail_request(_request, timeout):
        nonlocal calls
        calls += 1
        raise _http_error(
            status,
            {"error": {"message": "Do not retry", "type": "request_error"}},
        )

    monkeypatch.setattr(groq_module, "urlopen", fail_request)
    evaluator = GroqEvaluator(sleep=delays.append, notify=lambda _message: None)

    with pytest.raises(GroqError):
        evaluator.evaluate(_conversation(), _rubric())

    assert calls == 1
    assert delays == []


def test_groq_retries_transient_server_error(monkeypatch):
    _configure(monkeypatch)
    calls = 0
    delays = []

    def unavailable_then_succeed(_request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(
                503,
                {"error": {"message": "Temporarily unavailable", "type": "server_error"}},
            )
        return FakeHTTPResponse(_api_response())

    monkeypatch.setattr(groq_module, "urlopen", unavailable_then_succeed)
    result = GroqEvaluator(sleep=delays.append, notify=lambda _message: None).evaluate(
        _conversation(), _rubric()
    )

    assert result.conversation_id == "one"
    assert calls == 2
    assert delays == [1]


def test_groq_retries_transient_network_error_then_succeeds(monkeypatch):
    _configure(monkeypatch)
    calls = 0
    delays = []

    def offline_then_succeed(_request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise URLError("temporary connection failure")
        return FakeHTTPResponse(_api_response())

    monkeypatch.setattr(groq_module, "urlopen", offline_then_succeed)
    result = GroqEvaluator(sleep=delays.append, notify=lambda _message: None).evaluate(
        _conversation(), _rubric()
    )

    assert result.conversation_id == "one"
    assert calls == 2
    assert delays == [1]


@pytest.mark.parametrize(
    ("status", "prefix"),
    [
        (400, "request rejected"),
        (401, "authentication failed"),
        (403, "access forbidden"),
        (404, "endpoint or model not found"),
        (413, "request too large"),
        (422, "could not be processed"),
        (429, "rate limit exceeded"),
        (500, "server error"),
    ],
)
def test_groq_distinguishes_common_http_errors(monkeypatch, status, prefix):
    _configure(monkeypatch)
    payload = {
        "error": {
            "message": "Safe Groq error detail",
            "type": "request_error",
            "code": "safe_error_code",
        }
    }

    def fail_request(_request, timeout):
        assert timeout == 30
        raise HTTPError(
            "safe-url",
            status,
            "failure",
            hdrs=None,
            fp=BytesIO(json.dumps(payload).encode("utf-8")),
        )

    monkeypatch.setattr(groq_module, "urlopen", fail_request)
    with pytest.raises(GroqError) as error:
        GroqEvaluator(sleep=lambda _delay: None, notify=lambda _message: None).evaluate(
            _conversation(), _rubric()
        )
    message = str(error.value)
    assert prefix in message
    assert f"HTTP {status}" in message
    assert "request_error" in message
    assert "safe_error_code" in message
    assert "Safe Groq error detail" in message


def test_groq_redacts_credentials_from_errors(monkeypatch):
    _configure(monkeypatch)
    long_token = "abcdefghijklmnopqrstuvwxyz1234567890TOKEN"
    payload = {
        "error": {
            "message": (
                "Authorization: Bearer test-groq-secret; "
                "api_key=test-groq-secret; credential=test-groq-secret; "
                f"token={long_token}"
            ),
            "type": "authentication_error",
        }
    }

    def fail_request(_request, timeout):
        assert timeout == 30
        raise HTTPError(
            "safe-url",
            401,
            "failure",
            hdrs=None,
            fp=BytesIO(json.dumps(payload).encode("utf-8")),
        )

    monkeypatch.setattr(groq_module, "urlopen", fail_request)
    with pytest.raises(GroqError) as error:
        GroqEvaluator().evaluate(_conversation(), _rubric())
    message = str(error.value)
    assert "test-groq-secret" not in message
    assert long_token not in message
    assert "[REDACTED]" in message


@pytest.mark.parametrize("error", [TimeoutError(), URLError(TimeoutError())])
def test_groq_handles_timeouts(monkeypatch, error):
    _configure(monkeypatch)

    def fail_request(_request, timeout):
        assert timeout == 30
        raise error

    monkeypatch.setattr(groq_module, "urlopen", fail_request)
    with pytest.raises(GroqError, match="timed out"):
        GroqEvaluator(sleep=lambda _delay: None, notify=lambda _message: None).evaluate(
            _conversation(), _rubric()
        )


def test_groq_handles_network_error(monkeypatch):
    _configure(monkeypatch)

    def fail_request(_request, timeout):
        assert timeout == 30
        raise URLError("offline")

    monkeypatch.setattr(groq_module, "urlopen", fail_request)
    with pytest.raises(GroqError, match="Could not reach"):
        GroqEvaluator(sleep=lambda _delay: None, notify=lambda _message: None).evaluate(
            _conversation(), _rubric()
        )


def test_groq_rejects_malformed_api_response(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(
        groq_module,
        "urlopen",
        lambda _request, timeout: FakeHTTPResponse(b"not-json", raw=True),
    )
    with pytest.raises(GroqError, match="malformed API response"):
        GroqEvaluator().evaluate(_conversation(), _rubric())


def test_groq_rejects_missing_completion_content(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(
        groq_module,
        "urlopen",
        lambda _request, timeout: FakeHTTPResponse({"choices": []}),
    )
    with pytest.raises(GroqError, match="missing chat completion content"):
        GroqEvaluator().evaluate(_conversation(), _rubric())


def test_groq_rejects_malformed_model_json(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(
        groq_module,
        "urlopen",
        lambda _request, timeout: FakeHTTPResponse(_api_response("not-json")),
    )
    with pytest.raises(GroqError, match="malformed JSON content"):
        GroqEvaluator().evaluate(_conversation(), _rubric())


def test_groq_rejects_structured_output_validation_failure(monkeypatch):
    _configure(monkeypatch)
    content = json.loads(_model_content())
    del content["criteria"][0]["confidence"]
    monkeypatch.setattr(
        groq_module,
        "urlopen",
        lambda _request, timeout: FakeHTTPResponse(_api_response(json.dumps(content))),
    )
    with pytest.raises(GroqError, match="invalid evaluation fields"):
        GroqEvaluator().evaluate(_conversation(), _rubric())
