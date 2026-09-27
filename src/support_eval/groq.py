from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from support_eval import __version__
from support_eval.llm_instructions import build_evaluation_instructions, build_system_prompt
from support_eval.models import (
    Conversation,
    CriterionEvaluation,
    EvaluationResult,
    LLMEvaluationResponse,
    Rubric,
)
from support_eval.transcript import llm_conversation_payload, transcript_messages

CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = f"support-eval/{__version__}"
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 8.0
SERVER_DELAY_SAFETY_MARGIN_SECONDS = 0.5
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


class GroqError(RuntimeError):
    """Safe, user-facing failure from Groq configuration or communication."""


class GroqEvaluator:
    """Evaluate conversations with Groq's OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        sleep: Callable[[float], None] = time.sleep,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self._api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self._model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL).strip()
        self._sleep = sleep
        self._notify = notify or self._notify_stderr
        if not self._api_key:
            raise GroqError("GROQ_API_KEY is required when using the Groq evaluator")
        if not self._model:
            raise GroqError("GROQ_MODEL cannot be empty")

    @property
    def provider_id(self) -> str:
        return "groq"

    @property
    def model_id(self) -> str:
        return self._model

    def evaluate(self, conversation: Conversation, rubric: Rubric) -> EvaluationResult:
        request = self._build_request(conversation, rubric)
        raw_response = self._send(request)
        model_content = self._extract_content(raw_response)
        try:
            model_payload = json.loads(model_content)
        except json.JSONDecodeError as exc:
            raise GroqError("Groq returned malformed JSON content") from exc
        try:
            validated = LLMEvaluationResponse.model_validate(model_payload)
        except ValidationError as exc:
            raise GroqError(f"Groq returned invalid evaluation fields: {exc}") from exc
        return EvaluationResult(
            conversation_id=validated.conversation_id,
            criteria=[
                CriterionEvaluation(
                    criterion_id=item.criterion_id,
                    status=item.result,
                    rationale=item.explanation,
                    evidence=item.evidence,
                    confidence=item.confidence,
                )
                for item in validated.criteria
            ],
        )

    def _build_request(self, conversation: Conversation, rubric: Rubric) -> Request:
        criterion_ids = [criterion.id for criterion in rubric.criteria]
        user_payload = {
            "instructions": build_evaluation_instructions(criterion_ids),
            "rubric": rubric.model_dump(),
            "conversation": llm_conversation_payload(conversation),
        }
        body = json.dumps(
            {
                "model": self._model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": build_system_prompt(criterion_ids)},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "support_evaluation",
                        "strict": True,
                        "schema": self._response_schema(conversation, rubric),
                    },
                },
            }
        ).encode("utf-8")
        return Request(
            CHAT_COMPLETIONS_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )

    @staticmethod
    def _response_schema(conversation: Conversation, rubric: Rubric) -> dict[str, object]:
        schema = LLMEvaluationResponse.model_json_schema()
        criterion_ids = [criterion.id for criterion in rubric.criteria]

        conversation_id_schema = schema["properties"]["conversation_id"]
        conversation_id_schema["enum"] = [conversation.id]

        criteria_schema = schema["properties"]["criteria"]
        criteria_schema["minItems"] = len(criterion_ids)
        criteria_schema["maxItems"] = len(criterion_ids)

        criterion_id_schema = schema["$defs"]["LLMCriterionEvaluation"]["properties"][
            "criterion_id"
        ]
        criterion_id_schema["enum"] = criterion_ids

        valid_message_ids = [message["message_id"] for message in transcript_messages(conversation)]
        evidence_schema = schema["$defs"]["EvidenceReference"]
        evidence_schema["properties"]["message_id"]["enum"] = valid_message_ids
        return schema

    def _send(self, request: Request) -> object:
        for attempt in range(MAX_ATTEMPTS):
            try:
                with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                raw_body = self._read_http_error_body(exc)
                if exc.code in RETRYABLE_HTTP_STATUSES and attempt < MAX_ATTEMPTS - 1:
                    delay = self._retry_delay(exc, raw_body, attempt)
                    if exc.code == 429:
                        message = f"Groq rate limit reached; retrying in {delay:g} seconds..."
                    else:
                        message = f"Groq service unavailable; retrying in {delay:g} seconds..."
                    self._notify(message)
                    self._sleep(delay)
                    continue
                raise GroqError(self._format_http_error(exc, raw_body)) from exc
            except TimeoutError as exc:
                if attempt < MAX_ATTEMPTS - 1:
                    self._retry_transient_network_error(attempt)
                    continue
                raise GroqError("Groq request timed out after maximum retry attempts") from exc
            except URLError as exc:
                if attempt < MAX_ATTEMPTS - 1:
                    self._retry_transient_network_error(attempt)
                    continue
                if isinstance(exc.reason, TimeoutError):
                    raise GroqError("Groq request timed out after maximum retry attempts") from exc
                raise GroqError(
                    "Could not reach the Groq API after maximum retry attempts"
                ) from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GroqError("Groq returned a malformed API response") from exc
        raise GroqError("Groq request failed after maximum retry attempts")

    def _retry_transient_network_error(self, attempt: int) -> None:
        delay = self._fallback_delay(attempt)
        self._notify(f"Groq network error; retrying in {delay:g} seconds...")
        self._sleep(delay)

    def _retry_delay(self, error: HTTPError, raw_body: str, attempt: int) -> float:
        server_delay = self._retry_after_header(error)
        if server_delay is None:
            server_delay = self._retry_delay_from_body(raw_body)
        if server_delay is not None:
            return server_delay + SERVER_DELAY_SAFETY_MARGIN_SECONDS
        return self._fallback_delay(attempt)

    @staticmethod
    def _retry_after_header(error: HTTPError) -> float | None:
        headers = error.headers
        value = headers.get("Retry-After") if headers is not None else None
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())

    @staticmethod
    def _retry_delay_from_body(raw_body: str) -> float | None:
        if not raw_body.strip():
            return None
        messages: list[str] = []
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            messages.append(raw_body)
        else:
            candidates = [payload]
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                candidates.append(payload["error"])
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                for key in ("retry_after", "retry_after_seconds"):
                    value = candidate.get(key)
                    if isinstance(value, int | float) and value >= 0:
                        return float(value)
                message = candidate.get("message")
                if isinstance(message, str):
                    messages.append(message)

        for message in messages:
            match = re.search(
                r"(?i)(?:retry|try again)\s+(?:after|in)\s+"
                r"(\d+(?:\.\d+)?)\s*(milliseconds?|ms|seconds?|secs?|s)\b",
                message,
            )
            if not match:
                continue
            delay = float(match.group(1))
            if match.group(2).casefold().startswith(("millisecond", "ms")):
                delay /= 1000
            return delay
        return None

    @staticmethod
    def _fallback_delay(attempt: int) -> float:
        return min(BACKOFF_BASE_SECONDS * (2**attempt), BACKOFF_MAX_SECONDS)

    @staticmethod
    def _notify_stderr(message: str) -> None:
        print(message, file=sys.stderr)

    @staticmethod
    def _read_http_error_body(error: HTTPError) -> str:
        try:
            return error.read().decode("utf-8", errors="replace")
        except (AttributeError, OSError):
            return ""

    def _format_http_error(self, error: HTTPError, raw_body: str | None = None) -> str:
        if error.code == 400:
            prefix = "Groq request rejected"
        elif error.code == 401:
            prefix = "Groq authentication failed"
        elif error.code == 403:
            prefix = "Groq access forbidden"
        elif error.code == 404:
            prefix = "Groq endpoint or model not found"
        elif error.code == 413:
            prefix = "Groq request too large"
        elif error.code == 422:
            prefix = "Groq request could not be processed"
        elif error.code == 429:
            prefix = "Groq rate limit exceeded"
        elif error.code >= 500:
            prefix = "Groq server error"
        else:
            prefix = "Groq API request failed"

        identifiers, detail = self._read_error_details(error, raw_body)
        context = [f"HTTP {error.code}", *identifiers]
        formatted = f"{prefix} ({' / '.join(context)})"
        if detail:
            formatted += f": {detail}"
        return formatted

    def _read_error_details(
        self, error: HTTPError, raw_body: str | None = None
    ) -> tuple[list[str], str | None]:
        if raw_body is None:
            raw_body = self._read_http_error_body(error)
        if not raw_body.strip():
            return [], None
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return [], self._redact(raw_body)

        error_data = payload.get("error", payload) if isinstance(payload, dict) else payload
        identifiers: list[str] = []
        message: object = None
        if isinstance(error_data, dict):
            for name in ("status", "type", "code", "reason"):
                identifier = self._safe_identifier(error_data.get(name))
                if identifier and identifier not in identifiers:
                    identifiers.append(identifier)
            message = error_data.get("message") or error_data.get("detail")
        elif isinstance(error_data, str):
            message = error_data
        if not isinstance(message, str) or not message.strip():
            message = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        return identifiers, self._redact(message)

    @staticmethod
    def _safe_identifier(value: object) -> str | None:
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
            return value
        return None

    def _redact(self, value: str) -> str | None:
        text = " ".join(value.split())
        if not text:
            return None
        text = text.replace(self._api_key, "[REDACTED]")
        text = re.sub(r"(?i)\bBearer\s+[^\s,;\"']+", "Bearer [REDACTED]", text)
        text = re.sub(
            r"(?i)(authorization|api[_ -]?key|token|credential|secret)"
            r"(\s*[:=]\s*[\"']?)[^\s,;\"'}]+",
            r"\1\2[REDACTED]",
            text,
        )
        text = re.sub(
            r"(?<![A-Za-z0-9])[A-Za-z0-9_+/.=-]{32,}(?![A-Za-z0-9])",
            "[REDACTED]",
            text,
        )
        return text[:1000]

    @staticmethod
    def _extract_content(response: object) -> str:
        try:
            content = response["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise GroqError("Groq response is missing chat completion content") from exc
        if not isinstance(content, str) or not content.strip():
            raise GroqError("Groq response has empty chat completion content")
        return content
