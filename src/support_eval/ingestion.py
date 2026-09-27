from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from support_eval.models import Conversation, Message, Rubric


class InputError(ValueError):
    """Raised when a local input file cannot be normalized or validated."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"Could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON in {path}: {exc}") from exc


def load_conversations(path: str | Path) -> list[Conversation]:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        return _load_csv_conversations(source)
    if suffix == ".json":
        return _load_json_conversations(source)
    raise InputError(f"Unsupported conversation format '{source.suffix}'; use .csv or .json")


def _load_csv_conversations(path: Path) -> list[Conversation]:
    grouped: OrderedDict[str, list[Message]] = OrderedDict()
    required = {"conversation_id", "role", "message"}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = set(reader.fieldnames or [])
            missing = required - headers
            if missing:
                raise InputError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
            for row_number, row in enumerate(reader, start=2):
                conversation_id = (row.get("conversation_id") or "").strip()
                if not conversation_id:
                    raise InputError(f"CSV row {row_number} has an empty conversation_id")
                try:
                    message = Message(
                        role=(row.get("role") or "").strip().casefold(),
                        content=(row.get("message") or "").strip(),
                        timestamp=(row.get("timestamp") or "").strip() or None,
                    )
                except ValidationError as exc:
                    raise InputError(f"Invalid message on CSV row {row_number}: {exc}") from exc
                grouped.setdefault(conversation_id, []).append(message)
    except OSError as exc:
        raise InputError(f"Could not read {path}: {exc}") from exc
    if not grouped:
        raise InputError("Conversation file contains no messages")
    return [
        Conversation(id=conversation_id, messages=messages)
        for conversation_id, messages in grouped.items()
    ]


def _load_json_conversations(path: Path) -> list[Conversation]:
    payload = _read_json(path)
    records = payload.get("conversations") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise InputError("JSON must be a non-empty list or an object with a conversations list")
    try:
        conversations = [Conversation.model_validate(record) for record in records]
    except ValidationError as exc:
        raise InputError(f"Invalid conversation JSON: {exc}") from exc
    seen: set[str] = set()
    for conversation in conversations:
        if conversation.id in seen:
            raise InputError(f"Duplicate conversation id '{conversation.id}' in JSON input")
        seen.add(conversation.id)
    return conversations


def load_rubric(path: str | Path) -> Rubric:
    source = Path(path)
    if source.suffix.casefold() != ".json":
        raise InputError("Rubric must be a JSON file")
    payload = _read_json(source)
    try:
        return Rubric.model_validate(payload)
    except ValidationError as exc:
        raise InputError(f"Invalid rubric in {source}: {exc}") from exc
