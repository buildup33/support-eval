from __future__ import annotations

from typing import Any

from support_eval.models import Conversation


def message_id(index: int) -> str:
    return f"m{index:04d}"


def transcript_messages(conversation: Conversation) -> list[dict[str, Any]]:
    return [
        {
            "message_id": message_id(index),
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,
        }
        for index, message in enumerate(conversation.messages, start=1)
    ]


def transcript_message_map(conversation: Conversation) -> dict[str, str]:
    return {
        message_id(index): message.content
        for index, message in enumerate(conversation.messages, start=1)
    }


def llm_conversation_payload(conversation: Conversation) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "messages": transcript_messages(conversation),
        "metadata": conversation.metadata,
    }
