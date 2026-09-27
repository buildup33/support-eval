from support_eval.models import Conversation, Message
from support_eval.transcript import llm_conversation_payload, transcript_messages


def test_message_ids_are_stable_and_preserve_normalized_message_order():
    conversation = Conversation(
        id="stable-ids",
        messages=[
            Message(role="customer", content="First"),
            Message(role="system", content="Second"),
            Message(role="agent", content="Third"),
        ],
    )

    first = transcript_messages(conversation)
    second = transcript_messages(conversation.model_copy(deep=True))

    assert [message["message_id"] for message in first] == ["m0001", "m0002", "m0003"]
    assert first == second
    assert llm_conversation_payload(conversation)["messages"] == first
