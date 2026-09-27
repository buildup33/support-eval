import json

import pytest

from support_eval.ingestion import InputError, load_conversations


def test_loads_and_groups_csv_messages(tmp_path):
    path = tmp_path / "conversations.csv"
    path.write_text(
        "conversation_id,role,message,timestamp\n"
        "one,customer,Hello,2026-01-01T00:00:00Z\n"
        "one,agent,Hi there,2026-01-01T00:01:00Z\n"
        "two,customer,Help,\n",
        encoding="utf-8",
    )
    conversations = load_conversations(path)
    assert [item.id for item in conversations] == ["one", "two"]
    assert len(conversations[0].messages) == 2
    assert conversations[0].messages[1].role == "agent"


def test_loads_json_conversations(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_text(
        json.dumps(
            {
                "conversations": [
                    {
                        "id": "json-one",
                        "messages": [{"role": "customer", "content": "Need help"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_conversations(path)[0].id == "json-one"


def test_rejects_duplicate_json_conversation_ids(tmp_path):
    path = tmp_path / "duplicate-conversations.json"
    path.write_text(
        json.dumps(
            {
                "conversations": [
                    {
                        "id": "duplicate-id",
                        "messages": [{"role": "customer", "content": "First"}],
                    },
                    {
                        "id": "duplicate-id",
                        "messages": [{"role": "customer", "content": "Second"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="Duplicate conversation id 'duplicate-id'"):
        load_conversations(path)


def test_rejects_csv_without_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,text\none,hello\n", encoding="utf-8")
    with pytest.raises(InputError, match="missing required columns"):
        load_conversations(path)
