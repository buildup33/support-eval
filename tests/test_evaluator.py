from support_eval.evaluator import FakeEvaluator
from support_eval.models import Conversation, EvidenceReference, Message, Rubric, RubricCriterion


def test_fake_evaluator_is_deterministic_and_offline():
    conversation = Conversation(
        id="one", messages=[Message(role="agent", content="Hello, I understand and resolved it")]
    )
    rubric = Rubric(
        name="Test",
        criteria=[
            RubricCriterion(
                id="quality",
                name="Quality",
                description="Test quality",
                weight=1,
                pass_threshold=0.5,
                keywords=["hello", "understand", "missing"],
            )
        ],
    )
    evaluator = FakeEvaluator()
    first = evaluator.evaluate(conversation, rubric)
    second = evaluator.evaluate(conversation, rubric)
    assert first == second
    assert first.criteria[0].rating == 2 / 3
    assert first.criteria[0].evidence == [EvidenceReference(message_id="m0001")]


def test_fake_evaluator_only_scores_agent_messages():
    conversation = Conversation(
        id="one",
        messages=[
            Message(role="customer", content="Please refund this"),
            Message(role="agent", content="We will review it"),
        ],
    )
    rubric = Rubric(
        name="Test",
        criteria=[
            RubricCriterion(
                id="resolution",
                name="Resolution",
                description="Resolution offered by the agent",
                weight=1,
                pass_threshold=0.5,
                keywords=["refund"],
            )
        ],
    )
    result = FakeEvaluator().evaluate(conversation, rubric)
    assert result.criteria[0].rating == 0
    assert result.criteria[0].evidence == []
