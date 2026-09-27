from support_eval.llm_instructions import build_evaluation_instructions, build_system_prompt


def test_general_llm_instructions_cover_all_evaluation_boundaries():
    criterion_ids = ["conduct", "truthfulness", "outcome", "handoff"]
    system_prompt = build_system_prompt(criterion_ids)
    instructions = build_evaluation_instructions(criterion_ids)

    assert "Evaluate ALL 4 supplied criteria" in system_prompt
    assert "do not reveal chain-of-thought" in system_prompt
    assert "concise rationale" in system_prompt
    assert "One applicable violated rule is enough for fail" in instructions
    assert "irrelevant clauses do not affect the result" in instructions
    assert "all materially relevant factual claims" in instructions
    assert "materially weakest unsupported claim" in instructions
    assert "qualified possibility or hypothesis is not inaccurate" in instructions
    assert "customer's current issue" in instructions
    assert "immediately preceding agent exchange" in instructions
    assert "pass when the issue is actually resolved" in instructions
    assert "uncertain when a plausible troubleshooting step" in instructions
    assert "fail when the issue remains unresolved" in instructions
    assert "Terse or direct wording is not automatically unprofessional" in instructions
    assert "'please' or 'thank you'" in instructions
    assert "Do not return quotation text" in instructions
    assert "zero to three distinct structured references" in instructions
    assert "respecting any stricter wording" in instructions
