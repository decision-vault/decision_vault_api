from app.services.prd_multistep_service import PRDOrchestrator, Stage1Output


def test_parse_structured_recovers_from_unquoted_keys_and_missing_closers():
    raw = """
    ```json
    {
      executive_summary: "DecisionVault centralizes decision history across projects.",
      core_problem: "Teams lose the reasoning behind prior decisions.",
      why_tools_fail: "Chat and docs capture fragments but not durable rationale.",
      success_meaning: "Teams can trace why a decision happened within minutes.",
      primary_objective: "Provide a searchable decision record for every important project choice.",
      success_metrics: ["90% of major decisions recorded", "Median retrieval time under 2 minutes"],
      leading_indicators: ["Weekly active search usage", "Decision records created per project"],
      personas: [
        {"name": "Sarah", "role": "Engineering Lead", "description": "Needs fast context.", "pain_points": ["Context lost in threads"], "goals": ["Reduce repeated debates"]},
        {"name": "David", "role": "Product Manager", "description": "Needs rationale visibility.", "pain_points": ["Conflicting summaries"], "goals": ["Align roadmap tradeoffs"]},
        {"name": "Maya", "role": "CTO/Founder", "description": "Needs durable institutional memory.", "pain_points": ["Revisiting closed decisions"], "goals": ["Scale decision quality"]}
      ]
    ```
    """

    parsed = PRDOrchestrator._parse_structured(raw, Stage1Output)

    assert parsed.executive_summary.startswith("DecisionVault centralizes")
    assert len(parsed.personas) == 3
    assert parsed.personas[0].role == "Engineering Lead"


def test_parse_structured_recovers_from_prefixed_json_and_extra_closing_tokens():
    raw = """
    json
    {
      "executive_summary": "DecisionVault creates a single source of truth for project decisions.",
      "core_problem": "Decision context is scattered across tools.",
      "why_tools_fail": "Existing tools optimize for communication, not traceable decision memory.",
      "success_meaning": "Users can answer why a decision was made without interviewing past contributors.",
      "primary_objective": "Capture durable decision records with linked evidence.",
      "success_metrics": ["Search adoption across pilot teams"],
      "leading_indicators": ["Records created each week"],
      "personas": []
    }}}
    """

    parsed = PRDOrchestrator._parse_structured(raw, Stage1Output)

    assert parsed.core_problem == "Decision context is scattered across tools."
    assert parsed.success_metrics == ["Search adoption across pilot teams"]


def test_parse_structured_backfills_missing_required_stage_fields():
    raw = """
    {
      "executive_summary": "DecisionVault creates durable project memory.",
      "core_problem": "Teams cannot reliably recover why important decisions were made.",
      "success_metrics": ["Search adoption across pilot teams"],
      "leading_indicators": ["Decision records created each week"],
      "personas": []
    }
    """

    parsed = PRDOrchestrator._parse_structured(raw, Stage1Output)

    assert parsed.why_tools_fail == "Insufficient information provided."
    assert parsed.success_meaning == "Insufficient information provided."
    assert parsed.primary_objective == "Insufficient information provided."
