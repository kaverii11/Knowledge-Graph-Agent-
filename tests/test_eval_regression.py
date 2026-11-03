"""Regression gate for prompt / model / agent changes.

    pytest -m llm

Re-runs the agent eval and the tool-selection eval against the live graph and fails if any score
drops more than TOLERANCE below the saved baseline (evals/baseline.json). LLM output varies run to
run, hence the tolerance. Update the baseline deliberately with --save-baseline.
"""
import json
import os

import pytest

TOLERANCE = 0.10
BASELINE = json.load(open(os.path.join(os.path.dirname(__file__), "..", "evals", "baseline.json")))


@pytest.mark.llm
def test_agent_quality_has_not_regressed():
    from evals.run_agent_evals import run
    now = run(tag="regression-check")
    for metric in ["query_ok_rate", "retrieval_hit_rate", "correctness", "grounded_rate"]:
        assert now[metric] >= BASELINE["agent"][metric] - TOLERANCE, f"{metric} dropped: {now[metric]:.2f}"


@pytest.mark.llm
def test_tool_selection_has_not_regressed():
    from evals.tool_selection_eval import run
    assert run() >= BASELINE["tool_selection"]["accuracy"] - TOLERANCE
