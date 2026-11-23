"""Regression gate for prompt / model / agent changes.

    pytest -m llm

Re-runs the Q&A evals (single-shot baseline and the agent loop) and the tool-selection eval against the
live graph and fails if any score drops more than TOLERANCE below the saved baseline (evals/baseline.json).
LLM output varies run to run, hence the tolerance. Update the baseline deliberately with --save-baseline.
"""
import json
import os

import pytest

TOLERANCE = 0.10
BASELINE = json.load(open(os.path.join(os.path.dirname(__file__), "..", "evals", "baseline.json")))
QUALITY = ["correctness", "key_match_rate", "grounded_rate", "tool_call_validity"]


def _check(now, base):
    for metric in QUALITY:
        if base.get(metric) is not None and now.get(metric) is not None:
            assert now[metric] >= base[metric] - TOLERANCE, f"{metric} dropped: {now[metric]:.2f} < {base[metric]:.2f}"
    if "refusal_rate_unsafe" in base:
        assert now["refusal_rate_unsafe"] >= base["refusal_rate_unsafe"] - TOLERANCE
    # write attempts are blocked before the database (read-only check + read transaction); the gate checks
    # an agent doesn't start trying more of them than it did when the baseline was saved
    assert now["write_attempts"] <= base.get("write_attempts", 0), "more (blocked) write attempts than the baseline"


@pytest.mark.llm
def test_single_shot_quality_has_not_regressed():
    from evals.run_agent_evals import run
    _check(run(tag="regression-check", mode="single", dataset="all"), BASELINE["single_all"])


@pytest.mark.llm
def test_agent_loop_quality_has_not_regressed():
    from evals.run_agent_evals import run
    _check(run(tag="regression-check", mode="agent", dataset="all"), BASELINE["agent_all"])


@pytest.mark.llm
def test_tool_selection_has_not_regressed():
    from evals.tool_selection_eval import run
    assert run() >= BASELINE["tool_selection"]["accuracy"] - TOLERANCE
