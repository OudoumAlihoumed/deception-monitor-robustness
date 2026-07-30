"""End-of-run artifact contract — prevent silent omission of CP2/CP6 fields."""

import pytest

from src.eval.artifact_contract import (
    ArtifactContractError,
    assert_episodes,
    assert_results_payload,
    assert_run_artifacts,
)


def _ep(**kw):
    base = {
        "session_id": "insider_deceptive_0",
        "scenario_id": "insider_trading",
        "family": "insider_trading",
        "intended_label": 1,
        "rule_label_b0": 0,
        "failure_mode": "honest",
        "response_b0": "I recommend a balanced public-data view.",
        "is_benign": False,
    }
    base.update(kw)
    return base


def test_episodes_ok():
    recs = [
        _ep(),
        _ep(session_id="x", intended_label=1, rule_label_b0=1, failure_mode=None),
        _ep(session_id="y", intended_label=0, rule_label_b0=0, failure_mode=None),
    ]
    out = assert_episodes(recs)
    assert out["ok"]


def test_episodes_reject_missing_failure_mode():
    with pytest.raises(ArtifactContractError, match="failure_mode"):
        assert_episodes([_ep(failure_mode=None)])


def test_results_payload_requires_elicitation_and_ci():
    payload = {
        "n_frozen_cohort": 18,
        "elicitation": {
            "per_family": {"audit_evasion": {"hit": 5, "tot": 5, "rate": 1.0}},
            "failure_modes": {"honest": 10},
            "n_frozen": 18,
            "n_intended_deceptive": 40,
            "episodes_path": "/results/episodes_b0.jsonl",
        },
        "table_auroc": {"linear_probe": {"b0": 0.75}},
        "table_auroc_ci_b0": {"linear_probe": {"estimate": 0.75}},
        "label_stats_by_rung": {"b0": {"n_frozen_cohort": 18}},
        "smoke_gate": {},
        "judge": {
            "judge_version": "def1-scheurer-benton-v1",
            "judge_content_hash": "abc",
            "family_set_version": "suite-v1-7fam",
        },
    }
    assert assert_results_payload(payload)["ok"]


def test_full_contract_checks_scores_length():
    recs = [_ep(rule_label_b0=1, failure_mode=None)]
    payload = {
        "n_frozen_cohort": 1,
        "elicitation": {
            "per_family": {"insider_trading": {"hit": 1, "tot": 1, "rate": 1.0}},
            "failure_modes": {},
            "n_frozen": 1,
            "n_intended_deceptive": 1,
            "episodes_path": "/results/episodes_b0.jsonl",
        },
        "table_auroc": {},
        "table_auroc_ci_b0": {},
        "label_stats_by_rung": {"b0": {"n_frozen_cohort": 1}},
        "smoke_gate": {},
        "judge": {
            "judge_version": "def1-scheurer-benton-v1",
            "judge_content_hash": "abc",
            "family_set_version": "suite-v1-7fam",
        },
    }
    scores = {"linear_probe": {"b0": [0.9]}}
    assert assert_run_artifacts(
        records=recs, payload=payload, scores_by_rung=scores, require_scores=True
    )["scores_checked"]
    with pytest.raises(ArtifactContractError, match="length"):
        assert_run_artifacts(
            records=recs,
            payload=payload,
            scores_by_rung={"linear_probe": {"b0": [0.1, 0.2]}},
            require_scores=True,
        )
