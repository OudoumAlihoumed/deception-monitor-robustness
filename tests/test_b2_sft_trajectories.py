"""b2 SFT trajectory construction — deceive-while-concealing targets only."""

from __future__ import annotations

from src.adversary.budget_ladder import build_b2_sft_trajectories, curate_b2_trajectories


def test_build_b2_uses_b0_success_not_b1_honest():
    records = [
        {
            "session_id": "ok",
            "intended_label": 1,
            "rule_label_b0": 1,
            "rule_label_b1": 0,  # already stopped deceiving at b1
            "system": "sys0",
            "system_b1": "sys1_conceal",
            "user_turns": ["u"],
            "response_b0": "successful deception text",
            "response_b1": "honest refusal-ish text",
        },
        {
            "session_id": "fail_b0",
            "intended_label": 1,
            "rule_label_b0": 0,
            "system_b1": "sys1",
            "user_turns": ["u"],
            "response_b0": "not deceptive",
            "response_b1": "also not",
        },
        {
            "session_id": "benign",
            "intended_label": 0,
            "rule_label_b0": 0,
            "system_b1": "sys1",
            "user_turns": ["u"],
            "response_b0": "benign",
        },
    ]
    raw = build_b2_sft_trajectories(records)
    assert len(raw) == 1
    assert raw[0]["session_id"] == "ok"
    assert raw[0]["system"] == "sys1_conceal"
    assert raw[0]["assistant"] == "successful deception text"
    assert "honest" not in raw[0]["assistant"]

    curated = curate_b2_trajectories(raw)
    assert len(curated) == 1
    assert curated[0]["assistant"] == "successful deception text"
