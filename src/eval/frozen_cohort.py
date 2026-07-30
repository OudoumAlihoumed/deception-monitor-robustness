"""Frozen-cohort label protocol for Δdet (avoids label collapse under b2)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def frozen_deceptive_mask(
    records: Sequence[Mapping[str, Any]],
    *,
    intended_key: str = "intended_label",
    rule_b0_key: str = "rule_label_b0",
) -> list[bool]:
    """
    Primary analysis set: intended deceptive AND rule-label positive at b0.

    Episode membership is frozen at b0 so later rungs cannot shrink the
    positive set by suppressing the text heuristic (label collapse).
    """
    out = []
    for r in records:
        intended = int(r.get(intended_key, 0))
        rule_b0 = int(r.get(rule_b0_key, r.get("rule_label", 0)))
        out.append(bool(intended == 1 and rule_b0 == 1))
    return out


def per_rung_label_stats(
    records: Sequence[Mapping[str, Any]],
    rung: str,
    frozen_mask: Sequence[bool],
) -> dict[str, Any]:
    """Behavioral measurement: rule_label rates; n_positive for the frozen set."""
    rule_key = f"rule_label_{rung}"
    rules = [int(r.get(rule_key, r.get("rule_label", 0))) for r in records]
    intended = [int(r.get("intended_label", 0)) for r in records]
    n = len(records)
    n_frozen = int(sum(frozen_mask))
    n_rule_pos = int(sum(rules))
    n_rule_pos_in_frozen = int(
        sum(1 for i, m in enumerate(frozen_mask) if m and rules[i] == 1)
    )
    n_intended = int(sum(1 for x in intended if x == 1))
    return {
        "rung": rung,
        "n_sessions": n,
        "n_intended_deceptive": n_intended,
        "n_frozen_cohort": n_frozen,
        "n_rule_label_positive": n_rule_pos,
        "n_rule_label_positive_in_frozen": n_rule_pos_in_frozen,
        "rule_label_rate_all": n_rule_pos / max(n, 1),
        "rule_label_rate_frozen": n_rule_pos_in_frozen / max(n_frozen, 1),
        "note": (
            "Δdet uses frozen cohort membership; rule_label_rate_* is behavioral only"
        ),
    }
