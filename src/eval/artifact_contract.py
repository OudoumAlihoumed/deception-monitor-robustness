"""
End-of-run artifact contract.

We have been burned twice by runs that produced summary metrics but omitted
the fields needed for post-hoc diagnosis (responses → CP2; per-episode scores → CP6).
Assert presence once, at commit time.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence


# Slim episode fields every b0 elicit must persist (no activations).
EPISODE_REQUIRED_FIELDS: tuple[str, ...] = (
    "session_id",
    "scenario_id",
    "family",
    "intended_label",
    "rule_label_b0",
    "failure_mode",  # null OK when rule_label_b0==1 or intended_label==0
    "response_b0",
    "is_benign",
)

# Top-level results JSON keys that must be present after a ladder / smoke run.
RESULTS_REQUIRED_KEYS: tuple[str, ...] = (
    "n_frozen_cohort",
    "elicitation",
    "table_auroc",
    "table_auroc_ci_b0",
    "label_stats_by_rung",
    "smoke_gate",
    "judge",
)

ELICITATION_REQUIRED_KEYS: tuple[str, ...] = (
    "per_family",
    "failure_modes",
    "n_frozen",
    "n_intended_deceptive",
    "episodes_path",
)


class ArtifactContractError(RuntimeError):
    """Raised when a run omits contract-required artifacts."""


def assert_episode_record(record: dict[str, Any], *, idx: Optional[int] = None) -> None:
    missing = [k for k in EPISODE_REQUIRED_FIELDS if k not in record]
    if missing:
        where = f" record[{idx}]" if idx is not None else ""
        raise ArtifactContractError(f"episode{where} missing fields: {missing}")
    if not record.get("session_id"):
        raise ArtifactContractError("episode session_id empty")
    # response may be empty string only for hard failures — still must be str
    if not isinstance(record.get("response_b0"), str):
        raise ArtifactContractError(
            f"episode {record.get('session_id')}: response_b0 must be str"
        )


def assert_episodes(records: Sequence[dict[str, Any]], *, min_n: int = 1) -> dict[str, Any]:
    if len(records) < min_n:
        raise ArtifactContractError(f"need ≥{min_n} episodes, got {len(records)}")
    for i, r in enumerate(records):
        assert_episode_record(r, idx=i)
    n_intended = sum(1 for r in records if int(r.get("intended_label", 0)) == 1)
    n_with_fm = sum(
        1
        for r in records
        if int(r.get("intended_label", 0)) == 1
        and int(r.get("rule_label_b0", 0)) == 0
        and r.get("failure_mode") is not None
    )
    n_fail = sum(
        1
        for r in records
        if int(r.get("intended_label", 0)) == 1 and int(r.get("rule_label_b0", 0)) == 0
    )
    if n_fail and n_with_fm < n_fail:
        raise ArtifactContractError(
            f"failure_mode missing on {n_fail - n_with_fm}/{n_fail} non-frozen intended episodes"
        )
    return {
        "n_episodes": len(records),
        "n_intended_deceptive": n_intended,
        "n_failure_mode_tagged": n_with_fm,
        "ok": True,
    }


def assert_per_episode_scores(
    scores_by_monitor: dict[str, Any],
    *,
    n_sessions: int,
    required_monitors: Iterable[str] = ("linear_probe",),
    rung: str = "b0",
) -> None:
    for m in required_monitors:
        if m not in scores_by_monitor:
            raise ArtifactContractError(f"scores missing monitor={m}")
        arr = scores_by_monitor[m]
        if isinstance(arr, dict):
            if rung not in arr:
                raise ArtifactContractError(f"scores[{m}] missing rung={rung}")
            arr = arr[rung]
        try:
            n = len(arr)
        except TypeError as e:
            raise ArtifactContractError(f"scores[{m}/{rung}] not a sequence") from e
        if n != n_sessions:
            raise ArtifactContractError(
                f"scores[{m}/{rung}] length {n} != n_sessions {n_sessions}"
            )


def assert_results_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in RESULTS_REQUIRED_KEYS if k not in payload]
    if missing:
        raise ArtifactContractError(f"results payload missing keys: {missing}")
    elic = payload.get("elicitation") or {}
    miss_e = [k for k in ELICITATION_REQUIRED_KEYS if k not in elic]
    if miss_e:
        raise ArtifactContractError(f"elicitation block missing keys: {miss_e}")
    if not elic.get("per_family"):
        raise ArtifactContractError("elicitation.per_family empty")
    judge = payload.get("judge") or {}
    for k in ("judge_version", "judge_content_hash", "family_set_version"):
        if not judge.get(k):
            raise ArtifactContractError(f"judge.{k} missing — freeze identity before commit")
    # frozen mask must be recoverable: n_frozen matches label_stats
    n_frozen = int(payload.get("n_frozen_cohort", -1))
    if n_frozen < 0:
        raise ArtifactContractError("n_frozen_cohort missing")
    stats0 = (payload.get("label_stats_by_rung") or {}).get("b0") or {}
    if stats0 and int(stats0.get("n_frozen_cohort", n_frozen)) != n_frozen:
        raise ArtifactContractError("n_frozen_cohort disagrees with label_stats_by_rung.b0")
    return {
        "ok": True,
        "n_frozen_cohort": n_frozen,
        "elicitation_keys": list(elic.keys()),
        "judge_version": judge.get("judge_version"),
        "judge_content_hash": judge.get("judge_content_hash"),
    }


def assert_run_artifacts(
    *,
    records: Sequence[dict[str, Any]],
    payload: dict[str, Any],
    scores_by_rung: Optional[dict[str, Any]] = None,
    require_scores: bool = True,
) -> dict[str, Any]:
    """Full contract check before volume commit."""
    ep = assert_episodes(records)
    res = assert_results_payload(payload)
    if require_scores:
        if scores_by_rung is None:
            raise ArtifactContractError("scores_by_rung required but None")
        assert_per_episode_scores(scores_by_rung, n_sessions=len(records))
    return {"episodes": ep, "results": res, "scores_checked": require_scores}
