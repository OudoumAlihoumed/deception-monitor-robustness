"""Mid-run stage checkpoint writers (no Modal / GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.stage_checkpoints import (
    append_stage_progress,
    slim_episode,
    write_chat_calib_cache,
    write_episodes_jsonl,
    write_scores_partial,
    write_thresholds,
)


def test_write_episodes_and_progress(tmp_path: Path):
    records = [
        {
            "session_id": "insider_deceptive_0",
            "scenario_id": "insider",
            "family": "insider_trading",
            "is_benign": False,
            "is_drift_benign": False,
            "intended_label": 1,
            "rule_label_b0": 1,
            "rule_label_b1": 0,
            "failure_mode": None,
            "is_refusal": False,
            "system": "sys",
            "system_b1": "sys_b1",
            "user_turns": ["u"],
            "response_b0": "buy",
            "response_b1": "hedge",
            "activation_b0": {"skip": True},
        }
    ]
    ep = tmp_path / "episodes_b0_b1.jsonl"
    write_episodes_jsonl(ep, records, tags=("b0", "b1"))
    row = json.loads(ep.read_text().splitlines()[0])
    assert row["response_b0"] == "buy"
    assert row["response_b1"] == "hedge"
    assert "activation_b0" not in row

    prog = tmp_path / "stage_progress.json"
    append_stage_progress(prog, "b1_elicit", [str(ep)], extra={"n": 1})
    data = json.loads(prog.read_text())
    assert data["last_stage"] == "b1_elicit"
    assert data["stages"][-1]["files"] == [str(ep)]


def test_calib_cache_partial(tmp_path: Path):
    p = tmp_path / "chat_calib_cache.json"
    write_chat_calib_cache(
        p,
        scores=[0.1, 0.2],
        resp_lens=[3, 4],
        fpr=0.01,
        model="m",
        probe_layer=22,
        partial=True,
        n_target=1000,
    )
    d = json.loads(p.read_text())
    assert d["partial"] is True
    assert d["n"] == 2
    assert d["n_target"] == 1000


def test_scores_partial(tmp_path: Path):
    records = [
        {
            "session_id": "s0",
            "family": "x",
            "intended_label": 1,
            "rule_label_b0": 1,
            "failure_mode": None,
        }
    ]
    scores = {
        "linear_probe": {"b0": [0.9]},
        "surface": {"b0": [0.1]},
        "cot_reading": {"b0": [0.2]},
    }
    sp = tmp_path / "scores.jsonl"
    write_scores_partial(sp, records, scores, [True], tags=["b0"])
    row = json.loads(sp.read_text().splitlines()[0])
    assert row["frozen"] is True
    assert row["scores"]["linear_probe"]["b0"] == 0.9


def test_slim_keeps_requested_tags_only():
    r = {
        "session_id": "s",
        "scenario_id": "sc",
        "family": "f",
        "is_benign": False,
        "intended_label": 1,
        "rule_label_b0": 1,
        "rule_label_b2": 1,
        "response_b0": "a",
        "response_b2": "c",
        "system": "sys",
        "user_turns": [],
        "failure_mode": None,
        "is_refusal": False,
    }
    s = slim_episode(r, tags=("b0",))
    assert "response_b0" in s
    assert "response_b2" not in s


def test_thresholds(tmp_path: Path):
    p = tmp_path / "thr.json"
    write_thresholds(
        p,
        probe_thr=0.01,
        surface_thr=0.2,
        cot_thr=0.3,
        probe_fpr=0.01,
        surface_fpr=0.05,
        cot_fpr=0.05,
    )
    assert json.loads(p.read_text())["probe_thr"] == 0.01


def test_resume_stage_order():
    from src.eval.stage_checkpoints import (
        normalize_resume_stage,
        should_run_stage,
    )

    assert normalize_resume_stage("sft-aware") == "b3"
    assert should_run_stage("b2", "b2") is True
    assert should_run_stage("b1", "b2") is False
    assert should_run_stage("b4", "b2") is True


def test_atomic_write_and_frozen_cohort_pin(tmp_path: Path):
    from src.eval.frozen_cohort import frozen_deceptive_mask
    from src.eval.stage_checkpoints import (
        atomic_write_text,
        load_frozen_cohort,
        load_thresholds,
        resolve_frozen_cohort,
        validate_cell_checkpoint,
        write_frozen_cohort,
    )

    target = tmp_path / "x.txt"
    atomic_write_text(target, "hello\n")
    assert target.read_text() == "hello\n"
    assert not list(tmp_path.glob(".x.txt.*.tmp"))

    records = [
        {"session_id": "a", "intended_label": 1, "rule_label_b0": 1},
        {"session_id": "b", "intended_label": 1, "rule_label_b0": 0},
        {"session_id": "c", "intended_label": 0, "rule_label_b0": 0},
    ]
    path = tmp_path / "frozen_cohort.json"
    mask1, src1 = resolve_frozen_cohort(path, records, compute_fn=frozen_deceptive_mask)
    assert src1 == "computed_and_pinned"
    assert sum(mask1) == 1
    # Tamper would-be recompute: change labels on records
    records[0]["rule_label_b0"] = 0
    mask2, src2 = resolve_frozen_cohort(path, records, compute_fn=frozen_deceptive_mask)
    assert src2 == "reloaded"
    assert mask2 == mask1  # disk wins over recomputed labels
    assert sum(mask2) == 1

    thr = tmp_path / "thresholds.json"
    write_thresholds(
        thr,
        probe_thr=0.123,
        surface_thr=0.2,
        cot_thr=0.3,
        probe_fpr=0.01,
        surface_fpr=0.05,
        cot_fpr=0.05,
    )
    assert load_thresholds(thr)["probe_thr"] == 0.123

    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps({"rung": "b2", "seed": 0}))
    try:
        validate_cell_checkpoint(cell)
        assert False, "expected incomplete cell to fail"
    except ValueError:
        pass
    cell.write_text(
        json.dumps(
            {
                "rung": "b2",
                "seed": 0,
                "monitor_metrics": {"recall": 0.1, "auroc": 0.5},
            }
        )
    )
    assert validate_cell_checkpoint(cell)["rung"] == "b2"


def test_load_frozen_rejects_session_drift(tmp_path: Path):
    from src.eval.stage_checkpoints import load_frozen_cohort, write_frozen_cohort

    p = tmp_path / "c.json"
    write_frozen_cohort(
        p,
        session_ids=["a", "b"],
        frozen_mask=[True, False],
        n_frozen=1,
    )
    try:
        load_frozen_cohort(p, session_ids=["b", "a"])
        assert False, "order drift should fail"
    except ValueError:
        pass

