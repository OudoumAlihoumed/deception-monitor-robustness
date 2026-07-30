"""
Mid-run stage checkpoints — write artifacts immediately after each expensive stage.

Never keep calib / responses / activations only in RAM until end-of-run.
Caller must ``volume.commit()`` after these writers return (via ``_stage_commit``).

All JSON / JSONL writes are atomic (temp file in same dir → ``os.replace``).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


STAGE_PROGRESS_PATH = "checkpoints/stage_progress.json"
FROZEN_COHORT_PATH = "checkpoints/frozen_cohort.json"
THRESHOLDS_PATH = "checkpoints/thresholds.json"

_CORE_KEYS: tuple[str, ...] = (
    "session_id",
    "scenario_id",
    "family",
    "is_benign",
    "is_drift_benign",
    "intended_label",
    "failure_mode",
    "is_refusal",
    "system",
    "user_turns",
)

CELL_CHECKPOINT_REQUIRED: tuple[str, ...] = (
    "rung",
    "seed",
    "monitor_metrics",
)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> str:
    """Write via temp file in the same directory, then atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return str(path)


def atomic_write_json(path: Path, obj: Any, *, indent: int | None = 2) -> str:
    return atomic_write_text(
        path,
        json.dumps(obj, indent=indent, default=str),
    )


def slim_episode(record: dict[str, Any], *, tags: Sequence[str] = ("b0",)) -> dict[str, Any]:
    """Drop activations / specs; keep texts + labels for listed rungs."""
    out: dict[str, Any] = {k: record.get(k) for k in _CORE_KEYS if k in record}
    tag_set = set(tags)
    # b0 always included
    tag_set.add("b0")
    for tag in sorted(tag_set):
        resp_k, rule_k = f"response_{tag}", f"rule_label_{tag}"
        if resp_k in record:
            out[resp_k] = record.get(resp_k)
        if rule_k in record:
            out[rule_k] = record.get(rule_k)
        if tag == "b1" and "system_b1" in record:
            out["system_b1"] = record.get("system_b1")
    return out


def write_episodes_jsonl(
    path: Path,
    records: Sequence[dict[str, Any]],
    *,
    tags: Sequence[str] = ("b0",),
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(slim_episode(r, tags=tags), default=str) + "\n" for r in records
    ]
    return atomic_write_text(path, "".join(lines))


def write_chat_calib_cache(
    path: Path,
    *,
    scores: Sequence[float],
    resp_lens: Sequence[Any],
    fpr: float,
    model: str,
    probe_layer: int,
    partial: bool = False,
    n_target: Optional[int] = None,
    probe_thr: Any = None,
) -> str:
    path = Path(path)
    payload = {
        "n": int(len(scores)),
        "n_target": int(n_target) if n_target is not None else int(len(scores)),
        "partial": bool(partial),
        "fpr": float(fpr),
        "scores": [float(x) for x in scores],
        "resp_lens": list(resp_lens),
        "model": model,
        "probe_layer": int(probe_layer),
        "probe_thr": probe_thr,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    return atomic_write_json(path, payload, indent=None)


def write_thresholds(
    path: Path,
    *,
    probe_thr: Any,
    surface_thr: Any,
    cot_thr: Any,
    probe_fpr: float,
    surface_fpr: float,
    cot_fpr: float,
) -> str:
    return atomic_write_json(
        path,
        {
            "probe_thr": probe_thr,
            "surface_thr": surface_thr,
            "cot_thr": cot_thr,
            "probe_fpr": probe_fpr,
            "surface_fpr": surface_fpr,
            "cot_fpr": cot_fpr,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def load_thresholds(path: Path) -> dict[str, Any]:
    """Load pinned operating points. Raises if missing / unparsable / incomplete."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"thresholds missing: {path}")
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        raise ValueError(f"thresholds unreadable at {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"thresholds not a JSON object: {path}")
    for k in ("probe_thr", "probe_fpr"):
        if k not in data:
            raise ValueError(f"thresholds missing required field {k} at {path}")
    if data.get("probe_thr") is None:
        raise ValueError(f"thresholds.probe_thr is null at {path}")
    return data


def write_frozen_cohort(
    path: Path,
    *,
    session_ids: Sequence[str],
    frozen_mask: Sequence[bool],
    n_frozen: int,
) -> str:
    if len(session_ids) != len(frozen_mask):
        raise ValueError("session_ids and frozen_mask length mismatch")
    return atomic_write_json(
        path,
        {
            "n_sessions": len(session_ids),
            "n_frozen": int(n_frozen),
            "session_ids": list(session_ids),
            "frozen_mask": [bool(x) for x in frozen_mask],
            "frozen_session_ids": [
                sid for sid, m in zip(session_ids, frozen_mask) if m
            ],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Pinned at first b0 commit. Resume MUST reload this mask — "
                "do not recompute from rule_label_b0 (judge edits would silently "
                "change Δdet denominator)."
            ),
        },
    )


def load_frozen_cohort(
    path: Path,
    *,
    session_ids: Sequence[str],
) -> list[bool]:
    """
    Reload pinned frozen mask aligned to ``session_ids`` order.
    Hard-fails on parse errors, size mismatch, or session-id drift.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"frozen cohort missing: {path}")
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        raise ValueError(f"frozen cohort unreadable at {path}: {e}") from e
    for k in ("session_ids", "frozen_mask", "n_frozen"):
        if k not in data:
            raise ValueError(f"frozen cohort missing field {k}")
    saved_ids = list(data["session_ids"])
    saved_mask = [bool(x) for x in data["frozen_mask"]]
    if len(saved_ids) != len(saved_mask):
        raise ValueError("frozen cohort session_ids/mask length mismatch")
    if list(session_ids) != saved_ids:
        raise ValueError(
            "frozen cohort session_ids order/content drifted vs current records — "
            "refusing to resume (would change 𝒞)"
        )
    if int(sum(saved_mask)) != int(data["n_frozen"]):
        raise ValueError(
            f"frozen cohort n_frozen={data['n_frozen']} != sum(mask)={sum(saved_mask)}"
        )
    return saved_mask


def resolve_frozen_cohort(
    path: Path,
    records: Sequence[dict[str, Any]],
    *,
    compute_fn,
) -> tuple[list[bool], str]:
    """
    Load pinned 𝒞 from disk if present; else compute once and pin.

    Returns (mask, source) where source is ``reloaded`` or ``computed_and_pinned``.
    If a pin exists but session ids/order drifted (e.g. n=16 → n=60), recompute
    and overwrite the pin — do not silently use a mismatched 𝒞.
    """
    path = Path(path)
    session_ids = [str(r["session_id"]) for r in records]
    if path.exists():
        try:
            return load_frozen_cohort(path, session_ids=session_ids), "reloaded"
        except ValueError as e:
            # Bank size / order changed — pin is for a different run.
            print(
                f"[frozen-cohort] existing pin incompatible ({e}); "
                "recomputing and overwriting pin",
                flush=True,
            )
    mask = list(compute_fn(records))
    write_frozen_cohort(
        path,
        session_ids=session_ids,
        frozen_mask=mask,
        n_frozen=int(sum(mask)),
    )
    return mask, "computed_and_pinned"


def validate_cell_checkpoint(path: Path) -> dict[str, Any]:
    """Parse cell JSON and require load-bearing fields (existence alone is not enough)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cell checkpoint missing: {path}")
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        raise ValueError(f"cell checkpoint unreadable at {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"cell checkpoint not a JSON object: {path}")
    missing = [k for k in CELL_CHECKPOINT_REQUIRED if k not in data]
    if missing:
        raise ValueError(f"cell checkpoint {path.name} missing fields: {missing}")
    mm = data.get("monitor_metrics")
    if not isinstance(mm, dict) or "recall" not in mm or "auroc" not in mm:
        raise ValueError(
            f"cell checkpoint {path.name} monitor_metrics incomplete "
            "(need recall + auroc)"
        )
    return data


def write_activations_tag(
    act_dir: Path,
    tag: str,
    records: Sequence[dict[str, Any]],
    *,
    capture_layers: Sequence[int],
    capture_max_pool: bool = True,
    capture_per_token: bool = False,
) -> list[str]:
    """Serialize mean-pooled (and optional max) activations for one rung."""
    import numpy as np

    act_dir = Path(act_dir)
    act_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    meta = {
        "volume_backed": True,
        "volume_mount": "/results",
        "pooling": "mean_over_response_tokens",
        "also_max_pool": bool(capture_max_pool),
        "per_token": bool(capture_per_token),
        "capture_layers": list(capture_layers),
        "dtype": "float32",
        "rung": tag,
        "key_schema": "{session_id}__L{layer}[__mean|__max|__tok]",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    for r in records:
        act = r.get(f"activation_{tag}")
        if not isinstance(act, dict):
            continue
        sid = r["session_id"]
        meta_stash = act.get("__meta__") if isinstance(act.get("__meta__"), dict) else {}
        max_map = meta_stash.get("max") or {}
        tok_map = meta_stash.get("per_token") or {}
        for L, vec in act.items():
            if L == "__meta__":
                continue
            arrays[f"{sid}__L{int(L)}"] = np.asarray(vec, dtype=np.float32)
            arrays[f"{sid}__L{int(L)}__mean"] = np.asarray(vec, dtype=np.float32)
            if L in max_map or int(L) in max_map:
                mv = max_map.get(L, max_map.get(int(L)))
                if mv is not None:
                    arrays[f"{sid}__L{int(L)}__max"] = np.asarray(mv, dtype=np.float32)
            if tok_map and (L in tok_map or int(L) in tok_map):
                tv = tok_map.get(L, tok_map.get(int(L)))
                if tv is not None:
                    arrays[f"{sid}__L{int(L)}__tok"] = np.asarray(tv, dtype=np.float32)
    written: list[str] = []
    if arrays:
        npz = act_dir / f"{tag}_multilayer.npz"
        meta_p = act_dir / f"{tag}_multilayer.meta.json"
        # NumPy appends ".npz" unless the path already ends with it — temp must
        # end in ".npz" or os.replace looks for a missing file.
        npz_tmp = act_dir / f".{tag}_multilayer.partial.npz"
        if npz_tmp.exists():
            try:
                npz_tmp.unlink()
            except OSError:
                pass
        np.savez_compressed(npz_tmp, **arrays)
        os.replace(npz_tmp, npz)
        atomic_write_json(meta_p, meta)
        written = [str(npz), str(meta_p)]
    return written


def write_scores_partial(
    path: Path,
    records: Sequence[dict[str, Any]],
    scores_by_rung: dict[str, dict[str, Any]],
    frozen: Sequence[bool],
    *,
    tags: Iterable[str],
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tag_list = list(tags)
    lines: list[str] = []
    for i, r in enumerate(records):
        row = {
            "session_id": r["session_id"],
            "family": r.get("family"),
            "intended_label": r.get("intended_label"),
            "rule_label_b0": r.get("rule_label_b0"),
            "failure_mode": r.get("failure_mode"),
            "frozen": bool(frozen[i]) if i < len(frozen) else False,
            "scores": {
                m: {
                    tag: float(scores_by_rung[m][tag][i])
                    for tag in tag_list
                    if tag in scores_by_rung[m]
                }
                for m in scores_by_rung
            },
        }
        lines.append(json.dumps(row) + "\n")
    return atomic_write_text(path, "".join(lines))


def append_stage_progress(
    path: Path,
    stage: str,
    files: Sequence[str],
    *,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {"stages": []}
    else:
        data = {"stages": []}
    entry: dict[str, Any] = {
        "stage": stage,
        "files": list(files),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        entry["extra"] = extra
    stages = list(data.get("stages") or [])
    stages.append(entry)
    data["stages"] = stages
    data["last_stage"] = stage
    data["updated_utc"] = entry["utc"]
    atomic_write_json(path, data)
    return data


# ---------------------------------------------------------------------------
# Resume-from-stage (orchestration only — does not change fp16 / Apollo load)
# ---------------------------------------------------------------------------

# User-facing rung names. Resume means: skip finished work *before* this stage,
# then continue the ladder with the same model/probe protocol.
RESUME_STAGE_ORDER: tuple[str, ...] = ("b0", "calib", "b1", "b2", "b3", "b4")


def normalize_resume_stage(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip().lower()
    if not s:
        return None
    aliases = {
        "b0_elicit": "b0",
        "elicit": "b0",
        "alpaca": "calib",
        "calibration": "calib",
        "chat_calib": "calib",
        "b1_elicit": "b1",
        "sft": "b2",
        "sft-agnostic": "b2",
        "sft-aware": "b3",
        "rl": "b4",
    }
    s = aliases.get(s, s)
    if s not in RESUME_STAGE_ORDER:
        raise ValueError(
            f"resume_from_stage={name!r} invalid; choose one of {list(RESUME_STAGE_ORDER)}"
        )
    return s


def resume_stage_index(stage: str) -> int:
    return RESUME_STAGE_ORDER.index(normalize_resume_stage(stage) or "b0")


def should_run_stage(stage: str, resume_from: Optional[str]) -> bool:
    """True if this stage should execute (not skipped by resume)."""
    if not resume_from:
        return True
    return resume_stage_index(stage) >= resume_stage_index(resume_from)


def should_skip_before(stage: str, resume_from: Optional[str]) -> bool:
    """True if work *before* ``stage`` should be loaded from disk, not recomputed."""
    if not resume_from:
        return False
    return resume_stage_index(stage) < resume_stage_index(resume_from)


def default_episodes_path_for_resume(resume_from: Optional[str], results_root: Path) -> Path:
    """Prefer richest episode file available for the resume point."""
    root = Path(results_root)
    rf = normalize_resume_stage(resume_from)
    candidates: list[Path] = []
    if rf in ("b3", "b4"):
        candidates.append(root / "episodes_through_b2.jsonl")
    if rf in ("b2", "b3", "b4", "b1"):
        candidates.append(root / "episodes_b0_b1.jsonl")
    candidates.append(root / "episodes_b0.jsonl")
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def assert_resume_artifacts(
    resume_from: str,
    *,
    results_root: Path = Path("/results"),
) -> dict[str, Any]:
    """
    Hard-fail early if Volume lacks the artifacts needed to skip prior stages.

    Does not inspect model weights — fp16/Apollo load is unchanged.
    """
    root = Path(results_root)
    rf = normalize_resume_stage(resume_from)
    assert rf is not None
    if rf == "b0":
        return {
            "resume_from": rf,
            "episodes_path": None,
            "notes": ["resume_from=b0 → full ladder from the start"],
            "ok": True,
        }
    missing: list[str] = []
    notes: list[str] = []

    ep = default_episodes_path_for_resume(rf, root)
    if not ep.exists():
        missing.append(str(ep))
    else:
        rows = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
        if not rows:
            missing.append(f"{ep} (empty)")
        elif rf in ("b2", "b3", "b4"):
            n_b1 = sum(1 for r in rows if r.get("response_b1"))
            if n_b1 < len(rows):
                missing.append(
                    f"{ep}: need response_b1 on all rows for resume≥b2 "
                    f"(have {n_b1}/{len(rows)})"
                )
            if rf in ("b3", "b4"):
                n_b2 = sum(1 for r in rows if r.get("response_b2"))
                alt = root / "episodes_through_b2.jsonl"
                if n_b2 < len(rows) and not alt.exists():
                    missing.append(
                        f"response_b2 missing ({n_b2}/{len(rows)}); "
                        "need episodes_through_b2.jsonl from a prior pack_b2 commit"
                    )

    if rf in ("b1", "b2", "b3", "b4"):
        # Starting at b1 or later ⇒ calib must already be cached (complete).
        calib = root / "chat_calib_cache.json"
        if not calib.exists():
            missing.append(str(calib))
        else:
            try:
                c = json.loads(calib.read_text())
                if c.get("partial"):
                    missing.append(f"{calib} is partial (n={c.get('n')}/{c.get('n_target')})")
                elif int(c.get("n", 0)) < 8:
                    missing.append(f"{calib} too small n={c.get('n')}")
            except Exception as e:
                missing.append(f"{calib} unreadable: {e}")

        # τ must be pinned — existence alone is not enough.
        thr = root / THRESHOLDS_PATH
        try:
            pinned = load_thresholds(thr)
            notes.append(f"thresholds.probe_thr={pinned['probe_thr']}")
        except Exception as e:
            missing.append(f"{thr}: {e}")

        # 𝒞 should be pinned for resume ≥ b1. If missing (older Volume), bootstrap
        # is allowed once via resolve_frozen_cohort — but prefer a prior pin.
        cohort = root / FROZEN_COHORT_PATH
        if not cohort.exists():
            notes.append(
                f"{cohort} missing — will compute_and_pin from episodes on entry "
                "(first pin after upgrade; subsequent resumes must reload)"
            )
        else:
            try:
                data = json.loads(cohort.read_text())
                for k in ("session_ids", "frozen_mask", "n_frozen"):
                    if k not in data:
                        raise ValueError(f"missing field {k}")
                if int(sum(bool(x) for x in data["frozen_mask"])) != int(data["n_frozen"]):
                    raise ValueError("n_frozen ≠ sum(mask)")
                notes.append(f"frozen_cohort.n_frozen={data['n_frozen']}")
            except Exception as e:
                missing.append(f"{cohort}: {e}")

    if rf in ("b3", "b4"):
        # Optional but preferred: validate any existing b2 cell checkpoint parses.
        cells = root / "checkpoints" / "cells"
        if cells.exists():
            for cell_p in sorted(cells.glob("b2_*.json")):
                try:
                    validate_cell_checkpoint(cell_p)
                except Exception as e:
                    missing.append(f"{cell_p}: {e}")

    if missing:
        raise FileNotFoundError(
            f"resume_from_stage={rf} missing artifacts:\n  - "
            + "\n  - ".join(missing)
            + "\nRe-run earlier stages once (with stage commits); then resume."
        )

    return {
        "resume_from": rf,
        "episodes_path": str(ep) if ep.exists() else None,
        "notes": notes,
        "ok": True,
    }


def load_activations_tag_into_records(
    act_dir: Path,
    tag: str,
    records: Sequence[dict[str, Any]],
) -> int:
    """
    Attach activation_{tag} dicts from a prior npz commit.
    Returns number of records hydrated. 0 ⇒ caller should re-extract.
    """
    import numpy as np

    npz_path = Path(act_dir) / f"{tag}_multilayer.npz"
    if not npz_path.exists():
        return 0
    data = np.load(npz_path)
    n_hit = 0
    for r in records:
        sid = r["session_id"]
        act: dict[Any, Any] = {}
        prefix = f"{sid}__L"
        for key in data.files:
            if not key.startswith(prefix):
                continue
            if key.endswith("__max") or key.endswith("__tok"):
                continue
            # keys: {sid}__L{layer} or {sid}__L{layer}__mean
            core = key[: -len("__mean")] if key.endswith("__mean") else key
            try:
                layer = int(core.rsplit("__L", 1)[-1])
            except ValueError:
                continue
            act[layer] = np.asarray(data[key], dtype=np.float32)
        if act:
            r[f"activation_{tag}"] = act
            n_hit += 1
    return n_hit
