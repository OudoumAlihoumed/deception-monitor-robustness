#!/usr/bin/env python3
"""
E4/E5 — Aggregate per-cell Modal checkpoints into summary tables.

Cell files under results/**/checkpoints/cells/*.json are the unit of analysis.
Top-level modal_experiment_results.json tables named *_last_cell are **one**
seed×λ only — never treat them as the paper Table 1 without this aggregate.

Usage:
  PYTHONPATH=. python scripts/aggregate_cells.py
  PYTHONPATH=. python scripts/aggregate_cells.py --roots results/modal_runs/full_*
  PYTHONPATH=. python scripts/aggregate_cells.py --write results/summary_pooled.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]

CELL_RE = re.compile(
    r"^(?P<rung>b[234])_seed(?P<seed>\d+)(?:_lam(?P<lam>[0-9.]+))?\.json$"
)


def _mean_std(xs: list[float]) -> dict[str, Any]:
    """Summarise numeric values across seeds/cells.

    n==2: emit both values + range (never mean±std — paper must not print ± on n=2).
    """
    n = len(xs)
    if n == 0:
        return {"n": 0, "n_seeds": 0, "mean": None, "std": None}
    if n == 2:
        lo, hi = (xs[0], xs[1]) if xs[0] <= xs[1] else (xs[1], xs[0])
        return {
            "n": 2,
            "n_seeds": 2,
            "values": [float(xs[0]), float(xs[1])],
            "range": [float(lo), float(hi)],
            "reporting": "values_and_range",
            "note": "n_seeds=2: do not report mean±std",
        }
    m = sum(xs) / n
    if n == 1:
        return {"n": 1, "n_seeds": 1, "mean": m, "std": 0.0}
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return {"n": n, "n_seeds": n, "mean": m, "std": math.sqrt(var)}


def discover_cell_files(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".json":
            found.append(root)
            continue
        found.extend(sorted(root.rglob("checkpoints/cells/*.json")))
        found.extend(sorted(root.glob("*.json")))  # synthetic flat dumps
    # unique by resolve
    uniq = {}
    for p in found:
        uniq[p.resolve()] = p
    return list(uniq.values())


def parse_cell(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return {"path": str(path), "error": f"read_fail: {e}"}
    m = CELL_RE.match(path.name)
    rung = data.get("rung") or (m.group("rung") if m else None)
    seed = data.get("seed")
    if seed is None and m:
        seed = int(m.group("seed"))
    lam = data.get("lambda")
    if lam is None and m and m.group("lam") is not None:
        lam = float(m.group("lam"))
    metrics = data.get("monitor_metrics") or data.get("metrics") or {}
    return {
        "path": str(path),
        "rung": rung,
        "seed": seed,
        "lambda": lam,
        "gpu_seconds": data.get("gpu_seconds"),
        "lambda_worked": data.get("lambda_worked"),
        "adapter_live": data.get("adapter_live"),
        "continue_from_b3": data.get("continue_from_b3"),
        "monitor_metrics": metrics,
        "raw_keys": sorted(data.keys()),
    }


def _dig_recall(metrics: dict, monitor: str, rung: str) -> Optional[float]:
    # Expected shapes:
    #   metrics["recall"][monitor][rung]
    #   metrics["table_recall_at_alpha_fpr"][monitor][rung]
    for key in ("recall", "table_recall_at_alpha_fpr", "table_recall_at_1pct_fpr"):
        block = metrics.get(key)
        if isinstance(block, dict) and monitor in block:
            v = block[monitor].get(rung)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _dig_auroc(metrics: dict, monitor: str, rung: str) -> Optional[float]:
    for key in ("auroc", "table_auroc"):
        block = metrics.get(key)
        if isinstance(block, dict) and monitor in block:
            v = block[monitor].get(rung)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def dig_delta_det_entry(entry: Any) -> dict[str, Any]:
    """
    Parse a table_delta_det[monitor][rung] cell.

    Null / non-reportable Δdet must stay excluded — never coerced to 0.0
    (would fabricate a gap) and never silently dropped without a reason.
    """
    if entry is None:
        return {
            "reportable": False,
            "delta_det": None,
            "omit_reason": "null_entry",
            "adaptive_outcome": None,
        }
    if not isinstance(entry, dict):
        if isinstance(entry, (int, float)):
            return {"reportable": True, "delta_det": float(entry), "omit_reason": None}
        return {
            "reportable": False,
            "delta_det": None,
            "omit_reason": "non_numeric",
            "adaptive_outcome": None,
        }
    reportable = entry.get("reportable")
    d = entry.get("delta_det")
    if reportable is False or d is None:
        gate = entry.get("gate") or {}
        return {
            "reportable": False,
            "delta_det": None,
            "delta_det_raw": entry.get("delta_det_raw", gate.get("delta_det_raw")),
            "omit_reason": gate.get("omit_reason")
            or entry.get("adaptive_outcome")
            or "gated",
            "adaptive_outcome": entry.get("adaptive_outcome") or gate.get("outcome"),
        }
    return {
        "reportable": True,
        "delta_det": float(d),
        "omit_reason": None,
        "adaptive_outcome": entry.get("adaptive_outcome"),
    }


def summarize_delta_det_table(table: dict[str, Any]) -> dict[str, Any]:
    """Aggregate reportable Δdet only; log exclusions separately."""
    out: dict[str, Any] = {"by_monitor": {}, "excluded": []}
    for monitor, by_rung in (table or {}).items():
        if not isinstance(by_rung, dict):
            continue
        reportable_vals: dict[str, list[float]] = defaultdict(list)
        for rung, entry in by_rung.items():
            parsed = dig_delta_det_entry(entry)
            if parsed["reportable"] and parsed["delta_det"] is not None:
                reportable_vals[str(rung)].append(float(parsed["delta_det"]))
            else:
                out["excluded"].append(
                    {
                        "monitor": monitor,
                        "rung": rung,
                        "omit_reason": parsed.get("omit_reason"),
                        "adaptive_outcome": parsed.get("adaptive_outcome"),
                        "delta_det_raw": parsed.get("delta_det_raw"),
                    }
                )
        out["by_monitor"][monitor] = {
            rung: _mean_std(vals) for rung, vals in sorted(reportable_vals.items())
        }
    out["note"] = (
        "Excluded rungs are listed with omit_reason; they are not averaged as 0 "
        "and not silently dropped from the exclusion log."
    )
    return out


def aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, list] = defaultdict(list)
    by_lambda: dict[str, list] = defaultdict(list)
    by_rung: dict[str, list] = defaultdict(list)

    for c in cells:
        if c.get("error"):
            continue
        rung = c.get("rung") or "unknown"
        seed = c.get("seed")
        lam = c.get("lambda")
        by_rung[str(rung)].append(c)
        if seed is not None:
            by_seed[f"{rung}_seed{seed}"].append(c)
        if lam is not None:
            by_lambda[f"{rung}_lam{lam}"].append(c)

    def summarize_group(group: list[dict]) -> dict[str, Any]:
        gpu = [float(c["gpu_seconds"]) for c in group if c.get("gpu_seconds") is not None]
        lw = [
            1.0
            for c in group
            if isinstance(c.get("lambda_worked"), dict) and c["lambda_worked"].get("passed")
        ]
        al = [
            1.0
            for c in group
            if isinstance(c.get("adapter_live"), dict) and c["adapter_live"].get("passed")
        ]
        out: dict[str, Any] = {
            "n_cells": len(group),
            "gpu_seconds": _mean_std(gpu),
            "lambda_worked_pass_rate": (sum(lw) / len(group)) if group else None,
            "adapter_live_pass_rate": (sum(al) / len(group)) if group else None,
            "paths": [c["path"] for c in group],
        }
        # Probe recall / AUROC at this cell's rung if logged
        for monitor in ("linear_probe", "surface", "cot_reading"):
            recalls, aurocs = [], []
            for c in group:
                r = c.get("rung")
                m = c.get("monitor_metrics") or {}
                if r:
                    rv = _dig_recall(m, monitor, r)
                    av = _dig_auroc(m, monitor, r)
                    if rv is not None:
                        recalls.append(rv)
                    if av is not None:
                        aurocs.append(av)
            out[f"recall_{monitor}"] = _mean_std(recalls)
            out[f"auroc_{monitor}"] = _mean_std(aurocs)
        return out

    return {
        "schema_version": "1.0",
        "result_type": "cell_checkpoint_aggregate",
        "warning": (
            "Use summary_pooled / summary_by_seed / summary_by_lambda for Table 1. "
            "Never use modal_experiment_results.json *_last_cell fields as aggregates."
        ),
        "n_cells_loaded": len(cells),
        "n_cells_ok": sum(1 for c in cells if not c.get("error")),
        "summary_by_rung": {k: summarize_group(v) for k, v in sorted(by_rung.items())},
        "summary_by_seed": {k: summarize_group(v) for k, v in sorted(by_seed.items())},
        "summary_by_lambda": {k: summarize_group(v) for k, v in sorted(by_lambda.items())},
        "summary_pooled": {
            k: summarize_group(v)
            for k, v in sorted(by_rung.items())
            if k in ("b2", "b3", "b4")
        },
        "cells": cells,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=[
            ROOT / "results" / "checkpoints" / "cells",
            ROOT / "results" / "modal_runs",
            ROOT / "results" / "modal_laptop_bundle",
        ],
    )
    ap.add_argument("--write", type=Path, default=ROOT / "results" / "summary_pooled.json")
    ap.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Optional modal_experiment_results.json — summarize gated Δdet (null ≠ 0)",
    )
    args = ap.parse_args()

    files = discover_cell_files(args.roots)
    cells = [parse_cell(p) for p in files]
    cells = [c for c in cells if c is not None]
    summary = aggregate(cells)
    summary["roots"] = [str(r) for r in args.roots]
    summary["n_files"] = len(files)

    if args.results_json and args.results_json.exists():
        payload = json.loads(args.results_json.read_text())
        summary["delta_det_from_results"] = summarize_delta_det_table(
            payload.get("table_delta_det") or {}
        )

    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(json.dumps(summary, indent=2, default=str))
    print(
        json.dumps(
            {
                "wrote": str(args.write),
                "n_files": len(files),
                "n_ok": summary["n_cells_ok"],
                "by_rung": list(summary["summary_by_rung"].keys()),
                "delta_det_excluded": (
                    summary.get("delta_det_from_results", {}).get("excluded")
                ),
                "warning": summary["warning"],
            },
            indent=2,
        )
    )
    if summary["n_cells_ok"] == 0:
        print(
            "NOTE: no cell metrics yet — after Modal cells land (with monitor_metrics), "
            "re-run this script for pooled Table 1 (n=2 → values+range)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
