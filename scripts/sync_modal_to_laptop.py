#!/usr/bin/env python3
"""
Save Modal GPU implementation + pull run outputs onto this laptop.

Two modes:
  1) snapshot   — copy all code/config needed to reproduce the Modal run locally
  2) pull       — download results / cell checkpoints from the Modal volume

Examples:
  python scripts/sync_modal_to_laptop.py snapshot
  python scripts/sync_modal_to_laptop.py pull
  python scripts/sync_modal_to_laptop.py pull --run-id smoke_2026-07-23
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "deception-monitor-results"
BUNDLE_ROOT = ROOT / "results" / "modal_laptop_bundle"
RUNS_ROOT = ROOT / "results" / "modal_runs"

# Everything needed to re-read / re-run what Modal executes
SNAPSHOT_PATHS = [
    "scripts/modal_app.py",
    "scripts/preflight_check.py",
    "scripts/fetch_apollo_probes.py",
    "scripts/sync_modal_to_laptop.py",
    "scripts/apollo_offline_auroc_gate.py",
    "configs/experiment.yaml",
    "src/adversary/budget_ladder.py",
    "src/eval/frozen_cohort.py",
    "src/eval/robustness_gap.py",
    "src/eval/cross_fit.py",
    "src/monitors/apollo_probe.py",
    "src/monitors/linear_probe.py",
    "src/monitors/probe_adapter.py",
    "src/monitors/surface.py",
    "src/pipeline/calibration_chat.py",
    "src/pipeline/metrics_ext.py",
    "src/pipeline/residual.py",
    "src/pipeline/train_lora.py",
    "src/scenarios/agentic_deception.py",
    "src/stats/bootstrap.py",
    "docs/PREFLIGHT_CHECKLIST.md",
    "docs/CODE_STRUCTURE.md",
    "docs/WORKSHOP_ROADMAP_REVISED.md",
    "HF_LLAMA_ACCESS.md",
    "requirements.txt",
    "third_party/apollo_probes/instructed_pairs_detector.pt",
    "third_party/apollo_probes/roleplaying_detector.pt",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def cmd_snapshot(dest: Path | None = None) -> Path:
    dest = dest or (BUNDLE_ROOT / f"snapshot_{_stamp()}")
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for rel in SNAPSHOT_PATHS:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        copied.append(rel)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Local laptop copy of Modal GPU experiment implementation",
        "modal_app": "deception-monitor-robustness",
        "modal_volume": VOLUME,
        "copied": copied,
        "missing": missing,
        "how_to_pull_results": [
            "python scripts/sync_modal_to_laptop.py pull",
            "modal volume get deception-monitor-results / .",
        ],
        "how_to_smoke": [
            "modal run scripts/modal_app.py --n-sessions 16 --skip-sft --probe-layer 22",
        ],
        "how_to_full": [
            "modal run scripts/modal_app.py --n-sessions 180 --sft-steps 200 --run-full-grid",
        ],
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    (dest / "README.md").write_text(
        "\n".join(
            [
                "# Modal GPU implementation snapshot (laptop)",
                "",
                "Frozen copy of the code/config/probes that Modal runs.",
                "",
                "## After a Modal run, pull outputs to laptop",
                "```bash",
                "python scripts/sync_modal_to_laptop.py pull",
                "```",
                "",
                "Outputs land in `results/modal_runs/<run_id>/`.",
                "",
                "## Re-run notes",
                "- Needs Modal CLI + `huggingface` secret",
                "- Apollo `.pt` probes are included if present (gitignored in main repo)",
                "",
                f"Created: {manifest['created_utc']}",
                "",
            ]
        )
    )
    # Convenience: also refresh "latest" pointer
    latest = BUNDLE_ROOT / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    try:
        latest.symlink_to(dest.name)
    except OSError:
        # fallback copy pointer file
        (BUNDLE_ROOT / "LATEST_PATH.txt").write_text(str(dest))

    print(f"Snapshot → {dest}")
    print(f"  copied={len(copied)} missing={len(missing)}")
    if missing:
        print("  missing:", ", ".join(missing))
    return dest


def cmd_pull(run_id: str | None = None) -> Path:
    """Download Modal volume results onto the laptop."""
    run_id = run_id or f"pull_{_stamp()}"
    dest = RUNS_ROOT / run_id
    dest.mkdir(parents=True, exist_ok=True)

    # Prefer modal CLI volume get into dest
    # Remote layout: /results/modal_experiment_results.json, /results/checkpoints/cells/*.json
    cmds = [
        [
            "modal",
            "volume",
            "get",
            VOLUME,
            "modal_experiment_results.json",
            str(dest / "modal_experiment_results.json"),
        ],
        [
            "modal",
            "volume",
            "get",
            VOLUME,
            "checkpoints",
            str(dest / "checkpoints"),
        ],
    ]
    for cmd in cmds:
        print(">", " ".join(cmd))
        try:
            subprocess.run(cmd, check=False, cwd=str(ROOT))
        except FileNotFoundError:
            print(
                "ERROR: `modal` CLI not found. Install Modal, then re-run pull.\n"
                "  pip install modal && modal setup"
            )
            break

    # Also copy any local results JSON if present (from modal run --download)
    local_json = ROOT / "results" / "modal_experiment_results.json"
    if local_json.exists():
        shutil.copy2(local_json, dest / "modal_experiment_results.local_copy.json")

    meta = {
        "pulled_utc": datetime.now(timezone.utc).isoformat(),
        "volume": VOLUME,
        "dest": str(dest),
        "files": sorted(p.name for p in dest.rglob("*") if p.is_file()),
    }
    (dest / "PULL_META.json").write_text(json.dumps(meta, indent=2))
    print(f"Pull → {dest}")
    print(f"  files={len(meta['files'])}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["snapshot", "pull", "both"])
    ap.add_argument("--run-id", default=None, help="Folder name under results/modal_runs/")
    args = ap.parse_args()
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.mode in ("snapshot", "both"):
        cmd_snapshot()
    if args.mode in ("pull", "both"):
        cmd_pull(args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
