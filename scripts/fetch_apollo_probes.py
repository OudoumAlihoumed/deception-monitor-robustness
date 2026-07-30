#!/usr/bin/env python3
"""
Fetch published Goldowsky-Dill / Apollo residual probes.

Weights are gitignored (*.pt). Source of truth:
  https://github.com/ApolloResearch/deception-detection
  https://data.apolloresearch.ai/dd/

Default: copy from a local clone's example_results/, or download raw from GitHub
if the repo vendors example_results/instructed_pairs/detector.pt.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "third_party" / "apollo_probes"

# Raw paths in ApolloResearch/deception-detection (may move; override via --src-dir)
GITHUB_RAW = {
    "instructed_pairs_detector.pt": (
        "https://raw.githubusercontent.com/ApolloResearch/deception-detection/"
        "main/example_results/instructed_pairs/detector.pt"
    ),
    "roleplaying_detector.pt": (
        "https://raw.githubusercontent.com/ApolloResearch/deception-detection/"
        "main/example_results/roleplaying/detector.pt"
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src-dir",
        type=Path,
        default=None,
        help="Local deception-detection clone; copies example_results/*/detector.pt",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.src_dir:
        mapping = {
            "instructed_pairs_detector.pt": args.src_dir
            / "example_results"
            / "instructed_pairs"
            / "detector.pt",
            "roleplaying_detector.pt": args.src_dir
            / "example_results"
            / "roleplaying"
            / "detector.pt",
        }
        for name, src in mapping.items():
            dst = OUT / name
            if dst.exists() and not args.force:
                print(f"skip {dst} (exists)")
                continue
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)
            print(f"copied {src} → {dst}")
        return

    for name, url in GITHUB_RAW.items():
        dst = OUT / name
        if dst.exists() and not args.force:
            print(f"skip {dst} (exists)")
            continue
        print(f"download {url}")
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"wrote {dst} ({dst.stat().st_size} bytes)")
        except Exception as e:
            print(f"FAILED {name}: {e}")
            print("Clone https://github.com/ApolloResearch/deception-detection and re-run with --src-dir")


if __name__ == "__main__":
    main()
