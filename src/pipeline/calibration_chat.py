"""Public-chat calibration negatives (Apollo-style 1% FPR on unrelated chat)."""

from __future__ import annotations

from typing import Optional, Sequence


def load_alpaca_user_prompts(n: int = 300, seed: int = 0) -> list[str]:
    """
    Sample Alpaca instructions as control prompts (unrelated to deception scenarios).

    Used only for FPR calibration / thresholding — not for Δdet paired eval.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets to load Alpaca calibration prompts") from e

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    # Stable subsample
    idxs = list(range(len(ds)))
    rng_idxs = idxs[:: max(1, len(idxs) // max(n * 3, 1))][: n * 3]
    prompts: list[str] = []
    for i in rng_idxs:
        row = ds[int(i)]
        instr = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        if inp:
            text = f"{instr}\n{inp}"
        else:
            text = instr
        if text:
            prompts.append(text)
        if len(prompts) >= n:
            break
    if len(prompts) < n:
        # wrap
        while len(prompts) < n and prompts:
            prompts.append(prompts[len(prompts) % len(prompts)])
    return prompts[:n]


def alpaca_as_chat_turns(prompts: Sequence[str]) -> list[list[str]]:
    """Each Alpaca prompt → single user turn list for our elicit API."""
    return [[p] for p in prompts]
