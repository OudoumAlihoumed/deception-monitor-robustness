"""Shared Apollo Stage-2 extraction helpers (must match the Modal ladder path)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def assert_response_span_matches_completion(
    tokenizer,
    input_ids,
    response_mask,
    completion_text: str,
    *,
    prefix_chars: int = 50,
) -> dict[str, Any]:
    """
    Decode tokens under the response-only mask and require they start with the
    known completion. Catches chat-template / boundary drift on pre-supplied
    transcripts (mask starts a few tokens early → prompt leaks into the pool).
    """
    import torch

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)
    if not isinstance(response_mask, torch.Tensor):
        response_mask = torch.as_tensor(response_mask)
    ids = input_ids[0] if input_ids.ndim == 2 else input_ids
    mask = response_mask[0] if response_mask.ndim == 2 else response_mask
    keep = ids[mask.bool()]
    decoded = tokenizer.decode(keep, skip_special_tokens=True)
    expected = (completion_text or "").strip()
    got = decoded.strip()
    n = min(int(prefix_chars), len(expected))
    prefix = expected[:n]
    if n > 0 and not got.startswith(prefix):
        # Allow whitespace / quote drift of a few chars
        got_norm = " ".join(got.split())
        pref_norm = " ".join(prefix.split())
        if not got_norm.startswith(pref_norm[: max(n // 2, 20)]):
            raise AssertionError(
                "response-span decode does not start with completion prefix. "
                f"expected_prefix={prefix!r} decoded_start={got[:80]!r}"
            )
    return {
        "passed": True,
        "decoded_prefix": got[:80],
        "expected_prefix": prefix,
        "n_response_tokens": int(mask.sum().item()),
    }


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    return pearson_corr(ra, rb)


def extract_acts_like_ladder(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    probe_layer: int,
    *,
    layer_capture: Any = None,
    completion_text: Optional[str] = None,
    assert_span: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Same residual path as ``modal_app.residual`` / ladder:
    ``collect_residual_mean(..., response_only=True)`` → primary layer vector.

    Returns ``(activation, n_pooled_tokens)``.
    """
    from src.pipeline.residual import (
        collect_residual_mean,
        response_pool_prompt_len,
        response_attention_mask,
    )

    if assert_span and completion_text is None and messages and messages[-1].get("role") == "assistant":
        completion_text = messages[-1].get("content") or ""

    if assert_span and completion_text:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        pl = response_pool_prompt_len(
            tokenizer, messages, full_input_ids=inputs["input_ids"]
        )
        attn = inputs["attention_mask"]
        pool_mask = response_attention_mask(attn, pl, assert_excludes_prompt=True)
        assert_response_span_matches_completion(
            tokenizer, inputs["input_ids"], pool_mask, completion_text
        )

    vec, n_pooled = collect_residual_mean(
        model,
        tokenizer,
        messages,
        probe_layer,
        response_only=True,
        layer_capture=layer_capture,
        return_all_layers=False,
        return_n_pooled=True,
    )
    return np.asarray(vec, dtype=np.float64), int(n_pooled)


def mann_whitney_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    correct = 0.0
    for p in pos:
        correct += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return correct / (len(pos) * len(neg))
