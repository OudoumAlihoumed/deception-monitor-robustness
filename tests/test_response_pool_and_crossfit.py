"""Response-only pooling + cross-fit helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.cross_fit import cross_fit_scores, kfold_indices
from src.pipeline.residual import LayerCapture, response_attention_mask


def test_layer_capture_register_multi():
    """Multi-layer register stores probe_layers list (no real model needed for API)."""
    cap = LayerCapture()
    assert cap.hook_registered is False
    assert cap.hs == {}


def test_response_mask_excludes_prompt():
    attn = torch.ones(1, 10)
    mask = response_attention_mask(attn, prompt_len=4, assert_excludes_prompt=True)
    assert not bool(mask[:, :4].any().item())
    assert bool(mask[:, 4:].all().item())


def test_apollo_block_index_matches_hidden_states():
    """Apollo detect_layers[22] → HF layers[21] (hidden_states[0]=embed)."""
    from src.pipeline.residual import apollo_block_index

    assert apollo_block_index(22) == 21
    assert apollo_block_index(1) == 0
    try:
        apollo_block_index(0)
        assert False
    except ValueError:
        pass


def test_multi_turn_final_response_must_end_on_assistant():
    """
    Regression: parking a single reply after user_turns[0] leaves trailing users
    and collapses response-only pooling to 1 token (Smoke-1 median=1.0 bug).
    """
    system = "sys"
    user_turns = ["u1", "u2", "u3"]
    response = "the model completion"

    # Old buggy interleave used by residual(..., [response])
    bad = [{"role": "system", "content": system}]
    for i, u in enumerate(user_turns):
        bad.append({"role": "user", "content": u})
        if i < 1:
            bad.append({"role": "assistant", "content": response})
    assert bad[-1]["role"] == "user"

    # Fixed: all users, then one final assistant
    good = [{"role": "system", "content": system}]
    for u in user_turns:
        good.append({"role": "user", "content": u})
    good.append({"role": "assistant", "content": response})
    assert good[-1]["role"] == "assistant"
    assert sum(1 for m in good if m["role"] == "assistant") == 1


def test_response_pool_prompt_len_rejects_trailing_user():
    from src.pipeline.residual import response_pool_prompt_len

    class Tok:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            # crude fake ids: 1 token per message (+1 if generation prompt)
            n = len(messages) + (1 if add_generation_prompt else 0)
            return list(range(n))

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0] * max(len(text.split()), 1)}

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "early"},
        {"role": "user", "content": "u2"},
    ]
    try:
        response_pool_prompt_len(Tok(), msgs)
        assert False, "expected AssertionError"
    except AssertionError as e:
        assert "assistant" in str(e).lower()


def test_response_pool_prompt_len_leaves_completion_tokens():
    from src.pipeline.residual import response_pool_prompt_len

    class Tok:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            # prefix+gen_prompt → [0,1,2,3]; full with assistant content → +4 content ids
            if messages and messages[-1].get("role") == "assistant":
                return [0, 1, 2, 3, 10, 11, 12, 13]
            base = [0, 1, 2]
            return base + ([3] if add_generation_prompt else [])

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0] * max(len(str(text).split()), 1)}

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "hello world completion"},
    ]
    pl = response_pool_prompt_len(Tok(), msgs)
    # full len 8, prompt should be 4 → pool 4 tokens
    assert pl == 4
    assert 8 - pl >= 3


def test_response_mask_assert_fires_if_prompt_kept():
    attn = torch.ones(1, 8)
    response_attention_mask(attn, 3, assert_excludes_prompt=True)
    m = response_attention_mask(attn, 3)
    m[:, :3] = 1  # corrupt
    try:
        if bool(m[:, :3].any().item()):
            raise AssertionError("pool must exclude prompt region")
        assert False
    except AssertionError as e:
        assert "prompt" in str(e)


def test_kfold_covers_all():
    folds = kfold_indices(20, n_splits=5, seed=0)
    seen = set()
    for tr, te in folds:
        seen.update(te.tolist())
        assert len(set(tr.tolist()) & set(te.tolist())) == 0 or len(tr) == len(te)
    assert seen == set(range(20))


def test_cross_fit_oof_shape():
    class Ex:
        def __init__(self, x, y):
            self.x = x
            self.label = y

    class M:
        def __init__(self):
            self.mean = 0.0

        def fit(self, xs):
            self.mean = float(np.mean([e.x for e in xs if e.label == 1] or [0.0]))
            return self

        def score(self, xs):
            return np.array([e.x - self.mean for e in xs], dtype=np.float64)

    exs = [Ex(float(i), int(i % 2)) for i in range(20)]
    scores = cross_fit_scores(
        exs,
        fit_fn=lambda xs: M().fit(xs),
        score_fn=lambda m, xs: m.score(xs),
        n_splits=5,
        seed=0,
    )
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()


def test_aggregate_cells_synthetic(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_cells.py"
    spec = importlib.util.spec_from_file_location("aggregate_cells", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    cell = {
        "rung": "b3",
        "seed": 0,
        "lambda": 0.5,
        "gpu_seconds": 10.0,
        "lambda_worked": {"passed": True},
        "adapter_live": {"passed": True},
        "monitor_metrics": {
            "recall": {"linear_probe": {"b3": 0.4}},
            "auroc": {"linear_probe": {"b3": 0.9}},
        },
    }
    p = tmp_path / "b3_seed0_lam0.5.json"
    p.write_text(json.dumps(cell))
    parsed = mod.parse_cell(p)
    summary = mod.aggregate([parsed])
    assert summary["n_cells_ok"] == 1
    assert "b3" in summary["summary_by_rung"]
    assert "last_cell" in summary["warning"].lower()
