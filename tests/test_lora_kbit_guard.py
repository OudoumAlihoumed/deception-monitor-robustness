"""fp16 LoRA must not treat full-precision models as k-bit (OOM on 70B)."""

from __future__ import annotations

from types import SimpleNamespace

from src.pipeline.train_lora import _is_kbit_model


def test_fp16_model_is_not_kbit():
    model = SimpleNamespace(
        is_loaded_in_4bit=False,
        is_loaded_in_8bit=False,
        config=SimpleNamespace(quantization_config=None),
    )
    model.named_parameters = lambda: [("w", SimpleNamespace())]
    assert _is_kbit_model(model) is False


def test_4bit_flag_detected():
    model = SimpleNamespace(
        is_loaded_in_4bit=True,
        is_loaded_in_8bit=False,
        config=SimpleNamespace(quantization_config=None),
    )
    model.named_parameters = lambda: []
    assert _is_kbit_model(model) is True
