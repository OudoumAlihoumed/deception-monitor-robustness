# Unlocking official Llama-3.3-70B on Hugging Face

## Status

**Meta license approved** for account access to
[`meta-llama/Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct).

The Modal app **prefers the official Meta checkpoint** and only falls back to
Unsloth NF4 if the official download fails.

## Quick check

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('meta-llama/Llama-3.3-70B-Instruct','config.json'))"
```

## Modal

Ensure the Modal secret `huggingface` still holds a token for the approved account:

```bash
modal secret create huggingface HF_TOKEN=hf_... --force
```

Primary model in `configs/experiment.yaml` and `scripts/modal_app.py`:
`meta-llama/Llama-3.3-70B-Instruct` — **fp16, no 4-bit** (matches Apollo
`models.py`). Needs **2×H200** (`GPU = "H200:2"`).