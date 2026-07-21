# Unlocking official Llama-3.3-70B on Hugging Face

## Short answer

**Llama 3 / 3.1 / 3.3 70B Instruct are on Hugging Face.**  
They live under `meta-llama/*` and are **gated by Meta’s license**.  
Your account (`Oudoum`) can *see* the repos but **cannot download** until you accept the license.

| Repo | Status on HF | Your download |
|------|----------------|---------------|
| [`meta-llama/Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Exists (~690k downloads) | **Blocked** (GatedRepoError) |
| [`meta-llama/Llama-3.1-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) | Exists | **Blocked** |
| [`meta-llama/Meta-Llama-3-70B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) | Exists (original Llama 3) | **Blocked** |
| [`unsloth/Llama-3.3-70B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-bnb-4bit) | Ungated NF4 mirror | **Works** (current Modal runs) |

## What you must do (2 minutes)

1. Open **https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct** while logged in as **Oudoum**.
2. Click **Agree and access repository** (fill Meta’s form if asked).
3. Wait until the page shows you have access (sometimes instant, sometimes a short delay).
4. Tell me (or re-run): the Modal app now **auto-prefers** the official model when download works, and falls back to Unsloth otherwise.

```bash
# quick local check after you accept:
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('meta-llama/Llama-3.3-70B-Instruct','config.json'))"
```

## Why Modal used Unsloth

Meta’s official weights returned `403 GatedRepoError` for this token.  
Unsloth’s checkpoint is the **same Llama-3.3-70B-Instruct architecture** in NF4 — scientifically valid for the white-box probe protocol, with the license situation disclosed in the paper Limitations section.

After you unlock Meta, the next Modal run will load `meta-llama/Llama-3.3-70B-Instruct` automatically.
