# Archived Qwen3-32B experiment (before Llama-3.3-70B redo)

**Do not treat these as the workshop primary results.**  
This folder freezes the last Qwen run so the new Llama-70B pipeline can overwrite `results/` safely.

## Run metadata
- **model:** `Qwen/Qwen3-32B`
- **result_type:** `modal_qwen3_32b_experiment`
- **n_sessions:** `60` (n_test=30)
- **probe_layer:** `40`
- **gpu:** `H200`
- **timestamp_utc:** `2026-07-20T13:18:07.693633+00:00`
- **sft:** `{
  "skipped": false,
  "steps": 200,
  "rl_steps": 100,
  "lora_params": 41943040,
  "lambda_probe": 0.5,
  "ft_seed": 0,
  "rungs": [
    "b1",
    "b2",
    "b3",
    "b4"
  ]
}`
- **rule_label_agreement_b0:** `0.7`

## recall@1%FPR
```json
{
  "surface": {
    "b0": 0.13333333333333333,
    "b1": 0.2,
    "b2": 0.0,
    "b3": 0.0,
    "b4": 0.26666666666666666
  },
  "linear_probe": {
    "b0": 0.5333333333333333,
    "b1": 0.26666666666666666,
    "b2": 0.2,
    "b3": 0.5333333333333333,
    "b4": 0.4
  },
  "cot_reading": {
    "b0": 0.5333333333333333,
    "b1": 0.26666666666666666,
    "b2": 0.2,
    "b3": 0.06666666666666667,
    "b4": 0.6666666666666666
  }
}
```

## Known science issues in this run (why we archive, not cite as final)
1. Probe adapter returned sigmoid (weak b3 softplus).
2. Train/eval used intended_label, not rule_label.
3. CoT features included system prompt leak.
4. b4 continued from b3 LoRA instead of fresh base.
5. TOST retention used response length, not deception success.
6. Single FT seed, single λ=0.5 (not 3×3 grid).
7. Wrong primary model vs Apollo (Qwen vs Llama-3.3-70B).

## Files
- `modal_experiment_results.json` — full metrics dump
- `modal_qwen3_*.log` / `modal_full_run.log` — Modal logs from this campaign

## Next
New primary runs go to `results/modal_experiment_results.json` under Llama-3.3-70B after P0 fixes.
