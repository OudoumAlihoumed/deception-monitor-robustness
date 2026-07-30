# Artifact contract (every GPU run)

Burned twice by summary-only commits (no `response_b0` → CP2 blind; no per-episode scores → CP6 CI unrecoverable).
Burned again 2026-07-25: Alpaca calib n=1000 + full b1 finished, then OOM at b2 — **neither calib scores nor `response_b1` were on the Volume**, so resume could not reuse them. Root cause: mid-run commits only covered slim `episodes_b0.jsonl`; calib/b1 were deferred to end-of-run.

**Rule:** if a stage spent GPU time, it must `write + results_vol.commit()` before the next stage starts. Tracked in `/results/checkpoints/stage_progress.json`.

## Mid-run commits (mandatory — survive OOM)

| After stage | path(s) | when |
|---|---|---|
| b0 elicit | `episodes_b0.jsonl` (**every 4** partial) + acts + **`frozen_cohort.json`** on final | survive mid-b0 crash |
| Alpaca calib | `chat_calib_cache.json` | **every 250** + final |
| thresholds | `checkpoints/thresholds.json` | pin τ; resume reloads |
| b1 elicit | `episodes_b0_b1.jsonl` | **every 4** + final acts |
| each SFT/RL cell | `checkpoints/cells/{cell_id}.json` | atomic + field validation |

`--no-run-full-grid` = **one seed**, not one λ. Pass `--lambda-sweep 0.5,8.0` to run both.

Every commit also appends to `checkpoints/stage_progress.json` (`[stage-commit] …` in logs).
JSON/JSONL writers use **temp → `os.replace`** (atomic). Resume validates parse +
required fields (not mere file existence).

**Resume invariants:** \(\mathcal{C}\) and \(\tau\) come from disk (`source=reloaded`);
b3/b4 always `_reload_base()` + fresh `_make_adapter()` (never inherit in-memory LoRA).

Helpers: `src/eval/stage_checkpoints.py`. Volume commit: `_stage_commit()` in `scripts/modal_app.py`.

## Slim episodes — `/results/episodes_b0.jsonl`

One JSON object per episode:

| field | required | notes |
|---|---|---|
| `session_id` | yes | |
| `scenario_id` | yes | |
| `family` | yes | |
| `intended_label` | yes | 0/1 |
| `rule_label_b0` | yes | 0/1 |
| `failure_mode` | yes (nullable) | `refusal` / `honest` / `incomplete` / `null` if success or benign |
| `response_b0` | yes | full assistant text |
| `is_benign` | yes | |
| `frozen` | yes (ladder) | membership in frozen deceptive cohort |

## Per-episode scores — `/results/scores_by_rung.jsonl`

| field | required |
|---|---|
| `session_id` | yes |
| `frozen` | yes |
| `rule_label_b0` | yes |
| `failure_mode` | yes (nullable) |
| `scores.{surface,linear_probe,cot_reading}.{b0…b4}` | yes on ladder runs |

## Results JSON — `/results/modal_experiment_results.json`

Must include: `n_frozen_cohort`, `elicitation.per_family`, `elicitation.failure_modes`, `table_auroc`, `table_auroc_ci_b0`, `label_stats_by_rung`, `smoke_gate`, `judge`, `artifact_contract`.

## Judge identity (load-bearing)

Every commit must include `judge` from `src.scenarios.judge_version.judge_identity()`:

| field | purpose |
|---|---|
| `judge_version` | human-readable pin (e.g. `def1-scheurer-benton-v1`) |
| `judge_content_hash` | hash of label-helper source — silent edits change it |
| `family_set_version` | suite pin |
| `included_families` / `excluded_families` | Methods accounting |

Do **not** change the judge after activations for a cohort are extracted.

## Enforcement

`src.eval.artifact_contract.assert_run_artifacts(...)` runs **before** the final volume commit in `run_experiment`. Failure aborts the run (no silent summary-only artifact).

b0 / calib-final / b1-final stage commits **hard-fail** if write+commit raises (run will not continue past an unsaved stage).
