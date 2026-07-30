# Modal outputs on this laptop

## Folders

| Path | What |
|------|------|
| `results/modal_laptop_bundle/snapshot_*/` | Frozen copy of **all Modal GPU code/config/probes** |
| `results/modal_laptop_bundle/latest` | Symlink to newest snapshot |
| `results/modal_runs/<run_id>/` | Per-run JSON after `modal run` (auto-saved) |
| `results/modal_experiment_results.json` | Latest run (overwrite) |

## Save implementation now (no GPU)

```bash
python scripts/sync_modal_to_laptop.py snapshot
```

## After any Modal run — pull volume (cells + JSON)

```bash
python scripts/sync_modal_to_laptop.py pull
# or both:
python scripts/sync_modal_to_laptop.py both
```

`modal run` also auto-writes a timestamped folder under `modal_runs/` when `download=True`.

## Volume name

`deception-monitor-results` (Modal)
