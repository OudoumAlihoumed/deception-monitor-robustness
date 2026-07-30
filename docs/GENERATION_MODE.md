# Generation mode (scope correction)

As of `judge_version` `def1-scheurer-benton-v1` / suite freeze:

`_generate` in `scripts/modal_app.py` builds:

```
system → user_turn_0 → user_turn_1 → … → user_turn_k → [assistant generation]
```

There are **no interleaved assistant replies**. The model produces one completion after a stack of user messages.

## Consequences for the paper

- Episodes are **not** true multi-turn agentic dialogues.
- Definition 1’s deceptive turn \(t^\star\) is localized inside that **single** assistant reply (often answering mainly the last user turn).
- Figure 1C / Methods must describe this as **multi-prompt single-shot** (or fix generation to true turn-by-turn before claiming multi-turn).

This is independent of the Scheurer/Benton judge fix.
