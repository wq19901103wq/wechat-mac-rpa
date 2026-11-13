# Human / AI Judge Benchmark V2

> **历史口径，不再控制执行。** 现行唯一流程见 `JUDGE_C0_CANONICAL_WORKFLOW.md`。本文中的 dev 50 / validation 30 / final 100 等数字仅保留作历史证据。

## Scope

This benchmark only measures whether an isolated Judge can distinguish a pre-April-2026 human reply from a protected-C0 reply. It does not measure Bot quality and must not replace C0 or start the Bot.

## Frozen baseline

- Protected C0 SHA256: `625b47803fff55a2e18914c5e734fcb2f01ae7c16e5180fdebdc817badd2656b`
- Split manifest SHA256: `9f5e5916275e2756ee8cd18215efd41443d91d8e9d05a4c1b87e93d8889b4a11`
- Baseline Judge profile SHA256: `39a1ab46a5ce85985e38ff1838f207d63f53bbf058ed2b0ba882e6dcae2b7ed2`
- Data: dev 50, validation 30, final 100.
- Chats never cross splits. Judge references exclude every test chat and target human reply.

## Isolation

Each case uses a fresh model API client and one request. The request contains one blind case and the same frozen reference pack. It contains no answer key, option identity, other case, tool definition, shared conversation, Codex history, or filesystem access. Canonical and swapped option orders are separate calls.

The answer key is opened only by a separate scoring command after predictions are written. Dev may show case-level errors. Validation and final may show aggregates only. Final is limited to one resumable invocation with one frozen Judge design.

## Iteration loop

1. Run the unchanged baseline on dev in canonical and swapped order.
2. Attribute dev errors using full context and raw responses.
3. Change one hypothesis only, then rerun the same dev protocol.
4. Keep a candidate only when both option orders improve and order consistency does not regress.
5. Use validation aggregates only to choose among dev-qualified candidates.
6. Freeze the chosen profile and model before opening final.
7. Run final once and report the observed score without retuning.

Stop an iteration direction after two consecutive candidates fail to improve both dev orders. Never compensate for weak discrimination by exposing answers, adding tools, sharing cases, or repeatedly testing final.

## Hypothesis order

1. Prefer contextual and response-target fit over superficial length, punctuation, or generic “AI tone”.
2. Use reference material as evidence of stable personal habits, not text to copy or a source of current-case facts.
3. Distinguish natural humor from factual claims; treat unsupported personal facts as unknown rather than automatically false.
4. Add a short evidence check before choosing, especially when both options are plausible.
5. Consider a model change only after prompt-level hypotheses plateau; compare accuracy, order consistency, latency, and failures under the same protocol.

## Reported metrics

- Accuracy overall and by private/group and ordinary/hard-surface strata.
- Canonical/swapped agreement on dev and validation.
- Invalid response and API failure counts.
- Final target for a usable Judge: at least 75/100 overall and at least 65% in each reported stratum. These are engineering gates, not human ground truth and not evidence that Bot quality improved.
