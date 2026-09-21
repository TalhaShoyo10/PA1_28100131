# Agent Rules — AToML PA1

- Assignment PDF is authoritative.
- Do not fabricate results.
- Preserve exact experimental information boundaries.
- Keep seeds, splits, preprocessing, checkpoints, and hyperparameters explicit.
- Make experiments reproducible.
- Do not silently weaken the assignment because of compute.
- Record deviations explicitly.
- Teach Talha what the code is doing.
- Do not write the final report on Talha's behalf.

## Experiment loop

Requirement → hypothesis → controls/variables → implementation → smoke test → real run → saved outputs → diagnostics → evidence-based interpretation → progress update.

Keep separate:
- requirement
- implementation choice
- hypothesis
- result
- interpretation
- alternative explanation
- limitation

A lower MMD/domain-separability score is not automatically evidence of better target recognition.

## Colab

Use Colab for expensive computation, not as a separate codebase. Notebook cells should mainly clone/mount, install, verify GPU, prepare data, invoke repository scripts, and persist outputs.

## Context retention

Every substantive agent update should refer to Talha by name.
