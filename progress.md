# progress.md — ATML PA1 (Beyond IID)

**Living record.** Read before substantial work; update after meaningful work.

Status vocabulary: `ACTUAL` (real experiment/output exists) · `PRELIMINARY`
(limited/smoke evidence) · `HYPOTHESIS` (expected, not observed) · `UNVERIFIED`
(not independently checked).

**No experimental numbers appear anywhere in this file yet, because no
experiment has been run yet.**

---

## Current state — 2026-09-22

Repository skeleton, shared foundation, and all task configurations are built
and locally verified. **Zero training has been run.** No dataset has been
downloaded. No result, table, or figure exists.

| Layer | Status |
|---|---|
| Governance docs read (PDF + 5 md files) | `ACTUAL` |
| Directory skeleton | `ACTUAL` |
| `common/` foundation (seed, metrics, logging, config, plotting) | `ACTUAL` |
| Task 1–4 configs | `ACTUAL` |
| Test suite (322 tests) | `ACTUAL` — all passing, 1 skipped (needs AdaIN weights) |
| Shared PACS protocol (`shared/pacs.py`, `pacs_protocol.py`, `mmd.py`) | `ACTUAL` |
| Task 2 models + all 4 methods | `ACTUAL` — unit-tested, not yet trained |
| Task 1 backbones + interventions + AdaIN | `ACTUAL` — unit-tested |
| Task 2 `train.py` + evaluation | `ACTUAL` — all 4 methods smoke-tested end-to-end |
| Task 3 DAN-DG + SAM + `train.py` | `ACTUAL` — both smoke-tested end-to-end |
| Task 4 CIFAR ResNet + 4 scores + PROSER | `ACTUAL` — unit-tested |
| Task 1 eval + representation + orchestrator | `ACTUAL` |
| Task 2 `evaluate_final.py`, Task 3 `evaluate_sketch.py` | `ACTUAL` |
| Task 4 data, `train.py`, `extract_outputs.py`, `evaluate_osr.py` | `ACTUAL` — OSR pipeline verified on synthetic cache |
| README, COLAB.md, git init + 8 commits | `ACTUAL` |
| **ALL CODE COMPLETE** | `ACTUAL` — 38 modules, 322 tests |
| Data download / splits | **NOT STARTED** |
| Any experiment | **NOT STARTED** |
| Report artifacts | **NOT STARTED** |

---

## Decisions (locked with Talha, 2026-09-22)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Task 1 dataset: STL-10** | Assignment's recommended option; 10 distinct classes, lower compute, ample test images for the balanced 500-image subset. |
| D2 | **Controlled studies: λ_MMD (T2) + λ_DG (T3), matched** | Same MMD mechanism and same grid {0.1, 1, 10} on both sides, so Task 3 RQ4 (value of unlabeled Sketch) rests on matched curves rather than a single point. |
| D3 | **RPL: scaffold only** | Optional extension. Config + stub exist; `method.implemented: false`. Revisit only if required work finishes early. |
| D4 | **Git: local now, Talha creates GitHub remote** | Nothing leaves the machine until Talha says so. |

### Task 1 experimental-design choices (assignment leaves these to us)

| Choice | Selected | Rationale |
|---|---|---|
| Additional colour intervention | **Fixed hue rotation, 90°** | Cleanest counterpart to grayscale: grayscale *removes* colour, hue rotation *changes* it while provably preserving geometry, edges, luminance and saturation. Palette transfer and class-swapped statistics also perturb contrast and global statistics, confounding colour sensitivity with appearance sensitivity. |
| Cue-conflict method | **AdaIN**, α = 1.0, ≥5 unordered pairs, bidirectional, ≥200 valid | Follows Geirhos et al. (2019) and Huang & Belongie (2017). |
| Rejection rule | Visual, **defined before evaluation**, model-blind | Filtering on predictions would make the shape-bias measurement circular. |
| Visualization | **t-SNE**, perplexity 30, 1000 iters, PCA init, cosine metric, seed 6304 | t-SNE emphasises local-neighbour retention, which is exactly the question at issue: does a transformed image stay beside its clean counterpart? |

---

## Files created

**Root:** `.gitignore`, `requirements.txt`, `progress.md`

**`common/`** — `seed.py` (seed 6304 + determinism + DataLoader seeding),
`metrics.py` (accuracy, macro-F1, per-class, consistency, shape bias/coverage,
AUROC, threshold calibration, FPR@95TPR), `logging.py` (run provenance incl.
git SHA, CSV/JSON persistence, `MetricHistory`), `config.py` (YAML loader with
single-level inheritance + CLI overrides), `plotting.py` (shared matplotlib
style, stable per-method colours).

**`task1/configs/`** — `base.yaml`, `interventions.yaml`

**`task2/configs/`** — `base.yaml`, `source_only.yaml`, `dan.yaml`, `dann.yaml`,
`cdan.yaml`, `dan_lambda_study.yaml`

**`task3/configs/`** — `base.yaml`, `erm.yaml`, `dan_dg.yaml`, `sam.yaml`,
`dan_dg_lambda_study.yaml`

**`task4/configs/`** — `base.yaml`, `vanilla.yaml`, `gcsc.yaml`, `proser.yaml`,
`rpl.yaml`

**`shared/`** — `pacs.py` (dataset + `TargetAccessViolation` guard),
`pacs_protocol.py` (stratified splits, transforms, domain-balanced batching,
frozen-BN policy), `mmd.py` (multi-kernel MMD shared by Task 2 DAN and Task 3
DAN-DG)

**`task2/models/`** — `backbone.py` (ResNet-18 + 7-class head, exposes the
512-d pre-classifier feature), `domain_discriminator.py` (GRL, DANN schedule,
discriminator, CDAN multilinear map)

**`task2/methods/`** — `base.py` (shared interface), `source_only.py`,
`dan.py`, `dann.py`, `cdan.py`

**`tests/`** — `test_config.py`, `test_metrics.py`, `test_protocol.py`,
`test_pacs_protocol.py`, `test_methods.py`, `test_task2_methods.py`

---

## Bugs found and fixed

### B1 — `Config` attribute access shadowed by `dict` methods — FIXED `ACTUAL`

**Symptom.** `cfg.study.values` returned `<built-in method values>` instead of
`[0.1, 1.0, 10.0]`.

**Cause.** `Config` used `__getattr__`, which Python consults only when normal
attribute lookup *fails*. Keys colliding with real `dict` methods (`values`,
`keys`, `items`, `get`, `update`, …) therefore resolved to the bound method.

**Why it mattered.** A sweep driver reading `cfg.study.values` would have
iterated a method object — most likely skipping the controlled study silently,
or crashing far from the cause. This sat directly on decision D2.

**Fix.** Override `__getattribute__` so present keys win over dict methods;
absent keys now raise a descriptive `AttributeError` instead of returning
something falsy. Regression-guarded by parametrised tests over all colliding
names plus an end-to-end check on the real sweep config.

---

## Verification performed

- `python -m pytest tests/ -q` → **322 passed, 1 skipped** `ACTUAL`
- Metric sanity relationships confirmed: threshold calibration accepts exactly
  95% of knowns by construction; rejection + FPR@95TPR = 100; AUROC = 100 for
  perfect separation, 50 for identical distributions, 0 for reversed ranking.
- Config inheritance, deep-merge and CLI overrides verified on the real configs.

### Protocol constraints now mechanically enforced by tests

These are the assignment's "Before You Submit" requirements, encoded so a
future edit that violates one fails a test rather than producing a quietly
invalid result:

1. Tasks 2 and 3 share an identical protocol on 20 fields (splits, backbone,
   head, preprocessing, sampling, optimizer, budget, early stopping, seed) —
   required because Task 2's Source-only checkpoint **is** Task 3's ERM
   baseline.
2. Task 3 never streams Sketch during training (`target_batch == 0`,
   `target_access_forbidden == true`) under any method.
3. Task 2 checkpoint selection uses mean source-validation macro-F1 only.
4. CDAN's forbidden options (entropy conditioning, detaching `f` or `p`) are off.
5. Task 4 thresholds calibrate on CIFAR-10 validation only; CIFAR-100 is
   evaluation-only and the near/far grouping is the fixed 16 classes.
6. GCSC differs from Vanilla by RandAugment **and nothing else**.
7. Task 1's cue-conflict rejection rule is model-blind.
8. Matched λ studies use an identical grid and MMD construction.

### Behavioural properties verified (not just config values) `ACTUAL`

- **Frozen BN actually works**: BN running mean/var do not move under a
  distribution-shifted forward pass, while γ/β still receive gradients. A
  control test confirms plain `model.train()` *does* move them, so the test is
  testing the policy rather than passing vacuously.
- **MMD is a real discrepancy**: ~0 for identical samples, non-negative,
  monotonically increasing with distributional separation, symmetric,
  differentiable w.r.t. both inputs, and finite on collapsed features.
- **GRL negates gradients** by exactly −α; α=0 blocks them entirely.
- **DANN at p=0 → α=0 → zero target-feature gradient**, confirming the
  schedule lets the representation learn the class task before adversarial
  pressure ramps up.
- **CDAN does not detach** `f` or `p`: both the backbone and the classifier
  head receive adversarial gradient (discriminator input = 3584-d).
- **Source-only refuses target data** outright, protecting the shared baseline.
- **Domain-balanced batching**: every update carries exactly 8 examples per
  source domain; shorter domains cycle rather than truncating the epoch.

---

## Not yet done

**The code is complete.** What remains needs data or GPU:

- **Datasets**: PACS must be downloaded manually (no torchvision loader) and
  placed at `data/PACS/<domain>/<class>/`. STL-10 and CIFAR auto-download but
  are slow on this connection (~40 kB/s measured); they download fast on Colab.
- **AdaIN weights**: blocked by the sandbox here, download normally on Colab.
- **All real experiments** — see COLAB.md for the run order.
- Talha to create the public GitHub repo and push.
- Report-support artifacts once real results exist.
- **All data handling**: STL-10, PACS, CIFAR-10/100 downloads; split manifests
  (`shared/splits/pacs_sketch_seed6304.json`, Task 1 eval subset, Task 4 split).
- **All experiments.**
- `README.md`, `AGENT.md`, `DIRECTORY_STRUCTURE.md`, `COLAB.md`, Colab notebook.
- Git init and first commit.
- Report-support artifacts (ledgers, tables, figures, RQ evidence map).

---

## Blockers and risks

| # | Issue | Severity | Note |
|---|---|---|---|
| R1 | **Local machine has no CUDA** (`torch 2.9.1+cpu`, no `nvidia-smi`) | Expected, not a blocker | Matches the intended design: local = build/test, Colab = train. All real runs must go to Colab. |
| R2 | **Deadline 2026-09-25, three days out** | **HIGH** | Task 4 alone is 3 × 100-epoch CIFAR runs; Tasks 2–3 are 6 main + 6 sweep PACS runs. Colab GPU time is the binding constraint. Ordering of work matters. |
| R3 | `open_clip` and `umap-learn` not installed locally | Low | In `requirements.txt`; needed only when Task 1 code runs. |
| R4 | Talha must create the public GitHub repo | Low | Required by the assignment and by the Colab clone step. |

---

## Research-question evidence map

No evidence exists yet. Every cell is pending by construction.

| Task | RQ | Experiment | Artifact | Status |
|---|---|---|---|---|
| 1 | RQ1 colour + cue conflict → shape/texture/colour reliance; role of coverage | grayscale, hue-90, AdaIN cue conflict | `task1/results/` | pending |
| 1 | RQ2 translation + patch shuffle → locality, global structure, position | translation curve, 4×4 shuffle | `task1/results/`, `figures/task1/` | pending |
| 1 | RQ3 prediction vs representation change; CLIP zero-shot vs head | cosine stability, t-SNE | `task1/results/` | pending |
| 1 | RQ4 architecture vs pretraining/supervision attribution | all Task 1 + failure cases | interpretation notes | pending |
| 2 | RQ1 size of source→target gap; worst classes | Source-only ERM | `task2/results/` | pending |
| 2 | RQ2 does lower domain separability track target accuracy? | all 4 methods + separability | `task2/results/` | pending |
| 2 | RQ3 class-conditional (CDAN) vs marginal (DAN/DANN) | CDAN vs DAN/DANN | per-class target analysis | pending |
| 2 | RQ4 effect of alignment strength; target-free selection | λ_MMD ∈ {0.1, 1, 10} | `task2/results/` | pending |
| 3 | RQ1 do mean/worst source predict Sketch? | ERM/DAN-DG/SAM source vs Sketch | `task3/results/` | pending |
| 3 | RQ2 does source invariance help Sketch? | DAN-DG + source separability | `task3/results/` | pending |
| 3 | RQ3 does SAM reduce the sharpness proxy; does the ranking agree? | sharpness proxy, all 3 models | `task3/results/` | pending |
| 3 | RQ4 value of unlabeled Sketch (DAN vs DAN-DG) | **matched λ studies (D2)** | `task2/` + `task3/` | pending |
| 4 | RQ1 semantic similarity vs rejection; which labels absorb unknowns | near/far AUROC + failures | `task4/results/` | pending |
| 4 | RQ2 what MSP/MLS/Energy/Mahalanobis capture | 4 scores, one frozen model | `task4/results/` | pending |
| 4 | RQ3 does GCSC augmentation help CSA, OSR, both, neither? | Vanilla vs GCSC | `task4/results/` | pending |
| 4 | RQ4 does PROSER improve rejection; CSA trade-off | Vanilla/GCSC/PROSER | `task4/results/` | pending |
| Cross | Synthesis: preserve / suppress / reject | all four tasks | `report/` | pending |

---

## Repository state

38 source modules, 322 tests passing (1 skipped: needs AdaIN weights), 8
structured git commits, clean working tree. No datasets, checkpoints or
downloaded weights are tracked.

## Next action

**Talha's directive (2026-09-22): build the ENTIRE repository's code first,
then run all training at once.** That is now DONE.

1. **Talha**: create the public GitHub repo, then `git remote add origin <url>`
   and `git push -u origin main`.
2. **Talha**: obtain PACS and place it at `data/PACS/` (or on Drive for Colab).
3. Open Colab, follow COLAB.md: verify GPU, run the test suite, smoke-test,
   then run the two dependency chains first
   (Task2 source_only → Task3 ERM; Task4 vanilla → PROSER).
4. Sync results back, then build report-support artifacts from REAL numbers.

### Task 4 OSR pipeline verified on a synthetic cache `PRELIMINARY`

CIFAR downloads at only ~40 kB/s on this connection (170 MB would take over an
hour), so the OSR evaluation was exercised with synthetic cached outputs
instead. Every metric behaved as designed: known acceptance landed at ~95% for
all four scores (threshold calibration works by construction), rejection +
FPR@95TPR summed to 100 in every row, near unknowns scored consistently harder
than far (MLS 98.42 vs 99.99 AUROC), and Mahalanobis reached 100 on
feature-shifted unknowns, which is exactly the signal it measures. Failure-case
extraction, the 3-panel score-distribution figure and all CSV outputs were
produced. **These numbers come from synthetic data and mean nothing about real
models** — the test was of the pipeline.

### Smoke tests on synthetic PACS `PRELIMINARY`

All four Task 2 methods and both Task 3 methods were run end-to-end through
their real training scripts on a synthetic PACS tree (7 classes x 4 domains x
6 random images). Splits came out exactly 80/20 (35 train / 7 val per domain).
Every loss component was finite and method-specific terms appeared as intended:
DAN logged `mmd_loss`, DANN/CDAN logged `domain_acc` near 0.5 with the ramped
`alpha`, DAN-DG logged all three source pairs separately. Checkpoints,
per-domain training curves and run records with git provenance were all
written. **The accuracy numbers are meaningless** (random pixels, 1 epoch,
3 batches) — this tested wiring, not learning.

The Task 3 ERM guard was confirmed to fire: `task3/train.py --config erm.yaml`
refuses to train and points at Task 2's Source-only checkpoint.

### SAM verified against its defining properties `ACTUAL`

Perturbation norm exactly 0.050000 (= rho) in global L2; the step follows the
ascent direction; loss genuinely rises at theta+eps; parameters are restored
before the base update; single-pass misuse raises. The sharpness proxy leaves
the model bit-identical, clears gradients, restores training mode, and grows
with rho.

### B4 — sklearn `multi_class` deprecation `ACTUAL`

`LogisticRegression(multi_class=...)` is deprecated in sklearn 1.5 and REMOVED
in 1.7. Colab ships newer sklearn than this machine, so the 3-way source
separability diagnostic would have broken there. Removed the argument (the
default is already multinomial) before it could fail remotely.

### D5 — Code/explanation separation (Talha's instruction, 2026-09-22)

All inline comments and rationale prose were stripped from source files; only
one-line docstrings remain. Every explanation now lives in `docs/` — one file
per module (`docs/common.md`, `docs/shared.md`, `docs/task2.md`, …), written as
teaching material mapped back to the code. Source shrank from 1856 → 1381
lines with zero behaviour change (all tests still passed immediately after).
Test files were deliberately NOT stripped: they document intent and are meant
to be read.

### B2 — Hue-rotation geometry test measured the wrong channel `ACTUAL`

**Symptom.** `test_hue_rotation_preserves_geometry` failed: mean luma
difference 30.1 against a threshold of 5.

**Diagnosis.** The transform was correct; the test premise was wrong.
`PIL.convert("L")` computes ITU-R 601-2 luma = 0.299R + 0.587G + 0.114B, a
channel-WEIGHTED mix. HSV's V is `max(R,G,B)`. Rotating hue permutes which
channel holds the max, so luma necessarily moves even when shape content is
untouched. Verified on a minimal case: pure red (255,0,0) → green (0,255,0)
keeps **V at 255 exactly** while luma shifts 76 → 150.

**Confirmation the transform is right.** V-channel gradient energy before and
after a 90° rotation is identical to three decimals (108.765 / 108.765), i.e.
edge structure is genuinely preserved.

**Fix.** Test now measures geometry on the V channel and compares gradient
energy within 2%. Relevant to Task 1 interpretation: **grayscale and hue
rotation are not directly comparable through luma** — grayscale collapses to
luma, hue rotation preserves V.

### B3 — Minor fixes found while testing Task 1 transforms `ACTUAL`

- Non-writable NumPy array passed to `torch.from_numpy` raised a UserWarning
  about undefined write behaviour; fixed with an explicit `.copy()`.
- PIL's 8-bit RGB↔HSV roundtrip is lossy by up to ~7 levels **even at zero
  shift**. Measured, documented in the docstring and given a named test
  tolerance rather than silently corrected.
- Removed an unused `ImageEnhance` import.

### Bug found in my own test, not the code `ACTUAL`

`test_mmd_is_differentiable_wrt_both_inputs` originally wrote
`torch.randn(..., requires_grad=True) + 2.0`, producing a **non-leaf** tensor
whose `.grad` is never populated — so the assertion about the second argument
was vacuous and the test failed. Fixed to build a leaf tensor via
`.requires_grad_(True)`, with an explicit `is_leaf` assertion. MMD itself was
correct throughout.

**Sequencing note (R2).** Task 2's Source-only checkpoint is a hard dependency
for Task 3's ERM baseline, and Task 4's Vanilla checkpoint is a hard dependency
for PROSER. Both chains must start early; they cannot be parallelised away.
