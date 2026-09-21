# Directory Structure

Where things live, and where this repository deliberately departs from the
layout suggested in the assignment PDF.

---

## Layout

```text
README.md                  reproduction instructions, attribution
COLAB.md                   GPU workflow
progress.md                living record: state, decisions, bugs, evidence
DIRECTORY_STRUCTURE.md     this file
requirements.txt
.gitignore

common/                    used by all four tasks
  seed.py                  seed 6304, determinism, DataLoader seeding
  metrics.py               accuracy, macro-F1, consistency, shape bias, AUROC, thresholds
  config.py                YAML loader with inheritance and CLI overrides
  logging.py               run provenance (git SHA, versions, device), CSV/JSON output
  plotting.py              shared figure style, stable per-method colours

shared/                    Tasks 2 and 3 only
  pacs.py                  dataset access, TargetAccessViolation guard
  pacs_protocol.py         splits, transforms, domain-balanced batching, frozen BN
  mmd.py                   multi-kernel MMD (DAN and DAN-DG use the same code)
  make_splits.py           generates the committed split manifest
  splits/                  pacs_sketch_seed6304.json

task1/                     inductive biases and representations (STL-10)
  configs/                 base.yaml, interventions.yaml
  data/                    make_subset.py, transforms.py, make_cue_conflicts.py
  models/                  backbones.py (ResNet-50, ViT-B/16, CLIP), adain.py
  analysis/                evaluate_bias.py, representation.py
  scripts/run_task1.py     orchestrator
  results/

task2/                     unsupervised domain adaptation (PACS, Sketch target)
  configs/                 base + source_only, dan, dann, cdan, dan_lambda_study
  models/                  backbone.py, domain_discriminator.py (GRL, multilinear map)
  methods/                 base.py, source_only.py, dan.py, dann.py, cdan.py
  evaluation/              metrics.py, domain_separability.py
  train.py                 shared training loop for all four methods
  evaluate_final.py        target evaluation, run only after checkpoints are frozen
  results/

task3/                     domain generalization (Sketch unseen)
  configs/                 base + erm, dan_dg, sam, dan_dg_lambda_study
  methods/                 dan_dg.py, sam.py (SAMOptimizer + sharpness proxy)
  train.py                 refuses to retrain ERM
  evaluate_sketch.py       the ONLY script permitted to load Sketch
  results/

task4/                     open-set recognition (CIFAR-10 known)
  configs/                 base + vanilla, gcsc, proser, rpl
  data/cifar.py            CIFAR-10 splits, fixed CIFAR-100 unknown groups
  models/resnet_cifar.py   32x32 ResNet-18, dummy-classifier support
  methods/                 proser.py, rpl.py (stub)
  scores/posthoc.py        MSP, MLS, Energy, Mahalanobis, PROSER score
  train.py, extract_outputs.py, evaluate_osr.py
  cache/                   saved logits and features (not committed)
  results/

tests/                     322 tests, no datasets required
docs/                      design notes and explanations per module
notebooks/                 run_experiments.ipynb (invokes repo scripts only)
figures/                   generated figures
report/                    report-support artifacts
```

---

## Departures from the PDF's suggested layout

The PDF calls its per-task trees "suggested". Where this repository differs, it
is to avoid duplicating code that the assignment requires to be identical.

| PDF suggests | Here | Why |
|---|---|---|
| `task4/scores/msp.py`, `mls.py`, `energy.py`, `mahalanobis.py` | `task4/scores/posthoc.py` | Four one-function files; a single module keeps the shared "larger = more novel" convention visible in one place. |
| `task4/data/cifar10.py` + `cifar100_unknowns.py` + `make_splits.py` | `task4/data/cifar.py` | The known and unknown loaders share transforms and normalization constants. |
| `task3/models/backbone.py` | reuses `task2/models/backbone.py` | The assignment requires the *same* ResNet-18 initialization and head. A second copy could drift, and Task 2's Source-only checkpoint **is** Task 3's ERM baseline. |
| `task3/methods/erm.py` | none | ERM is not a separate method here: `task3/train.py` refuses to retrain it and loads Task 2's checkpoint instead. A module would invite exactly the retraining the assignment forbids. |
| `task3/evaluation/{domain_metrics,source_domain_separability,sharpness}.py` | `task2/evaluation/` + `task3/methods/sam.py` | Task 3 reuses Task 2's metric and separability code (`source_domain_separability` is the 3-way variant of the same function). The sharpness proxy lives beside SAM. |
| `task2/models/classifier_head.py` | inside `task2/models/backbone.py` | A single `nn.Linear` does not need its own module. |
| `task1/analysis/feature_similarity.py` | `task1/analysis/representation.py` | Cosine stability and the t-SNE visualization answer the same question and share inputs. |
| `task4/methods/{vanilla,gcsc}.py` | `task4/train.py` + configs | Vanilla and GCSC differ from each other *only* by RandAugment. Separate modules would imply a larger difference than exists and risk an uncontrolled second change. |
| `task4/methods/manifold_mixup.py` | inside `task4/methods/proser.py` | Manifold mixup is PROSER's data-placeholder mechanism and has no other caller. |

`task4/methods/rpl.py` exists as a **documented stub** that raises
`NotImplementedError`. RPL is the PDF's optional extension; the stub and its
config record the intended design without any risk of producing a number.

---

## What is not committed

Datasets, checkpoints (`*.pt`), cached features (`*.npz`), downloaded AdaIN
weights, and `__pycache__`. See `.gitignore`.

Small machine-readable results (`*.csv`, `*.json`) and figures **are**
committed, because the report depends on them and the assignment asks that
every reported number trace to a saved result file.
