# Beyond IID — ATML PA1

Experiments on learning beyond the IID, closed-set setting: inductive biases and
representations (Task 1), unsupervised domain adaptation (Task 2), domain
generalization (Task 3), and open-set recognition (Task 4).

Every experiment is configuration-driven, seeded with **6304**, and writes
machine-readable results that trace back to the command and commit that
produced them.

---

## Quick start

```bash
git clone <repository-url>
cd pa1-beyond-iid
pip install -r requirements.txt
python -m pytest tests/ -q          # 322 tests, no datasets required
```

Run any task with an explicit config:

```bash
python task2/train.py --config task2/configs/dan.yaml
```

Add `--smoke` for a fast reduced-scale sanity check (never for reported results).

---

## Repository layout

```text
common/      seed, metrics, logging, config loader, plotting
shared/      PACS dataset + protocol, multi-kernel MMD  (Tasks 2 and 3)
task1/       inductive biases: interventions, backbones, representation analysis
task2/       UDA: Source-only, DAN, DANN, CDAN
task3/       DG: ERM (reused), DAN-DG, SAM
task4/       OSR: Vanilla, GCSC, PROSER, post-hoc scores
tests/       322 tests covering protocol compliance and method behaviour
docs/        design notes and explanations for each module
figures/     generated figures
report/      report-support artifacts
```

Raw datasets and checkpoints are **not** committed. See *Datasets* below.

---

## Datasets

| Dataset | Used by | How to obtain |
|---|---|---|
| STL-10 | Task 1 | Auto-downloads via torchvision to `data/stl10` |
| PACS | Tasks 2, 3 | **Manual** — see below |
| CIFAR-10 / CIFAR-100 | Task 4 | Auto-downloads via torchvision to `data/cifar10`, `data/cifar100` |

### PACS

PACS has no torchvision loader. Download it (e.g. from the public Kaggle mirror)
and arrange it as:

```text
data/PACS/
  photo/dog/*.jpg
  art_painting/dog/*.jpg
  cartoon/dog/*.jpg
  sketch/dog/*.jpg
  ... (7 classes x 4 domains)
```

Then generate the shared split manifest, which both Tasks 2 and 3 reuse:

```bash
python shared/make_splits.py --config task2/configs/base.yaml
```

### AdaIN weights (Task 1 cue conflicts)

Pretrained VGG encoder and decoder weights (~110 MB) download automatically on
first use, or fetch manually — see [docs/adain_weights.md](docs/adain_weights.md).

---

## Reproducing each task

### Task 1 — Inductive Biases and Representations

STL-10, three frozen backbones (ResNet-50, ViT-B/16, CLIP ViT-B/32) plus CLIP
zero-shot. Interventions are generated **once** on a common 224x224 canvas so
every model receives byte-identical images.

```bash
python task1/data/make_subset.py        --config task1/configs/base.yaml
python task1/data/make_cue_conflicts.py --config task1/configs/interventions.yaml
python task1/scripts/run_task1.py       --config task1/configs/interventions.yaml
```

Outputs: `task1/results/` (intervention metrics, representation stability,
translation curves, cue-conflict manifest) and `figures/task1/` (t-SNE plots).

### Task 2 — Unsupervised Domain Adaptation

PACS, Sketch as the unlabeled target. Train each method, then evaluate once
every checkpoint is frozen.

```bash
for m in source_only dan dann cdan; do
  python task2/train.py --config task2/configs/$m.yaml
done
python task2/evaluate_final.py --config task2/configs/base.yaml

# Controlled study
for lam in 0.1 1.0 10.0; do
  python task2/train.py --config task2/configs/dan_lambda_study.yaml \
      --set method.lambda_mmd=$lam run_name=dan_lambda$lam
done
```

### Task 3 — Domain Generalization

Sketch is unavailable to training, diagnostics and model selection. ERM **reuses
Task 2's Source-only checkpoint** rather than retraining.

```bash
python task3/train.py --config task3/configs/dan_dg.yaml
python task3/train.py --config task3/configs/sam.yaml
python task3/evaluate_sketch.py --config task3/configs/base.yaml   # loads Sketch

for lam in 0.1 1.0 10.0; do
  python task3/train.py --config task3/configs/dan_dg_lambda_study.yaml \
      --set method.lambda_dg=$lam run_name=dan_dg_lambda$lam
done
```

### Task 4 — Open-Set Recognition

CIFAR-10 known, fixed CIFAR-100 classes as near/far unknowns (evaluation only).

```bash
python task4/train.py --config task4/configs/vanilla.yaml
python task4/train.py --config task4/configs/gcsc.yaml
python task4/train.py --config task4/configs/proser.yaml   # inits from vanilla

for m in vanilla gcsc proser; do
  python task4/extract_outputs.py --config task4/configs/$m.yaml
done
python task4/evaluate_osr.py --config task4/configs/vanilla.yaml
```

---

## Information boundaries

These are protocol requirements, enforced in code and covered by tests:

| Task | Rule | Enforcement |
|---|---|---|
| 2 | Target **labels** never influence training or checkpoint selection | Selection uses mean source-validation macro-F1 only |
| 3 | No Sketch image reaches training, diagnostics or selection | `TargetAccessViolation`; `target_batch == 0`; only `evaluate_sketch.py` may load it |
| 3 | ERM is not retrained | `task3/train.py` refuses `erm.yaml` and points at Task 2's checkpoint |
| 4 | CIFAR-100 is evaluation-only | Thresholds calibrate on CIFAR-10 validation; only the CIFAR-100 test split is read |
| 1 | Cue-conflict filtering is model-blind | The rejection rule accepts images only — no model or prediction argument exists |

---

## Reproducibility

- **Seed 6304** for every split, subset, permutation and comparison.
- Each run writes `run.json` with the full config, metrics, git commit
  (`-dirty` when the tree has uncommitted changes), library versions and device.
- Split manifests are committed, so the same seed selects the same images on any
  machine.
- Tasks 2 and 3 share one protocol module; a test asserts they agree on 20
  fields, because Task 2's Source-only checkpoint **is** Task 3's ERM baseline.

Results reported in the write-up come from full runs, never `--smoke`.

---

## Attribution

Implementations are original unless listed here.

**AdaIN style transfer** (`task1/models/adain.py`) — the AdaIN operation, VGG
encoder truncation and decoder architecture are reimplemented from Huang &
Belongie (2017), *Arbitrary Style Transfer in Real-time with Adaptive Instance
Normalization* (ICCV 2017). The **pretrained encoder and decoder weights** are
downloaded from the public PyTorch port
[`naoto0804/pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN)
(release `v0.0.0`, MIT licence) and are not authored here.

**Methods implemented from their papers**: DANN (Ganin et al., 2016), CDAN
(Long et al., 2018), DAN/MMD (Long et al., 2015), SAM (Foret et al., 2021),
PROSER (Zhou et al., 2021), MSP (Hendrycks & Gimpel, 2017), Energy (Liu et al.,
2020), MLS (Vaze et al., 2022).

**Libraries**: PyTorch, torchvision, OpenCLIP, scikit-learn, NumPy, SciPy,
matplotlib, UMAP.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/common.md](docs/common.md) | Shared utilities: seeding, metrics, config, logging |
| [docs/shared.md](docs/shared.md) | PACS protocol, frozen BatchNorm, MMD |
| [docs/task2.md](docs/task2.md) | The four UDA methods and what each tests |
| [docs/adain_weights.md](docs/adain_weights.md) | AdaIN weights and the cue-conflict rejection rule |
| [COLAB.md](COLAB.md) | GPU workflow |
| [progress.md](progress.md) | Living record: state, decisions, bugs, evidence |
