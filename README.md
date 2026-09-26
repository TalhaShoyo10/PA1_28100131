# Beyond IID — ATML PA1

Four studies on what happens when the IID, closed-set assumption breaks.

| Task | Setting | Dataset | Methods |
|---|---|---|---|
| 1 | Inductive biases | STL-10 | Frozen ResNet-50, ViT-B/16, CLIP ViT-B/32 |
| 2 | Unsupervised domain adaptation | PACS (Sketch target) | Source-only, DAN, DANN, CDAN |
| 3 | Domain generalization | PACS (Sketch unseen) | ERM, DAN-DG, SAM |
| 4 | Open-set recognition | CIFAR-10 / CIFAR-100 | Vanilla, GCSC, PROSER |

Every run is configuration-driven, seeded with **6304**, and writes results
that trace back to the command and commit that produced them.

---

## Setup

```bash
git clone https://github.com/TalhaShoyo10/PA1_28100131.git
cd PA1_28100131
pip install -r requirements.txt
python -m pytest tests/ -q          # 379 tests, no datasets needed
```

---

## Datasets

STL-10, CIFAR-10 and CIFAR-100 download automatically on first use.

**PACS must be downloaded manually** (no torchvision loader). Arrange it as:

```
data/PACS/
  photo/dog/*.jpg
  art_painting/dog/*.jpg
  cartoon/dog/*.jpg
  sketch/dog/*.png
  ... 7 classes x 4 domains
```

Then build the split manifest that Tasks 2 and 3 share:

```bash
python shared/make_splits.py --config task2/configs/base.yaml
```

Task 1's cue conflicts need pretrained AdaIN weights, which download
automatically on first use.

---

## Reproducing each task

Every script takes `--config` and accepts `--set key=value` to override any
config entry. Add `--smoke` for a fast sanity check; no reported number comes
from a smoke run.

**Task 1**

```bash
python task1/data/make_subset.py        --config task1/configs/base.yaml
python task1/data/make_cue_conflicts.py --config task1/configs/interventions.yaml
python task1/scripts/run_task1.py       --config task1/configs/interventions.yaml
```

**Task 2**

```bash
for m in source_only dan dann cdan; do
  python task2/train.py --config task2/configs/$m.yaml
done

for lam in 0.1 1.0 10.0; do
  python task2/train.py --config task2/configs/dan_lambda_study.yaml \
      --set method.lambda_mmd=$lam run_name=dan_lambda$lam
done

python task2/evaluate_final.py --config task2/configs/dan.yaml \
    --runs source_only dan dann cdan dan_lambda0.1 dan_lambda1.0 dan_lambda10.0
```

Pass all seven run names. `final_comparison.csv` is rebuilt on each call, and
the target-accuracy change is computed only when `source_only` is included.

**Task 3**

```bash
python task3/train.py --config task3/configs/dan_dg.yaml
python task3/train.py --config task3/configs/sam.yaml

for lam in 0.1 1.0 10.0; do
  python task3/train.py --config task3/configs/dan_dg_lambda_study.yaml \
      --set method.lambda_dg=$lam run_name=dan_dg_lambda$lam
done

python task3/evaluate_sketch.py --config task3/configs/base.yaml
```

ERM is not retrained. It reuses Task 2's Source-only checkpoint, so run Task 2
first. `evaluate_sketch.py` is the only script permitted to load Sketch.

**Task 4**

```bash
python task4/train.py --config task4/configs/vanilla.yaml
python task4/train.py --config task4/configs/gcsc.yaml
python task4/train.py --config task4/configs/proser.yaml   # starts from vanilla

for m in vanilla gcsc proser; do
  python task4/extract_outputs.py --config task4/configs/$m.yaml
done

python task4/evaluate_osr.py --config task4/configs/vanilla.yaml
```

**Figures**

```bash
python scripts/make_figures.py
```

Builds the translation curve and the training-curve figures from the committed
CSVs. No GPU needed.

---

## Layout

```
common/      seed, metrics, logging, config loader, plotting
shared/      PACS dataset and protocol, multi-kernel MMD  (Tasks 2 and 3)
scripts/     make_figures.py
task1/       interventions, backbones, representation analysis
task2/       Source-only, DAN, DANN, CDAN
task3/       ERM (reused), DAN-DG, SAM
task4/       Vanilla, GCSC, PROSER, post-hoc scores
tests/       379 tests covering protocol compliance and method behaviour
figures/     generated figures
```

Results live under each task's `results/` directory. Datasets and checkpoints
are not committed.

---

## Reproducibility

- Seed 6304 for every split, subset and permutation.
- Split manifests are committed, so the same images are selected on any
  machine: `shared/splits/`, `task1/data/eval_subset_seed6304.json`,
  `task4/data/cifar10_split_seed6304.json`.
- Every run writes `run.json` with its resolved config, git commit, library
  versions and device.
- Tasks 2 and 3 share one protocol module, so their results are directly
  comparable. Task 2's Source-only checkpoint is Task 3's ERM baseline.

### Information boundaries

Enforced in code and covered by tests:

| Task | Rule |
|---|---|
| 2 | Checkpoints selected on source-validation macro-F1 only |
| 3 | Training fails unless target access is disabled; only `evaluate_sketch.py` loads Sketch |
| 3 | ERM is loaded, never retrained |
| 4 | Thresholds calibrate on CIFAR-10 validation; CIFAR-100 is read only at evaluation |
| 1 | Cue-conflict filtering uses image statistics, with no model consulted |

---

## Attribution

Implementations are original except where noted.

**AdaIN style transfer** (`task1/models/adain.py`) reimplements Huang &
Belongie (2017). The pretrained encoder and decoder weights come from
[naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
(release `v0.0.0`, MIT licence) and are not authored here.

**Methods implemented from their papers:** DAN (Long et al., 2015), DANN
(Ganin et al., 2016), CDAN (Long et al., 2018), SAM (Foret et al., 2021),
PROSER (Zhou et al., 2021), MSP (Hendrycks & Gimpel, 2017), Energy (Liu et
al., 2020), MLS (Vaze et al., 2022), RandAugment (Cubuk et al., 2020).

**Libraries:** PyTorch, torchvision, OpenCLIP, scikit-learn, NumPy, SciPy,
matplotlib, UMAP.
