# Colab GPU Workflow

Written for: Talha, running the experiments on Colab.

Local machine builds and tests the code; Colab runs the training. The notebook
clones this repository and calls its scripts — it never reimplements anything.

**Flow:** local repo → GitHub → Colab → GPU run → Drive artifacts → local analysis

---

## 0. Before you start

Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**.

Colab VMs are temporary. Everything you want to keep goes to Drive, not
`/content`.

---

## 1. Setup cell

```python
import torch, subprocess
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU"
print(torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/AToML_PA1'
!mkdir -p {DRIVE}/{{checkpoints,results,figures,logs,data,adain_weights}}

!git clone https://github.com/<your-username>/<your-repo>.git /content/pa1
%cd /content/pa1

# torch/torchvision are preinstalled with the right CUDA build -- do not replace them.
!pip install -q open_clip_torch umap-learn pyyaml

!python -m pytest tests/ -q
```

The test suite needs no datasets and should report **322 passed, 1 skipped**
(the skip needs AdaIN weights). If tests fail here, stop and fix before
spending GPU time.

---

## 2. Data

```python
# STL-10 and CIFAR download automatically on first use.
!ln -sfn {DRIVE}/data /content/pa1/data

# PACS must be placed manually -- upload once to Drive, then it persists.
!ls {DRIVE}/data/PACS        # expect: photo art_painting cartoon sketch
```

Generate the shared split manifest once. **Tasks 2 and 3 must use the same
splits**, so never regenerate it between them:

```python
!python shared/make_splits.py --config task2/configs/base.yaml
```

---

## 3. Smoke test before the real run

Always. A protocol or path error surfaces in seconds instead of an hour in.

```python
!python task2/train.py --config task2/configs/source_only.yaml --smoke
!python task4/train.py --config task4/configs/vanilla.yaml --smoke
```

`--smoke` uses 1 epoch and a handful of batches. **Never report a `--smoke`
number.**

---

## 4. Run order

The two dependency chains cannot be parallelised:

```
Task 2 source_only  ──►  Task 3 ERM baseline
Task 4 vanilla      ──►  Task 4 PROSER
```

Start both early.

### Task 2 (~4 GPU-hours incl. the sweep)

```python
DRIVE_ARGS = f"output.checkpoint_dir={DRIVE}/checkpoints/task2 output.results_dir={DRIVE}/results/task2"

for m in ["source_only", "dan", "dann", "cdan"]:
    !python task2/train.py --config task2/configs/{m}.yaml --set {DRIVE_ARGS}

for lam in ["0.1", "1.0", "10.0"]:
    !python task2/train.py --config task2/configs/dan_lambda_study.yaml \
        --set method.lambda_mmd={lam} run_name=dan_lambda{lam} {DRIVE_ARGS}

!python task2/evaluate_final.py --config task2/configs/base.yaml --set {DRIVE_ARGS}
```

### Task 3 (~4 GPU-hours; SAM is ~2x ERM per epoch)

ERM reuses Task 2's checkpoint — `task3/train.py` refuses to retrain it.

```python
D3 = f"output.checkpoint_dir={DRIVE}/checkpoints/task3 output.results_dir={DRIVE}/results/task3"

!python task3/train.py --config task3/configs/dan_dg.yaml --set {D3}
!python task3/train.py --config task3/configs/sam.yaml    --set {D3}

for lam in ["0.1", "1.0", "10.0"]:
    !python task3/train.py --config task3/configs/dan_dg_lambda_study.yaml \
        --set method.lambda_dg={lam} run_name=dan_dg_lambda{lam} {D3}

# Only now does Sketch get loaded, after every Task 3 decision is frozen.
!python task3/evaluate_sketch.py --config task3/configs/base.yaml --set {D3}
```

### Task 4 (~6 GPU-hours — the longest)

```python
D4 = f"output.checkpoint_dir={DRIVE}/checkpoints/task4 output.results_dir={DRIVE}/results/task4 output.cache_dir={DRIVE}/results/task4/cache"

!python task4/train.py --config task4/configs/vanilla.yaml --set {D4}
!python task4/train.py --config task4/configs/gcsc.yaml    --set {D4}
!python task4/train.py --config task4/configs/proser.yaml  --set {D4}   # needs vanilla

for m in ["vanilla", "gcsc", "proser"]:
    !python task4/extract_outputs.py --config task4/configs/{m}.yaml --set {D4}

!python task4/evaluate_osr.py --config task4/configs/vanilla.yaml --set {D4}
```

### Task 1 (~2 GPU-hours, no training dependency)

Can run at any point. Cache AdaIN weights to Drive so they download once:

```python
T1 = f"output.results_dir={DRIVE}/results/task1 output.figures_dir={DRIVE}/figures/task1"

!python task1/data/make_subset.py --config task1/configs/base.yaml
!python task1/data/make_cue_conflicts.py --config task1/configs/interventions.yaml
!python task1/scripts/run_task1.py --config task1/configs/interventions.yaml --set {T1}
```

---

## 5. Sync results back

Checkpoints stay on Drive; only the small result files belong in git.

```python
!cp -r {DRIVE}/results/* /content/pa1/  2>/dev/null
!cp -r {DRIVE}/figures/* /content/pa1/figures/ 2>/dev/null

!cd /content/pa1 && git add -A '*.csv' '*.json' 'figures/**' && \
  git commit -m "Add Task N results" && git push
```

---

## Disconnects and long runs

Colab disconnects on idle. Keep the tab active and check back periodically.

Every training script saves its best checkpoint **every time validation
improves**, so a disconnect loses at most the epochs since the last
improvement — not the whole run. Re-running resumes from scratch, but the saved
checkpoint from a completed run is still usable.

If you hit the free-tier GPU limit mid-way: the tasks are independent apart from
the two chains above, so pick up with whichever task has not run yet.

---

## Compute discipline

Smoke tests may be small. **Final runs must follow the assignment protocol** —
do not silently reduce epochs, data, or the controlled-study grid. If a real
compute constraint forces a change, record it in `progress.md` *before*
changing the protocol, and state it as a limitation in the report.
