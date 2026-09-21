# `common/` — Shared Foundation

Written for: Talha, to understand what each shared utility does and why it was built that way.

These five modules are used by all four tasks. Nothing here is task-specific.

---

## `seed.py` — Reproducibility

**Concept.** The assignment fixes **seed 6304** for every split, subset selection, permutation, and model comparison. Centralising it means no task can silently use a different value.

**What `set_seed()` does.** Seeds four separate RNGs, because Python, NumPy and PyTorch each maintain their own:

| RNG | Used by |
|---|---|
| `random` | Python-level shuffling |
| `np.random` | Split construction, subset selection |
| `torch.manual_seed` | Weight init, dropout, augmentation |
| `torch.cuda.manual_seed_all` | GPU ops across all devices |

**`deterministic=True`** also sets `cudnn.deterministic = True` and disables `cudnn.benchmark`. Benchmark mode picks the fastest convolution algorithm by timing them at runtime, which can vary between runs and produce slightly different numerics. Turning it off costs some speed and buys reproducibility.

**Honest limitation.** This makes runs reproducible on the *same* hardware and library versions. A CPU smoke test and a GPU run will not produce bit-identical results, and neither will two different GPU models. That's expected — it's recorded in run metadata rather than hidden.

**`seed_worker` and `make_generator`.** DataLoader workers are separate processes. Without `worker_init_fn`, each worker reseeds NumPy unpredictably, so multi-worker loading isn't reproducible even with a global seed set. These two functions close that gap.

---

## `metrics.py` — All Measurement

### Closed-set metrics (Tasks 1–4)

**`macro_f1(y_true, y_pred, num_classes)`** — the `num_classes` argument matters more than it looks.

Macro-F1 averages per-class F1 scores. If a class is *absent* from a split, sklearn by default drops it from the average. Consider two PACS validation domains:

- Domain A has all 7 classes, scores perfectly → macro-F1 = 100
- Domain B is missing 1 class, scores perfectly on the other 6 → macro-F1 = 100 *if the absent class is dropped*, but 6/7 = 85.7 if it's counted as 0

Passing `num_classes` forces the second behaviour, so per-domain numbers are comparable. The test `test_macro_f1_counts_absent_class_as_zero` pins this.

**`per_class_accuracy`** returns **NaN** for absent classes, not 0.0. "This class never appeared" and "this class was always wrong" are different facts, and Tasks 2 and 3 both require per-class analysis where confusing them would mislead you.

**`prediction_consistency`** measures whether the *prediction* survived an intervention — **not** whether it was correct. An image the model got wrong before and identically wrong after counts as consistent. This is Task 1's definition:

$$\text{Consistency}(\delta) = \frac{1}{N}\sum_i \mathbb{1}[\hat y(x_i) = \hat y(T_\delta(x_i))]$$

### Task 1 shape-bias metrics

$$\text{Shape Bias} = \frac{N_{\text{shape}}}{N_{\text{shape}} + N_{\text{texture}}} \times 100 \qquad \text{Coverage} = \frac{N_{\text{shape}} + N_{\text{texture}}}{N_{\text{total}}} \times 100$$

**Why `shape_bias` returns NaN when both counts are zero.** If no prediction landed on either intended class, the score is *undefined*. Returning 0.0 would read as "fully texture-biased", which the data doesn't support. The assignment explicitly warns that a high shape-bias value resting on few decisions is weak evidence — hence coverage is always reported alongside.

### Open-set metrics (Task 4)

**Convention: larger `u(x)` = more novel.** Unknowns are the positive class, so a perfect detector scores AUROC = 100. Verified three ways in tests: 100 for perfect separation, 50 for identical distributions, 0 for reversed ranking.

**The threshold protocol.** τ = the 95th percentile of unknownness on the **CIFAR-10 validation set only**. Accept when `u(x) ≤ τ`. This aims to accept 95% of knowns while never touching an unknown example — which is precisely the information boundary the assignment enforces.

**FPR@95TPR is the same quantity as unknown acceptance** under this convention. Both are reported because the assignment asks for both phrasings; `test_fpr_at_95_tpr_equals_unknown_acceptance` confirms they're identical.

---

## `config.py` — Configuration-Driven Execution

**Why config files at all.** Every entry point takes `--config path/to.yaml`, so the exact settings behind a result are a *committed file* rather than a shell command lost to terminal history. The assignment requires preserving the configurations that produced your results.

**Single-level inheritance via `defaults:`.** Each task keeps one `base.yaml`; method configs override only what changes:

```yaml
defaults: base.yaml
run_name: dan
method:
  lambda_mmd: 1.0
```

This matters because Tasks 2 and 3 require initialization, sampling, augmentation, optimizer and budget to stay **identical** across methods. Inheritance makes an accidental divergence visible in the diff instead of buried in a duplicated file.

**`_deep_merge` is recursive.** A child overriding `train: {epochs: 1}` keeps the parent's `train.lr` rather than wiping the whole `train` block. Tested by `test_inheritance_deep_merges`.

### ⚠️ Bug B1 — the `dict` method shadowing trap

This is worth understanding because it was a real, silent bug.

`Config` subclasses `dict` to allow `cfg.lr` as well as `cfg["lr"]`. The obvious implementation is:

```python
def __getattr__(self, name):
    return self[name]
```

**This is broken.** Python only calls `__getattr__` when *normal attribute lookup fails*. `dict` already has methods named `values`, `keys`, `items`, `get`, `update`, `copy`, `pop`. So:

```python
cfg.study.values        # → <built-in method values>, NOT [0.1, 1.0, 10.0]
```

The sweep config for your λ study has a key literally called `values`. A driver looping `for v in cfg.study.values` would have iterated a method object — silently skipping the controlled study, or crashing somewhere far from the cause.

**The fix** overrides `__getattribute__` instead, which Python calls on *every* access, so present keys win over dict methods. Absent keys now raise a descriptive `AttributeError` rather than returning something falsy.

**Lesson worth keeping:** subclassing `dict` for attribute access is a common Python recipe and it has this exact hole. Nine parametrised tests guard it.

---

## `logging.py` — Provenance and Results

**`environment_metadata()`** captures git commit (with a `-dirty` suffix if the tree has uncommitted changes), Python and torch versions, platform, and GPU name. The assignment's submission checklist demands *"every reported number should trace to a saved result file or reproducible command"* — this is how a number traces back.

**`RunRecord`** bundles run name, task, full config, final metrics, and that environment block into one JSON per run.

**`save_csv`** takes the *union* of keys across rows as the header, so a row omitting an optional field (like `closed_set_accuracy` for a score that doesn't have one) writes blank instead of raising.

**`MetricHistory`** accumulates per-epoch rows and dumps a tidy CSV — this becomes the training-curve evidence Tasks 2 and 3 require (classification loss, MMD penalty, domain loss).

---

## `plotting.py` — Figure Defaults

Figures land in an 8-page NeurIPS PDF at small sizes, so defaults favour legibility when shrunk: larger relative fonts, thin grid lines at 30% alpha, no top/right spines.

`METHOD_COLORS` and `BACKBONE_COLORS` fix one colour per method/backbone so DAN is the same orange in every figure across the whole report. `matplotlib.use("Agg")` is set at import so it works headless on Colab identically to locally.

---

## How these fit together

```
config.py     →  what settings this run uses
seed.py       →  make it reproducible
[task code]   →  run the experiment
metrics.py    →  measure the result
logging.py    →  save result + provenance
plotting.py   →  render the figure
```

Every task follows this same spine.
