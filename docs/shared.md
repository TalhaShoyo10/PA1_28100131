# `shared/` — PACS Protocol and MMD

Written for: Talha, to understand the machinery Tasks 2 and 3 both depend on.

This folder exists because of one hard requirement in the assignment:

> *"The Source-only model from Task 2 is the ERM baseline here; load its saved checkpoint rather than retraining it under a different configuration."*

If Tasks 2 and 3 built their splits, transforms, or batching separately, a subtle divergence would make that shared baseline a **fiction**, and every comparison resting on it — including Task 3 RQ4 — would be invalid. So the shared parts live in exactly one place.

---

## `pacs.py` — Dataset Access

**PACS**: 7 classes (dog, elephant, giraffe, guitar, horse, house, person) × 4 domains (photo, art_painting, cartoon, sketch).

### Why the class order is a fixed tuple

`PACS_CLASSES` is a module-level tuple, so label index 0 is *always* dog, everywhere. If class order came from `os.listdir()`, it could differ between your Windows machine and a Colab Linux VM — and a checkpoint trained with one ordering would produce silently wrong labels under the other.

### Why paths are relative and sorted

`list_domain_images()` returns paths relative to the dataset root, sorted. This makes the **split manifest portable**: the same seed selects the same images on your laptop and on Colab. If paths were absolute, the manifest would be machine-specific and useless.

### `TargetAccessViolation` — the Task 3 guard

```python
if target_access_forbidden and domain == forbidden_domain:
    raise TargetAccessViolation(...)
```

**Concept.** Task 3's defining constraint is that no Sketch image may touch training, diagnostics, or checkpoint selection. That's a rule stated in prose in the PDF — prose can't stop a bug.

**What this does.** It turns a protocol violation from *"produces a plausible-looking but invalid number"* into *"crashes immediately with an explanatory message"*. A loud failure you notice beats a quiet one you submit.

---

## `pacs_protocol.py` — The Shared Protocol

### `stratified_split` — 80/20 within each source domain

**Concept.** Stratified means each class keeps its proportion in both halves. Random splitting could leave a rare class absent from validation entirely, making per-class validation metrics undefined.

**Determinism.** One RNG seeded once, classes visited in `sorted()` order. The result depends only on the seed and the label sequence — never on dict iteration order or filesystem ordering.

**The clamping rule.**
```python
n_val = int(round(n * val_fraction))
if n >= 2:
    n_val = max(1, min(n - 1, n_val))
```
This guarantees a class with ≥2 examples appears on *both* sides. PACS domains are class-imbalanced, and without the clamp a small class could vanish from validation. A class with exactly 1 example goes entirely to train, since you cannot stratify a single example across two splits.

### Transforms

| Split | Pipeline |
|---|---|
| Train | Resize 256 → **Random** 224 crop → horizontal flip → normalize |
| Val/Test | Resize 256 → **Center** 224 crop → normalize |

Evaluation must be deterministic — a random crop at validation time would make the same checkpoint score differently on repeat runs, and checkpoint selection would become noise-driven. `test_eval_transform_is_deterministic` pins this.

Normalization uses ImageNet statistics matching `ResNet18_Weights.IMAGENET1K_V1`, because the pretrained weights expect inputs in that distribution.

### `DomainBalancedIterator` — why it exists

**Requirement.** Every update must contain exactly 8 examples from each source domain (8 × 3 = 24).

**Why.** PACS domains differ substantially in size. If you pooled all three and sampled randomly, the largest domain would dominate every batch, and "training on three domains" would really mean "training mostly on one". The assignment warns: *"Pooling three source domains can hide source-specific behavior."*

**The cycling design.** Epoch length follows the **longest** loader; shorter domains cycle and repeat within an epoch. The alternative — truncating to the shortest domain — would throw away data from the larger ones every epoch.

### `InfiniteLoader` — the target stream

Task 2's adaptation needs 24 target examples on *every* update, regardless of how the target set's length compares to the source epoch. So it cycles endlessly rather than raising `StopIteration` mid-epoch.

---

## The frozen-BatchNorm policy ⚠️

This is the subtlest requirement in Tasks 2 and 3, and worth understanding properly.

### What BatchNorm does

BatchNorm keeps two things:
1. **Learnable affine parameters** γ and β — updated by gradient descent
2. **Running statistics** (mean, variance) — updated by *observing data*, not by gradients

In `train()` mode, every forward pass nudges the running statistics toward the current batch's statistics.

### Why the assignment freezes the statistics

> *"updating BatchNorm statistics on adaptation batches would make the running mean and variance depend on the source–target mixture and introduce an additional implicit form of adaptation."*

Adaptation batches contain both source and target images. If BN statistics updated on them, the model would be adapting **through a second, unspecified channel** — not the MMD or adversarial objective you're actually studying. DAN, DANN and CDAN would each get a free, uncontrolled helping of adaptation, and the comparison between them would no longer isolate their stated mechanisms.

### The implementation

```python
model.train()      # everything in training mode
set_bn_eval(model) # then BN modules alone back to eval
```

Order matters. `model.train()` first, then demote only the BN modules. γ and β **stay trainable** — only the running statistics freeze.

### How this is verified

The test doesn't just check a flag — it checks the **actual guarantee**:

1. Set `running_mean = 0.5`
2. Apply the policy
3. Forward-pass data with wildly different statistics (`randn * 10 + 5`)
4. Assert `running_mean` is still 0.5

Plus a **control test** confirming plain `model.train()` *does* move it — otherwise the first test could pass vacuously. Plus a third confirming γ and β still receive gradients.

`assert_bn_frozen()` is available as a runtime check inside training loops.

---

## `mmd.py` — Maximum Mean Discrepancy

**Deliberately shared between Task 2 and Task 3.** Task 2's DAN aligns source→unlabeled target; Task 3's DAN-DG aligns observed sources to each other. Same mechanism, different information. That's exactly what makes Task 3 RQ4 answerable: if the discrepancy measure also changed, you couldn't attribute any difference to target access.

### The formula

$$\text{MMD}^2 = \mathbb{E}[k(s,s')] - 2\,\mathbb{E}[k(s,t)] + \mathbb{E}[k(t,t')]$$

Intuition: "how similar are sources to each other, plus how similar are targets to each other, minus twice how similar sources are to targets." Zero when the two distributions match.

### The kernel trick

You never build the feature map φ. The kernel evaluates similarities directly — that's why MMD is computable at all for the infinite-dimensional RKHS the theory uses.

### The median heuristic bandwidth

Three RBF kernels at **0.5×, 1×, 2×** the median pairwise squared distance in the current batch.

**Why median, recomputed per batch.** Feature magnitudes drift substantially during fine-tuning. A fixed bandwidth would silently change the penalty's *effective strength* over training — early batches and late batches would be penalised on different scales, and you'd have no way to tell that from the loss curve.

**Why off-diagonal only.** Self-distances are all zero. Including them drags the median down, and on a small batch could pull it to zero entirely.

**The collapse guard.**
```python
if med <= 0:
    return torch.tensor(1.0, ...)
```
If features collapse to identical values, the median is 0 and every kernel would evaluate `exp(0/0)` → NaN, killing the run. The fallback keeps the penalty finite so the collapse shows up **in the loss curve** where you can diagnose it.

**`.detach()` on the bandwidth.** The median is a heuristic for choosing a kernel, not a learnable parameter. Without detaching, gradients would flow into the bandwidth computation and the model could game the penalty by manipulating its own kernel scale.

### Why the biased estimator

The biased version (diagonal included) is guaranteed **non-negative**, which keeps it well-behaved as a loss term. The unbiased version can go negative, which is awkward when you're minimising it. Its O(1/n) bias is constant given the fixed batch size the assignment specifies, so it doesn't distort comparisons across methods.

### Properties verified in tests

| Property | Why it matters |
|---|---|
| ≈ 0 for identical samples | Sanity: no penalty when there's no gap |
| Non-negative | Behaves as a penalty |
| **Grows with separation** | The core property — a wrong implementation could still "run" |
| Symmetric | MMD(a,b) = MMD(b,a) |
| Differentiable w.r.t. both inputs | Gradients must reach both domains |
| Finite on collapsed features | The NaN guard works |

### `pairwise_domain_mmd2` — Task 3's variant

Averages MMD² over all 3 unordered source pairs: (P,A), (P,C), (A,C). Bandwidth is recomputed **per pair**, since two domains' combined feature scale need not match another pair's.

It also returns per-pair values as plain floats for logging, so your training curves can reveal whether one pair dominates the average — useful evidence if DAN-DG behaves oddly.
