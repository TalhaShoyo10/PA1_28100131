# `task2/` — Unsupervised Domain Adaptation

Written for: Talha, to understand the four UDA methods and what each one actually tests.

**Setup.** PACS. Sources = Photo, Art Painting, Cartoon (labeled). Target = **Sketch** (images visible, labels forbidden until final evaluation). ResNet-18, 7-class head, full fine-tuning.

**This is a *transductive* UDA protocol**: the target images you finally evaluate on *are* visible during adaptation — just never their class labels.

---

## `models/backbone.py` — `PACSModel`

ResNet-18 with the ImageNet classifier replaced by a 7-class linear head.

### Why `forward(return_features=True)` exists

Every alignment method needs the **512-d pre-classifier feature** and the logits from the *same* forward pass:

| Method | What it needs the feature for |
|---|---|
| DAN | MMD input |
| DANN | Discriminator input |
| CDAN | The `f` half of `vec(f ⊗ p)` |

Recomputing the feature in a second pass would double the cost, and for any model with dropout or a stochastic path the two passes wouldn't even match.

`self.features = nn.Sequential(*list(resnet.children())[:-1])` takes everything up to and including global average pooling, giving a `(B, 512, 1, 1)` tensor that `torch.flatten(..., 1)` turns into `(B, 512)`.

---

## `methods/base.py` — The Shared Interface

**Design principle:** every method runs through the **same** training loop, sampling, optimizer and budget. Only the loss differs.

```python
compute_loss(source_features, source_logits, source_labels,
             target_features=None, target_logits=None,
             progress=0.0) -> MethodOutput
```

**Why this shape.** A method never sees the training loop, so it *cannot* quietly change epochs, learning rate, or batch composition. The assignment requires these held fixed across methods; encoding it in the interface makes violation structurally impossible rather than a matter of discipline.

`MethodOutput` returns the scalar loss plus a dict of plain floats for logging — this becomes the training-curve evidence the assignment requires.

**`classification_loss` is source-only.** Cross-entropy over source examples. Target class labels don't exist during training, so no method may add a target term here.

---

## The four methods

### 1. Source-only ERM — the baseline

Plain cross-entropy on domain-balanced source batches. **No target data at any point.**

**What it tests.** How well ordinary supervised learning transfers without ever seeing the target. This is your reference for every "change relative to Source-only" number.

**⚠️ This checkpoint is also Task 3's ERM baseline.** Hence the explicit guard:

```python
if target_features is not None:
    raise ValueError("SourceOnly received target features...")
```

If this ever received target data, Task 3's entire information boundary would be broken through the back door.

---

### 2. DAN — MMD alignment

$$\mathcal{L}_{\text{DAN}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{MMD}} \cdot \text{MMD}^2(F(x_s), F(x_t))$$

**The idea.** Add a statistical penalty for source and target features having different distributions. Minimise it, and the two become harder to tell apart.

**The critical limitation, and why it's interesting.** The MMD loss **knows nothing about classes**. It only asks that the two *marginal* distributions match. It would be perfectly happy aligning source-dogs with target-elephants, as long as the overall feature clouds overlap.

This is exactly what Task 2 RQ2 probes: *does lower domain separability correspond to better target recognition?* Not necessarily — and the domain-separability diagnostic exists to expose that gap.

**λ_MMD = 1** for the main comparison; your controlled study sweeps {0.1, 1, 10}.

---

### 3. DANN — adversarial alignment

Replaces the explicit statistical distance with a **learned adversary**.

**The architecture.** A discriminator (256 hidden units → ReLU → dropout 0.5 → 2 classes) tries to classify each feature as source or target.

**The gradient-reversal trick.** The GRL is the identity going forward, but multiplies the gradient by **−α** going backward:

```
Forward:   x ──────────────► discriminator
Backward:  x ◄──── ×(−α) ──── discriminator
```

So one single loss drives two opposing objectives:
- The **discriminator** descends its own loss → gets better at telling domains apart
- The **backbone** receives the negated gradient → gets better at *fooling* it

No alternating optimisation, no separate optimiser steps. Elegant.

**The α schedule.**

$$\alpha(p) = \frac{2}{1 + e^{-10p}} - 1, \qquad p \in [0,1]$$

α starts at **0** and ramps toward 1. Why: early in training the representation hasn't learned anything worth aligning yet. Full adversarial pressure from step 0 tends to destroy the features before they encode class information. The ramp lets classification establish itself first.

`test_dann_alpha_is_zero_at_training_start` verifies this concretely: at p=0, the target features receive **zero** gradient.

**⚠️ Interpreting domain accuracy.** The assignment warns explicitly: domain accuracy near 50% can mean *three different things* —
1. successful domain confusion (what you want)
2. an undertrained discriminator (meaningless)
3. collapsed features (bad)

That's why `domain_acc` is logged every step, to be read **alongside** classification loss and source performance rather than alone.

---

### 4. CDAN — class-conditional adversarial alignment

**The problem CDAN addresses.** DAN and DANN align *marginal* distributions. If the domains become similar "in the wrong way" — target-dogs landing where source-horses live — the discriminator is fooled but the classes are scrambled.

**The fix.** Give the discriminator the class prediction too, via the multilinear map:

$$g(x) = \text{vec}(f \otimes p), \qquad f \in \mathbb{R}^{512},\ p \in \mathbb{R}^{7} \ \Rightarrow\ 3584\text{-d}$$

Now the discriminator sees *"this is a feature that looks like a dog"* rather than just *"this is a feature"*. Fooling it requires aligning **semantically corresponding regions**, not just overall clouds.

**⚠️ Two explicit prohibitions, both enforced:**

| Forbidden | Why it matters |
|---|---|
| Entropy conditioning | A CDAN+E variant exists in the paper; the assignment wants plain CDAN |
| Detaching `f` or `p` | Gradients must flow into **both** backbone and classifier head |

Detaching `p` is a *very common shortcut* in public implementations — it stabilises training but cuts the classifier head out of the adversarial game entirely, silently changing the method. Two tests guard this by asserting `target_logits.grad` is non-zero.

**CDAN subclasses DANN** and overrides only `_discriminator_input()`. This guarantees identical hidden width, dropout, schedule and loss weight, so any difference in results is attributable to the conditioning rather than to discriminator capacity.

---

## The controlled study: λ_MMD ∈ {0.1, 1, 10}

**Pre-registered expectations** (stated *before* seeing any target result — this is the scientific discipline the assignment demands):

| Quantity | Expectation | Reasoning |
|---|---|---|
| Source validation | **Degrades** as λ grows | The penalty competes with cross-entropy for capacity; at λ=10 it should dominate the gradient |
| Domain separability | **Falls** toward 50% | That's literally what the penalty optimises |
| Target recognition | **Non-monotone**, peaking near λ=1 | Too little → gap remains. Too much → marginal matching mixes classes (Zhao et al. 2019) |

**Hard constraint.** These target results are **analysis only**. The main comparison stays at λ=1. You may not pick a post-hoc winner from target performance — that would be target leakage.

This grid is deliberately **matched** to Task 3's λ_DG sweep, so RQ4 can compare target-aware vs target-free alignment at equal pressure.

---

## The domain-separability diagnostic

After all checkpoints are frozen: freeze the backbone, collect equal numbers of source-validation and target features, 70/30 split with seed 6304, train a balanced logistic regression (C=1) to distinguish source from target. Its held-out accuracy is the score. **50% = chance.**

**⚠️ How to read it.** The assignment is blunt:

> *"A lower domain-separability score is evidence that domain information is harder to recover, not that class information has been preserved."*

Lower separability is **not** automatically better. The whole point of RQ2 is to test whether it tracks target recognition — and to find the cases where it doesn't.

---

## Information boundary — the rules for this task

| Permitted | Forbidden |
|---|---|
| Target **images** during adaptation | Target **class labels** during training |
| Target **domain identity** (source vs target) | Target labels for checkpoint selection |
| Target labels at final evaluation, after everything is frozen | Target labels influencing hyperparameters or method design |

Checkpoints are selected by **mean macro-F1 across the three source validation splits** — never by target performance.
