# Task 2 — Unsupervised Domain Adaptation

Written for: Talha, to understand Task 2 well enough to write and defend the report.

*(For code-level detail see [task2.md](task2.md). This document is about the science.)*

---

## The setup and the central tension

**PACS**: 7 classes (dog, elephant, giraffe, guitar, horse, house, person) across 4 domains. Photo, Art Painting and Cartoon are **labeled sources**. **Sketch is the target** — its images are visible during adaptation, its labels are not.

**The premise.** A model trained only on the sources will do badly on Sketch, because sketches look nothing like photos. But you *have* sketch images — just not their labels. Can unlabeled structure help?

**The tension that runs through the whole task.** Every method here makes source and target features harder to tell apart. But:

> Making domains indistinguishable is **not the same** as making classes recognisable.

An alignment loss that only matches marginal distributions would be perfectly satisfied by mapping source-dogs onto target-elephants, as long as the overall feature clouds overlap. Zhao et al. (2019) formalise this. RQ2 is precisely about finding where it happens.

---

## The protocol, and why each constraint exists

| Constraint | Reason |
|---|---|
| ResNet-18, full fine-tuning, 7-class head | Fixed so architecture isn't a variable |
| Stratified 80/20 source splits, seed 6304 | Same splits for every method **and for Task 3** |
| 8 examples per source domain per batch | Prevents the largest domain dominating |
| 24 target examples per update | Equal total source and target |
| AdamW, lr 1e-4, wd 1e-4, ≤30 epochs, patience 5 | Identical budget across methods |
| Checkpoint by **mean source-validation macro-F1** | Target labels must not influence selection |
| **Frozen BatchNorm running statistics** | See below — this one is subtle |

### The frozen-BatchNorm policy

BatchNorm keeps two separate things: learnable affine parameters (γ, β) updated by gradients, and running mean/variance updated by *observing data*.

Adaptation batches contain both source and target images. If BN statistics updated on them, the model would adapt through a **second, unspecified channel** — the running statistics would drift toward the source-target mixture. Every method would get a free, uncontrolled dose of adaptation, and you could no longer attribute differences to MMD vs adversarial vs conditional alignment.

So running statistics stay frozen at pretrained ImageNet values; γ and β stay trainable. This is verified behaviourally in the tests: statistics don't move under a distribution-shifted forward pass, while γ and β still receive gradients.

### Transductive UDA

The target images you finally evaluate on **are** the ones visible during adaptation. That's allowed and stated in the PDF. What's forbidden is target *labels* influencing training, checkpoint selection, hyperparameters, or method design.

---

## The four methods

### 1. Source-only ERM — the baseline

Plain cross-entropy on domain-balanced source batches. No target data at all.

**What it tests.** How far ordinary supervised learning gets you. Every "improvement" later is measured against this.

⚠️ **This exact checkpoint is also Task 3's ERM baseline.** That's why the code refuses to let it see target data — a leak here would break Task 3's information boundary through the back door.

---

### 2. DAN — statistical alignment

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_{\text{MMD}} \cdot \text{MMD}^2(F(x_s), F(x_t))$$

**Concept.** Maximum Mean Discrepancy measures how different two distributions are, using kernel similarities:

$$\text{MMD}^2 = \mathbb{E}[k(s,s')] - 2\,\mathbb{E}[k(s,t)] + \mathbb{E}[k(t,t')]$$

Read it as: "how similar sources are to each other, plus how similar targets are to each other, minus twice how similar sources are to targets." Zero when the distributions match.

**The kernel trick** means you never construct the feature map φ — kernels evaluate similarities directly, which is what makes MMD computable for the infinite-dimensional space the theory uses.

**Bandwidth via median heuristic**, recomputed *per batch*, at 0.5×/1×/2×. Why per batch: feature magnitudes drift a lot during fine-tuning, and a fixed bandwidth would silently change the penalty's effective strength over training with no visible sign in the loss curve.

**What it tests.** Whether reducing a *statistical* discrepancy helps, given that the loss has **no idea which target example is which class**. This is marginal alignment in its purest form.

---

### 3. DANN — adversarial alignment

**Concept.** Replace the fixed statistical measure with a *learned adversary*. A discriminator tries to classify each feature as source or target. The backbone tries to fool it.

**The gradient-reversal trick.** The GRL is the identity going forward but multiplies the gradient by −α going backward:

```
forward:   features ──────────────► discriminator
backward:  features ◄──── ×(−α) ──── discriminator
```

One loss, two opposing objectives, no alternating optimisation. The discriminator descends its loss; the backbone receives the negated gradient and ascends it.

**The schedule** $\alpha(p) = \frac{2}{1+e^{-10p}} - 1$ ramps from 0 to ~1.

*Why ramp.* Early in training the representation hasn't learned anything worth aligning. Full adversarial pressure from step 0 tends to destroy features before they encode class information. Verified: at p=0, α=0 and the target features receive exactly zero gradient.

⚠️ **Domain accuracy near 50% is ambiguous — it has three possible causes:**
1. successful domain confusion (what you want)
2. an undertrained discriminator (meaningless)
3. collapsed features (bad)

You cannot distinguish them from that number alone. Read it alongside classification loss and source performance — which is why `domain_acc` is logged every step. If domain accuracy is at chance *and* source accuracy collapsed, suspect collapse, not success.

---

### 4. CDAN — class-conditional alignment

**The problem it addresses.** DAN and DANN align *marginal* distributions. Nothing stops target-dogs landing where source-horses live — the discriminator is fooled, the classes are scrambled.

**The fix.** Give the discriminator the prediction too:

$$g(x) = \text{vec}(f \otimes p), \qquad f \in \mathbb{R}^{512},\ p \in \mathbb{R}^{7} \Rightarrow 3584\text{-d}$$

Now the discriminator sees *"a feature that looks like a dog"* rather than just *"a feature"*. Fooling it requires aligning **semantically corresponding regions**, not just overall clouds.

⚠️ **Two prohibitions in the PDF, both enforced in code:**
- No entropy conditioning (a CDAN+E variant exists; not wanted here).
- **No detaching `f` or `p`.** Detaching `p` is a very common shortcut in public implementations — it stabilises training but cuts the classifier head out of the adversarial game entirely, silently changing the method. Tests assert gradients reach both.

CDAN shares DANN's hidden width, dropout, schedule and loss weight, so any difference is attributable to **conditioning**, not capacity.

---

## The domain separability diagnostic

After every checkpoint is frozen: freeze the backbone, collect equal numbers of source-validation and target features, 70/30 split (seed 6304), train a balanced logistic regression (C=1) to predict source-vs-target. Held-out accuracy is the score. **50% = chance.**

⚠️ **How to read it — the PDF is blunt:**

> *"A lower domain-separability score is evidence that domain information is harder to recover, not that class information has been preserved."*

Lower is **not** automatically better. Four cases:

| Separability | Target accuracy | Reading |
|---|---|---|
| High | Low | Alignment didn't work |
| **Low** | **High** | Alignment worked as hoped |
| **Low** | **Low** | ⚠️ Domains merged but classes got mixed — or features collapsed |
| High | High | Transfer happened without needing invariance |

Row 3 is the interesting failure and exactly what RQ2 hunts for.

---

## Per-class analysis — why aggregates lie

Target labels are used **only here**, at the very end.

A method can improve mean target accuracy while *destroying* a specific class. Aggregate gains hide class-specific negative transfer. Concretely: if "guitar" gains 20 points and "person" loses 15, the mean looks fine and the model is now unusable for people.

`per_class_delta_<method>.csv` is sorted worst-first for exactly this reason. Look at the top rows and at the confusion pairs: *what* is the degraded class being mistaken for? A plausible confusion (horse↔dog) tells a different story from an implausible one (house↔person).

---

## The controlled study: λ_MMD ∈ {0.1, 1, 10}

**Pre-registered expectations** — stated before seeing results, which is the scientific point:

| Quantity | Expected | Reasoning |
|---|---|---|
| Source validation | **Degrades** as λ grows | The penalty competes with cross-entropy for capacity |
| Domain separability | **Falls** toward 50% | That's literally what's being optimised |
| Target accuracy | **Non-monotone**, peak near λ=1 | Too little → gap remains; too much → marginal matching mixes classes |

⚠️ **These results are analysis-only.** The main comparison stays at λ=1. Choosing a λ because it scored best on target would be **target leakage** — you'd be using target labels for model selection, which the PDF forbids.

**RQ4 asks what setting could have been chosen *without* target labels.** That's a genuine question: you'd have to argue from source performance and separability alone. Be honest if the source-side signal wouldn't have picked the best λ — that's a real and reportable finding about DG/UDA model selection.

This grid is deliberately **matched** to Task 3's λ_DG sweep, so the cross-task comparison happens at equal alignment pressure.

---

## The four research questions

**RQ1 — How large is the gap, and which classes fail?**
Source-only's source-vs-target numbers, plus per-class target accuracy and confusions. Which classes does Sketch break, and is there a pattern? (Sketches lack texture and colour entirely — does that predict which classes suffer?)

**RQ2 — Does lower separability mean better target recognition?**
The core question. Use the four-case table. Aggregate *and* class-level evidence. Look hard for the "low separability, low accuracy" case.

**RQ3 — Does class-conditional alignment beat marginal?**
CDAN vs DAN/DANN. Does conditioning preserve semantic structure? Look at per-class results, not just the mean — CDAN's claim is specifically about *which* regions get aligned.

**RQ4 — How does alignment strength trade off, and what could you have chosen blind?**
The λ sweep plus the honest model-selection question.

---

## Required evidence checklist

- [ ] Table: all 4 methods × each source validation domain, mean source acc/F1, target acc/F1, target delta, domain separability
- [ ] Classification and alignment/domain loss curves showing each method trained as intended
- [ ] Per-class target changes plus selected confusions supporting a positive/negative transfer claim
- [ ] Compact table or plot for the λ study

## Traps the PDF explicitly warns about

1. Low separability ≠ preserved class information
2. Domain accuracy at chance has three possible causes
3. Pooling three sources hides source-specific behaviour — **report each domain before the mean**
4. Source down + target up is a meaningful trade-off; **both down means investigate** excessive alignment or unstable optimisation
5. Never inspect target metrics while choosing checkpoints or settings

## Where the numbers land

| File | Contents |
|---|---|
| `task2/results/final_comparison.csv` | The main table |
| `task2/results/<method>/training_curve.csv` | Per-epoch losses, per-domain validation |
| `task2/results/per_class_delta_<method>.csv` | Per-class change vs Source-only, worst first |
| `task2/results/target_confusions_<method>.csv` | Top confusion pairs |
| `task2/results/target_detail_<method>.json` | Per-class accuracy, separability |
