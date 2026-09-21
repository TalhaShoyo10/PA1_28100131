# Task 4 — Open-Set Recognition

Written for: Talha, to understand Task 4 well enough to write and defend the report.

---

## The problem

A closed-set classifier assumes every input belongs to one of its training classes. Show a CIFAR-10 model a picture of a wardrobe and it must answer "truck" or "ship" or one of the other eight — and it may do so **with high confidence**.

Open-set recognition adds the option to say *"none of these."*

**Setup.** CIFAR-10's ten classes are known. Sixteen fixed CIFAR-100 test classes appear **only at evaluation** as unknowns:

| Group | Classes |
|---|---|
| **Near** | bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel |
| **Far** | bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe |

800 images each. The grouping is **fixed and may not be revised after seeing results** — otherwise you'd be tuning the benchmark to flatter your method.

**Why near vs far matters.** These are different problems. A wolf is genuinely dog-like; rejecting it requires fine semantic discrimination. A wardrobe shares almost nothing with any CIFAR-10 class. The PDF: *"Rejecting a wardrobe does not imply that an unseen wolf can be distinguished from dog."* Expect far AUROC to exceed near AUROC — and report them separately, always.

---

## Key terms

- **Closed-set accuracy (CSA)** — ordinary classification on knowns, *before* any rejection
- **Unknownness score u(x)** — larger = more novel (this convention holds throughout)
- **Rejection threshold τ** — converts the score into a known/unknown decision
- **Post-hoc score** — computed on a *fixed* trained model
- **Placeholder learning** — changes the classifier/representation using proxy unknowns built only from known data

---

## The information boundary

**CIFAR-100 is evaluation-only.** No CIFAR-100 image may influence training, checkpoint selection, score definition, or threshold selection. CIFAR-100's *training* partition is never touched at all.

Thresholds calibrate on **CIFAR-10 validation only** — that's the point of the protocol. In deployment you wouldn't have unknowns to tune on, so tuning on them here would inflate every number and make the evaluation meaningless.

---

## Architecture: why the stem is modified

ImageNet ResNet-18 starts with a 7×7 stride-2 convolution and a max-pool. On a 32×32 image that reduces to 8×8 **before a single residual block runs** — most of the image is thrown away.

Replace with 3×3 stride-1 and drop the max-pool. Standard practice for CIFAR.

Training: SGD lr 0.1, momentum 0.9, wd 5e-4, cosine decay, batch 128, 100 epochs, seed 6304. Checkpoint by CIFAR-10 validation accuracy.

---

## The four post-hoc scores

All four run on the **same frozen vanilla model**, reading the **same cached logits and features**. That's essential: any difference between them must come from the score definition, not from evaluation-time variation. One forward pass, cached once, five splits.

### MSP — Maximum Softmax Probability
$$u(x) = 1 - \max_k p_k(x)$$

**Captures:** *normalized* confidence.

**Limitation:** softmax discards absolute magnitude. Logits (1, 0.5, 0.25) and (10, 5, 2.5) give **identical** MSP, despite the second being far more strongly activated. Hendrycks & Gimpel (2017) established this as the baseline.

### MLS — Maximum Logit Score
$$u(x) = -\max_k z_k(x)$$

**Captures:** *absolute* logit magnitude — exactly what MSP throws away.

Vaze et al. (2022) argue this is why MLS tracks closed-set classifier quality: a better classifier produces more strongly-activated logits for genuine members, and that magnitude is itself evidence. Their paper is titled *"a Good Closed-Set Classifier is All You Need?"* — the question mark matters.

### Energy
$$u(x) = -\log\sum_k \exp(z_k(x))$$

**Captures:** *all* logits, not just the maximum. Two inputs with the same max but different spread score differently. Evidence distributed across several classes still counts as evidence.

### Mahalanobis
$$u(x) = \min_c (f(x)-\mu_c)^\top \Sigma^{-1} (f(x)-\mu_c)$$

**Captures:** *geometric* distance in feature space, not logits at all.

Class means and one shared diagonal Σ are estimated from **unaugmented** training features (augmented features would describe the augmentation distribution, not the class distribution). ε=1e-6 on the diagonal prevents division by zero.

**Why it's different in kind:** it can flag an input sitting far from every known cluster *even when the classifier is confident*. Logit-based scores cannot see that.

⚠️ **RQ2 asks where these agree and disagree.** That's the real question — not which wins. MSP and MLS disagreeing tells you magnitude matters. MLS and Mahalanobis disagreeing tells you logit evidence and feature geometry are pointing different ways. Find concrete examples.

---

## The three models

### Vanilla — baseline
Plain cross-entropy. Frozen, then all four scores computed on it.

### GCSC — "a Good Closed-Set Classifier"
**Exactly** the vanilla recipe with **one** change: RandAugment(num_ops=2, magnitude=9), inserted after crop/flip and before tensor conversion. Same init, optimizer, schedule, batch size, epochs, seed, checkpoint rule.

**What it tests.** Vaze et al. suggest closed-set quality and open-set rejection are correlated. So: does better augmentation → better CSA → better rejection?

⚠️ **The PDF warns it may not.** *"Better closed-set accuracy need not improve rejection."* Compare the CSA *change* against the near/far OSR *change* rather than assuming they move together. And check near and far separately — an effect may appear in one and not the other.

A test asserts GCSC and Vanilla differ only by RandAugment, because a second unintended change would break the controlled comparison.

### PROSER — learning placeholders

Initialized from the **selected Vanilla checkpoint** (not from scratch), with five randomly-initialized dummy classifiers appended. Fine-tuned 50 epochs, SGD lr 1e-3.

Two complementary objectives:

**1. Classifier placeholders (β=1).** For a normal CIFAR-10 example, the correct class should stay the largest. But once you *mask out* the correct class, a **dummy** should become the strongest remaining response.

Intuitively: the dummies learn the regions immediately *adjacent* to each known class — the "almost a dog, but not quite" territory — which later provides evidence for rejection.

**2. Data placeholders (γ=0.1).** No real unknowns exist during training, so PROSER manufactures proxies via **manifold mixup**:

$$\tilde{h} = \lambda h_i + (1-\lambda)h_j, \qquad y_i \neq y_j, \qquad \lambda \sim \text{Beta}(2,2)$$

Mixing happens **after layer2, before layer3** — in feature space, not pixel space. The mixed feature is pushed through the rest of the network and trained toward the **dummies**, not toward either original class.

*Why different classes are required:* mixing two dogs gives you a valid dog, not an unknown-like proxy.

*Why Beta(2,2):* it concentrates mass near λ=0.5, so mixtures land **between** the two class regions — which is where a proxy unknown should live. Beta(1,1) would be uniform; Beta(0.2,0.2) would pile up near the endpoints, producing near-copies of real examples.

Each mini-batch splits in half: first half for classifier placeholders, second for data placeholders.

⚠️ **The honest limitation, which RQ4 asks about:** *"Feature interpolations are proxy unknowns, not real unknown classes. They may tighten the boundaries between known classes without representing every direction from which an unknown can arrive."*

Mixup fills the space **between** known classes. A real unknown might arrive from a direction entirely outside the convex hull of your known classes. So PROSER should help most where unknowns genuinely sit between knowns — plausibly *near* unknowns — and less for *far* ones. Check whether your results show that.

**Two evaluation rows for PROSER:**
1. MLS on the **ten known logits only** — directly comparable with Vanilla and GCSC
2. The **placeholder-based score** (strongest dummy vs strongest known) — PROSER's own detector

CSA also uses only the ten known logits, keeping classification and rejection distinct.

---

## The two metrics, and why you need both

### AUROC
Threshold-free. "If I rank every image by unknownness, how well do knowns and unknowns separate?" 100 = perfect, 50 = chance.

Reported three ways: known vs near, known vs far, known vs all.

### Validation-calibrated rejection
τ = the 95th percentile of unknownness on **CIFAR-10 validation**. Accept when u(x) ≤ τ. Aims to accept 95% of knowns.

Report: achieved CIFAR-10 test acceptance, near rejection rate, far rejection rate. **FPR@95TPR** is the fraction of unknowns wrongly accepted — the same quantity as unknown acceptance under this convention.

⚠️ **Why both:** *"AUROC measures ranking over all thresholds; validation-calibrated rejection measures behavior at one operating point."* A score can rank well overall yet behave badly at the operating point you'd actually deploy. AUROC is the research metric; the calibrated point is the deployment metric.

Sanity checks that should hold in your results: known acceptance ≈ 95% for every score (by construction), and rejection + FPR@95TPR = 100 in every row.

---

## Failure analysis

Using the **vanilla MLS threshold**, inspect at least 3 wrongly-accepted near unknowns and 3 far. Record: unknown class, predicted CIFAR-10 class, score, threshold.

**The distinction to draw** — and it's the heart of RQ1:

- **Semantically plausible:** wolf → dog, leopard → cat, pickup_truck → truck. The model isn't malfunctioning; these genuinely resemble known classes. It's arguably doing something reasonable.
- **Surprising:** keyboard → frog, mushroom → ship. These suggest the model is keying on something spurious — colour statistics, background, texture — rather than object identity.

Surprising failures are more informative for the report. They tell you *what evidence the model was actually using*.

Note which CIFAR-10 labels **absorb** the unknowns. If "truck" swallows every vehicle-like unknown, that's a coherent story about a broad decision region.

---

## The four research questions

**RQ1 — How does semantic similarity affect rejection? Which classes are accepted, which labels absorb them, which failures are plausible?**
Near vs far AUROC gap, plus the failure table with the plausible/surprising distinction.

**RQ2 — What does each score capture? Use agreements and disagreements.**
The interesting analysis is *where they diverge*. Find images MSP accepts but Mahalanobis rejects, and explain why in terms of what each measures.

**RQ3 — Does GCSC improve CSA, OSR, both, or neither? Why might near and far differ?**
Four possible outcomes, all reportable. If CSA improves but OSR doesn't, that's direct evidence against the simple "good closed-set classifier is all you need" reading.

**RQ4 — Does PROSER improve rejection beyond Vanilla/GCSC, and at what CSA cost? Where does interpolation remain insufficient?**
Expect a trade-off — the placeholder objectives compete with pure classification. The second clause is the thoughtful part: mixup fills space *between* knowns, so where does that not help?

---

## Required evidence checklist

- [ ] Table: MSP/MLS/Energy/Mahalanobis on the frozen vanilla model — near/far/all AUROC + calibrated rejection
- [ ] Table: Vanilla/GCSC/PROSER with MLS as common score, **plus a PROSER placeholder-score row**, with CSA
- [ ] One compact score-distribution or ROC figure for MSP, MLS, Mahalanobis
- [ ] ≥3 near and ≥3 far failures with class, prediction, score, threshold
- [ ] Analysis of augmentation vs PROSER and the recognition/rejection trade-off

## Traps the PDF explicitly warns about

1. Near and far are different difficulties — never merge them into one number
2. CIFAR-100 is evaluation-only, including for thresholds
3. Confidence, logit magnitude, aggregate logits and feature distance are **different signals**
4. Better CSA need not mean better rejection
5. Feature interpolations are proxies, not real unknowns
6. AUROC and the calibrated operating point answer different questions — report both

## Where the numbers land

| File | Contents |
|---|---|
| `task4/results/osr_results.csv` | Every (model, score, group) row |
| `task4/results/posthoc_score_comparison.csv` | The four scores on vanilla |
| `task4/results/trained_model_comparison.csv` | Vanilla/GCSC/PROSER with MLS + PROSER score |
| `task4/results/failure_cases_vanilla_mls.csv` | Wrongly accepted unknowns |
| `figures/task4/score_distributions.png` | The 3-panel figure |
