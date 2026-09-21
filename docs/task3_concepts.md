# Task 3 — Domain Generalization

Written for: Talha, to understand Task 3 well enough to write and defend the report.

---

## What changes from Task 2

Same dataset, same backbone, same splits, same optimizer, same budget, same seed. **One thing changes: you no longer get to see Sketch.**

Not its labels — not the *images* either. Nothing in training, diagnostics, checkpoint selection or hyperparameter choice may touch Sketch.

That single change is the whole task. Task 2 asked *"does unlabeled target data help?"* Task 3 asks *"can you generalise to a domain you have never encountered?"*

---

## The information boundary — the defining constraint

Three rules, and they're stricter than they first appear:

1. **No Sketch image** reaches Task 3 training, source-side diagnostics, checkpoint selection, or hyperparameter selection.
2. **Task 2's Sketch results may not inform Task 3 settings.** You already saw Sketch in Task 2. If you used that knowledge to pick DAN-DG's λ or SAM's ρ, you'd be leaking target information through *yourself*. The PDF is explicit.
3. **ERM is not retrained.** It *is* Task 2's Source-only checkpoint, loaded unchanged.

Rule 3 is what makes the cross-task comparison meaningful. If Task 3's ERM were retrained, any DAN-vs-DAN-DG difference could come from two different baselines rather than from target access.

**Enforcement in code:** `PACSDataset` raises `TargetAccessViolation` if asked for Sketch under the forbidden flag; `train.py` refuses `erm.yaml` and points at Task 2's checkpoint; `target_batch` is 0; only `evaluate_sketch.py` may load Sketch. Tests assert all of it.

Why enforce in code at all? Because a rule stated in prose can't stop a bug. A loud crash beats a plausible-looking invalid number in your report.

---

## The three hypotheses

Each method embodies a different theory of what makes a model generalise.

| Method | Theory | Mechanism |
|---|---|---|
| **ERM** | Diverse labeled sources are enough | Nothing special — just train |
| **DAN-DG** | Domain-invariant features transfer | Remove what distinguishes Photo/Art/Cartoon |
| **SAM** | Flat minima transfer | Find parameters whose loss is locally stable |

These are genuinely different claims. DAN-DG operates in *feature space*; SAM operates in *parameter space* and doesn't mention domains at all.

---

### ERM — the baseline that matters

$$\mathcal{L}_{\text{ERM}} = \frac{1}{3}\sum_{e \in \{P,A,C\}} R_e(\theta)$$

Domain-balanced batches give each environment equal influence.

**Why this is a strong baseline.** Gulrajani & Lopez-Paz (2021), *In Search of Lost Domain Generalization*, found that properly-tuned ERM matches or beats most published DG methods. If your fancy method doesn't beat ERM, that's a real and publishable-style finding, not a failure on your part.

**Report:** per-source-domain accuracy and macro-F1, plus **mean** and **worst**.

---

### DAN-DG — pairwise source alignment

$$\mathcal{L} = \mathcal{L}_{\text{ERM}} + \frac{\lambda_{\text{DG}}}{3}\sum_{e<e'} \text{MMD}^2\big(F(X_e), F(X_{e'})\big)$$

**The adaptation.** Standard DAN aligns sources to an unlabeled *target*. DAN-DG never sees Sketch — it applies the *same* MMD mechanism to the three unordered source pairs: (Photo, Art), (Photo, Cartoon), (Art, Cartoon).

**Why the same MMD code as Task 2.** Deliberate, and it's what makes RQ4 answerable. If the discrepancy measure also changed, you couldn't attribute any Task 2 / Task 3 difference to target access.

**The hope.** If a representation can't distinguish Photo from Art from Cartoon, maybe it has learned something about *objects* rather than *rendering style* — and that might transfer to Sketch.

**The risk.** Aligning Photo/Art/Cartoon to each other does **nothing to constrain the direction in which Sketch differs**. Sketch is more extreme than any source: no colour, no texture, sparse lines. Invariance to the variation you *observed* need not give invariance to variation you *didn't*.

Bandwidths are recomputed **per pair**, since two domains' combined feature scale needn't match another pair's. Per-pair MMD values are logged, so you can see whether one pair dominates the average.

---

### SAM — parameter-space stability

$$\min_\theta \max_{\|\epsilon\|_2 \le \rho} \mathcal{L}_{\text{ERM}}(\theta + \epsilon)$$

**Concept.** Don't just find a low-loss point — find a point where the *whole neighbourhood* is low-loss. A sharp minimum sits in a narrow valley; a small parameter change sends loss soaring. A flat minimum tolerates perturbation.

**The intuition for DG:** a distribution shift acts somewhat like a perturbation. A solution that tolerates parameter perturbation might tolerate input-distribution change. That's the hypothesis — it is not a theorem.

**The two-step update, per batch:**
1. Forward/backward at θ → gradient g
2. Ascend: $\epsilon = \rho \cdot g/\|g\|_2$ — move to the **worst** nearby point
3. Forward/backward at θ+ε
4. Restore θ, apply the optimizer step using the *perturbed-point* gradient

Two forward/backward passes, so roughly **2× ERM wall-clock**. Budget for it.

Verified in tests: perturbation norm is exactly ρ=0.05 in global L2, follows the ascent direction, loss genuinely rises at θ+ε, and θ is restored before the update.

⚠️ **Frozen BN applies to both passes.** If BN statistics updated on the first pass, they'd shift before the second, making the perturbed loss incomparable to the clean one.

**SAM does not remove domain information.** It's a completely different route to generalization — worth saying explicitly in the report.

---

## The two diagnostics

### Source-domain separability (3-way, chance 33.3%)

Freeze the backbone, collect balanced features from the three source validation sets, 70/30 split (seed 6304), multinomial logistic regression (C=1) predicting Photo/Art/Cartoon.

⚠️ Note this differs from Task 2's **binary** source-vs-target separability (chance 50%). Don't compare the two numbers directly — different tasks, different chance levels.

**Reading it:** lower = stronger invariance across *observed* sources. But the PDF warns: *"Source-domain invariance is not the same as class invariance. DAN-DG can reduce domain separability while also removing information needed to distinguish classes."*

So the critical check: if DAN-DG drives separability toward 33% **and** source accuracy falls **and** Sketch doesn't improve — invariance destroyed class information. That's a finding, and RQ2 asks for exactly this evidence.

### The sharpness proxy

$$\Delta_{\text{sharp}} = \mathcal{L}_{\text{val}}(\theta + \epsilon) - \mathcal{L}_{\text{val}}(\theta), \qquad \epsilon = 0.05\,\frac{\nabla_\theta \mathcal{L}_{\text{val}}}{\|\nabla_\theta \mathcal{L}_{\text{val}}\|_2}$$

One fixed validation batch (32 per source, seed 6304), eval mode, identical for all three models.

⚠️ **What this does and doesn't establish.** The PDF: *"This is a standardized local diagnostic, not a proof that one model is globally flatter."*

You are measuring loss increase after **one** perturbation, in **one** direction (the gradient's), at **one** radius, on **one** batch. It supports claims about *local stability under this specified perturbation*. It does **not** establish that the whole loss landscape is flatter. Flatness is notoriously measurement-dependent. Phrase your claims accordingly.

The diagnostic is verified non-destructive: the model is bit-identical afterwards with no stray gradients — important, since a diagnostic that perturbed the model would corrupt everything measured after it.

---

## Mean vs worst source

Report both. **Mean-source** can hide a weak domain: 95/95/60 averages to 83, which looks fine while one environment is being neglected. **Worst-source** exposes it.

But the PDF cautions that neither reliably predicts Sketch: *"strong worst-source performance need not predict Sketch. Treat both as source-side diagnostics."*

RQ1 asks exactly how well they predict. If they predict poorly, say so — that's the honest and interesting answer, and it speaks to why DG model selection is hard.

---

## The four research questions

**RQ1 — How well do mean/worst source validation predict Sketch? Which class failures are invisible from aggregates?**
Compare source rankings to Sketch rankings. Then look at per-class Sketch results for failures the source numbers never hinted at.

**RQ2 — Does DAN-DG reduce source separability, and does that help Sketch? Any evidence it removed class information?**
Two separate questions. It may well succeed at the first and fail at the second. Evidence for "removed class info": separability down, source accuracy down, Sketch unimproved.

**RQ3 — Does SAM reduce the sharpness proxy? Does that ranking match Sketch performance?**
Two rankings: by Δ_sharp and by Sketch accuracy. Agreement supports the flatness hypothesis; disagreement is equally interesting.

**RQ4 — Target-aware DAN (Task 2) vs target-free DAN-DG (Task 3): what is unlabeled Sketch worth? What prevents causal attribution?**

The synthesis question, and the reason your λ grids are matched.

The comparison: both use identical MMD, identical kernels, identical grids, identical ERM baseline. The difference is *what they align to*.

**What prevents a clean causal claim** — you must state these limitations:
- DAN aligns source↔target; DAN-DG aligns source↔source. Different *pairings*, not merely target presence.
- DAN gets 3 pairs' worth of alignment signal from a genuinely different distribution; DAN-DG gets 3 pairs among similar ones.
- One seed, one architecture, one target domain. No error bars.
- Sketch is unusually distant from all three sources — conclusions may not transfer to a milder shift.

---

## Required evidence checklist

- [ ] Table: ERM/DAN-DG/SAM × each source domain, mean, worst, Sketch acc/F1, Sketch delta vs ERM
- [ ] Source-domain separability and sharpness proxy for all three
- [ ] Training curves including the MMD penalty where applicable
- [ ] Compact table or plot for the λ_DG study
- [ ] Per-class Sketch changes and failures, **including comparison with Task 2**

## Traps the PDF explicitly warns about

1. Source invariance ≠ class invariance
2. Strong mean hides a weak domain; strong worst doesn't predict Sketch
3. Lower sharpness proxy ≠ globally flatter landscape
4. DAN and DAN-DG share a mechanism but differ in information — don't conflate them
5. **Don't inspect Sketch to explain a training choice.** Target examples may be examined only in the final failure analysis.

## Where the numbers land

| File | Contents |
|---|---|
| `task3/results/final_comparison.csv` | Main table incl. separability and Δ_sharp |
| `task3/results/<method>/training_curve.csv` | Per-epoch losses, per-pair MMD |
| `task3/results/per_class_sketch_delta_<method>.csv` | Per-class change vs ERM |
| `task3/results/sketch_detail_<method>.json` | Per-class accuracy, separability, sharpness |
