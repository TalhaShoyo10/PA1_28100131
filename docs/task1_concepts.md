# Task 1 — Inductive Biases and Representations

Written for: Talha, to understand Task 1 well enough to write and defend the report.

Structure per experiment: **Concept → Why this design → What is measured → How to read the result → Traps.**

---

## The central question

Two models can have identical clean accuracy and still be using *completely different evidence* to get there. One might be reading shape, another texture, another colour. On ordinary images you cannot tell, because all the cues agree — a dog is dog-shaped *and* has dog fur *and* appears in dog-ish contexts.

**Controlled interventions break that agreement.** You change one factor, hold the rest fixed, and see whose predictions move.

A second, subtler question sits underneath: when a prediction *doesn't* change, does that mean the model didn't notice? Not necessarily. The representation may have shifted a lot while the classifier's argmax happened to survive. That is why Task 1 measures **both** prediction stability and representation stability.

---

## The three models, and what is confounded

| Model | Architecture | Pretraining | Supervision |
|---|---|---|---|
| ResNet-50 | Convolutional | ImageNet-1k | Labels |
| ViT-B/16 | Transformer, learned positional encoding | ImageNet-1k | Labels |
| CLIP ViT-B/32 | Transformer | ~400M web image-text pairs | Natural language |

⚠️ **This is the single most important caveat in Task 1, and RQ4 is entirely about it.**

These three differ on *architecture*, *pretraining data scale*, *supervision type*, *augmentation*, and *capacity — all at once*. So if CLIP behaves differently from ResNet, you **cannot** say "because it's a transformer." ViT is also a transformer and shares ResNet's pretraining.

The one comparison that *does* isolate something:

> **ResNet-50 vs ViT-B/16** share pretraining data, supervision type and roughly the era of training recipe. A difference between them is *more* attributable to architecture — though augmentation recipes still differ, so even this isn't clean.

> **CLIP head vs CLIP zero-shot** uses the *same frozen features*. Any difference is purely about the classifier on top, not the representation. This is a genuinely controlled comparison and RQ3 asks for it directly.

Write the report in those terms: what is isolated, what is confounded.

---

## Why backbones are frozen

Only a linear head is trained. Two reasons:

1. **You are measuring the pretrained representation**, not what fine-tuning could do to it. Fine-tuning would let each model reshape its features and you'd be comparing three training runs, not three inductive biases.
2. **The linear probe is a readability test.** If a linear classifier can extract class identity from the features, the information is present and linearly accessible.

All three heads use the same recipe (AdamW, lr 1e-3, wd 1e-4, ≤50 epochs, early stop at 5). That is deliberate: a difference in head quality must not be confused with a difference in representation quality.

---

## Experiment 1 — Clean baseline

**Concept.** Establish the reference. Everything else is measured *relative to this*.

**Measured.** Top-1 accuracy, macro-F1, mean maximum confidence.

**How to read it.** The PDF says to compare interventions using *both* absolute performance *and* change relative to each model's own clean baseline. This matters: if ResNet starts at 95% and CLIP at 85%, a 10-point drop means different things. Always report both `accuracy` and `accuracy_delta`.

**Why mean max confidence.** It is your calibration signal. A model that stays confident while accuracy collapses is more alarming than one that becomes appropriately uncertain.

---

## Experiment 2 — Colour

**Concept.** Does chromatic information drive the decision?

Two interventions, deliberately complementary:

| | Grayscale | Hue rotation 90° |
|---|---|---|
| Colour is | **removed** | **changed** |
| Geometry | preserved | preserved |
| Luminance | preserved | see note below |
| Saturation | destroyed | preserved |

**Why hue rotation is the chosen second intervention.** The PDF offered palette transfer or class-swapped colour statistics as alternatives. Hue rotation is the cleanest counterpart because it provably preserves geometry, edges and saturation — it changes *only* the hue angle. Palette transfer also perturbs contrast and global statistics, which would confound "sensitive to colour" with "sensitive to overall appearance."

⚠️ **A technical detail that matters for your interpretation.** Grayscale collapses the image *to* luma (0.299R + 0.587G + 0.114B). Hue rotation preserves HSV's V channel (`max(R,G,B)`) but *not* luma — rotating hue permutes which channel is the maximum. Concretely: pure red (255,0,0) → green (0,255,0) keeps V at 255 but shifts luma from 76 to 150.

So the two are **not comparable through luminance**. Geometry is verified preserved via V-channel gradient energy (identical to three decimals before and after). Don't claim "both preserve brightness" — they preserve different things.

**How to read it.** A model relying on colour drops sharply on both. A shape/texture-driven model barely moves. The interesting case is asymmetry: large grayscale drop but small hue drop would suggest the model uses saturation/chroma *presence* rather than specific hues.

Look at **per-class** patterns, not just the mean. Classes with diagnostic colours (deer/horse browns, sky-blue behind airplane and bird) should suffer more if colour is being used.

---

## Experiment 3 — Cue conflict (shape vs texture)

**Concept.** The most direct test in Task 1. Build an image whose *shape* says class A and whose *texture* says class B. Whatever the model predicts reveals which cue it weighted.

This follows Geirhos et al. (2019), who found ImageNet-trained CNNs to be strongly texture-biased — a result that surprised the field.

**How the images are made.** AdaIN style transfer: encode content and style into VGG feature space, replace the content's per-channel statistics with the style's, decode. Content keeps its spatial structure (shape); style supplies the statistics (texture).

$$\text{Shape Bias} = \frac{N_{\text{shape}}}{N_{\text{shape}} + N_{\text{texture}}} \times 100 \qquad \text{Coverage} = \frac{N_{\text{shape}} + N_{\text{texture}}}{N_{\text{total}}} \times 100$$

⚠️ **Coverage is not optional garnish — it determines whether shape bias means anything.**

Shape bias only counts predictions landing on one of the two intended classes. If the model predicts a *third* class, that prediction is discarded from the numerator and denominator. So:

- Shape bias 80% with coverage 60% → based on 120 of 200 images. Reasonable evidence.
- Shape bias 80% with coverage 8% → based on 16 images. **Nearly meaningless.**

The PDF states it plainly: *"A high shape-bias value based on very few shape-or-texture decisions is weak evidence."* Report them together, always, and say in the report what coverage you got.

**Why the rejection rule must be model-blind.** Stylization sometimes fails — output washed out, shape destroyed, style not applied. Those images are ambiguous and get discarded. But the rule uses **only image statistics** (structure ratio, saturation, amount of change), never model predictions.

If you filtered on predictions — dropping images the model "got wrong" — you'd keep only images behaving as expected, and shape bias would measure *your filtering*, not the model. The measurement would be circular. A test inspects the function signature and asserts no model argument exists, so it cannot happen even accidentally.

Accepted and rejected counts are both recorded in `summary.json`. Report them.

---

## Experiment 4 — Translation

**Concept.** Does moving the object change the answer?

Convolution is often described as translation-equivariant, but in practice: stride, pooling, padding and the global-average-pool readout all break exact equivariance. ViT is different again — it uses *learned absolute positional encodings*, so it has no architectural reason to be shift-invariant at all.

**Design.** Displacements 0, 8, 16, 32 pixels × 4 cardinal directions, averaged. **Reflection padding**, not zero padding — zero padding would add black bars, which is a different intervention (you'd be testing "response to black borders" as well as displacement).

$$\text{Consistency}(\delta) = \frac{1}{N}\sum_i \mathbb{1}[\hat y(x_i) = \hat y(T_\delta(x_i))]$$

⚠️ **Consistency measures prediction *stability*, not correctness.** An image wrong before and identically wrong after counts as consistent. It answers "did the prediction move?", not "was it right?" — report accuracy alongside it.

**How to read it.** The **slope** is the finding, not the intercept. All models will degrade; the question is how fast. Expect monotone decline. A ViT declining faster than ResNet is consistent with absolute positional encoding, but see the confounding caveat above before claiming architecture caused it.

---

## Experiment 5 — Patch shuffle

**Concept.** Destroy global spatial organisation while keeping every pixel and most local texture. A 4×4 grid, one non-identity permutation per image, seed 6304.

A model relying on local texture statistics should barely notice. A model relying on coherent global structure should collapse.

**Why 4×4 pixel-space and not 16×16 to match ViT tokens.** Deliberate, and the PDF says so: the experiment is about *reliance on coherent spatial arrangement*, not about the mechanics of a particular tokenization. Matching the token grid would test something narrower.

⚠️ **The trap the PDF highlights: "A confident prediction after patch shuffling is not necessarily a sensible prediction."**

This is why mean max confidence is reported here. A model that stays 90% confident on scrambled images is telling you it never needed global structure — it was reading texture all along. That's a *finding*, and arguably a more interesting one than the accuracy drop.

---

## Experiment 6 — Representation analysis

**Concept.** Everything above measured *predictions*. This measures the *representation* — and they can disagree.

$$I_T = \frac{1}{N}\sum_i \frac{f(x_i)^\top f(T(x_i))}{\|f(x_i)\|_2 \|f(T(x_i))\|_2}$$

Paired cosine similarity between each clean image's feature and its transformed counterpart's. 1.0 = unchanged, 0 = orthogonal, -1 = reversed.

**The four informative cases** — this is the heart of RQ3:

| Prediction | Representation | Interpretation |
|---|---|---|
| stable | stable | The model genuinely doesn't encode this factor |
| stable | **moved a lot** | The cue **is** encoded, but the classifier doesn't use it |
| **changed** | stable | The classifier is brittle near a decision boundary |
| changed | moved | Straightforward sensitivity |

Row 2 is the one worth hunting for. It's the concrete answer to *"a cue may remain encoded even when the classifier does not use it."*

**t-SNE.** One projection fitted to clean **and** transformed features **combined**, per backbone. Colour = true class, marker = condition.

Fitting one projection is what makes the conditions comparable — separate fits would land in unrelated coordinate systems.

⚠️ **Three things you must not claim:**
1. Coordinates from different backbones are **not comparable**. Each projection is fitted separately. Discuss within-plot structure only.
2. t-SNE preserves *neighbourhoods*, not distances. "These points are twice as far apart" is not a supportable statement.
3. Report the settings (perplexity 30, 1000 iterations, PCA init, cosine metric, seed 6304). t-SNE output depends strongly on them.

**What to look for:** Do classes stay separated? Do clean and transformed clusters overlap or split? Which transformation visibly drags points away from their clean clusters?

---

## The four research questions

**RQ1 — What do colour and cue-conflict jointly reveal? How does coverage affect your conclusions?**

Combine Experiments 2 and 3. The explicit second clause is about coverage — state your actual coverage numbers and say honestly how much they support.

**RQ2 — How do the models respond to translation and patch shuffling?**

Combine 4 and 5. Translation probes *positional* sensitivity; patch shuffle probes *global organisation*. A model robust to translation but destroyed by shuffling relies on structure but not absolute position.

**RQ3 — Do prediction changes agree with representation changes? Compare CLIP zero-shot with its trained head.**

Use the four-case table. The CLIP comparison is your cleanest controlled result: identical frozen features, different classifiers. Any difference is attributable to the classifier alone.

**RQ4 — What is attributable to architecture vs pretraining/supervision/augmentation/capacity?**

The confounding question. Be honest: ResNet-vs-ViT is your least-confounded architecture comparison; anything involving CLIP is confounded by 400M web pairs and language supervision. Use failure cases to argue specifics.

---

## Required evidence checklist

- [ ] Clean / grayscale / hue-90 / patch-shuffle comparison table
- [ ] Shape, texture and **other** decision counts, plus shape bias **and coverage**
- [ ] Translation curve: accuracy and consistency vs displacement
- [ ] Representation stability for all four required interventions
- [ ] t-SNE comparing clean and transformed, with settings reported
- [ ] A small set of informative cue-conflict agreements, disagreements or failures

## Where the numbers land

| File | Contents |
|---|---|
| `task1/results/intervention_results.csv` | Every (model, condition): accuracy, macro-F1, confidence, consistency, delta |
| `task1/results/representation_stability.csv` | Cosine stability per model and condition |
| `task1/results/translation_<backbone>.csv` | Translation curves |
| `task1/results/clip_zeroshot.json` | Zero-shot vs head agreement |
| `task1/data/generated/cue_conflict/summary.json` | **Accepted/rejected counts** |
| `figures/task1/tsne_*.png` | t-SNE plots |
