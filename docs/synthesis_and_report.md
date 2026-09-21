# Cross-Task Synthesis and Report Guide

Written for: Talha, for the report's discussion sections and the 8-page structure.

---

## The unifying question

The PDF states it directly:

> **What should a robust representation preserve, what should it become invariant to, and when should it reject an input rather than force a prediction?**

Each task answers one piece:

| Task | Question it answers |
|---|---|
| 1 | What evidence do models actually **use**? |
| 2 | Can unlabeled target data help you **adapt**? |
| 3 | Can you generalise to a domain you have **never seen**? |
| 4 | When should the model **refuse** to answer? |

The assignment explicitly asks you to synthesise these *"into one coherent study rather than four disconnected homework answers."* The discussion section is where most of the marks for insight live.

---

## The five cross-task threads

### 1. Visual cues and distribution shift (Tasks 1 ↔ 2/3)

Task 1 tells you which cues each model relies on. Tasks 2/3 show what happens under real domain shift.

**The connection to look for.** PACS Sketch removes *colour* and *texture* almost entirely, leaving shape and line structure. If Task 1 shows a texture-biased model, its collapse on Sketch is coherent — the cue it depended on is simply gone.

**Concrete question:** do the classes that fail worst on Sketch correspond to classes where texture mattered most?

⚠️ **This is a qualitative comparison, not a causal claim.** The PDF says so explicitly. Different datasets (STL-10 vs PACS), different architectures (ResNet-50/ViT/CLIP vs ResNet-18), different training (frozen probe vs full fine-tune). You are noting a *consistent pattern*, not demonstrating that texture bias caused the Sketch failure. Say that in the report.

### 2. Invariance versus discriminability (Tasks 2 ↔ 3)

The thread running through both alignment tasks: **does making domains less distinguishable actually improve recognition?**

Four cases to look for across DAN, DANN, CDAN, DAN-DG:

| Separability | Target/Sketch accuracy | What happened |
|---|---|---|
| ↓ | ↑ | Invariance removed nuisance variation — the hoped-for case |
| ↓ | ↓ | ⚠️ Invariance also removed **class-discriminative** information |
| ↔ | ↑ | Transfer happened without needing invariance |
| ↓ | ↔ | Domains merged but nothing useful followed |

Row 2 is the most valuable finding if you get it. Evidence: separability falls toward chance, source performance drops, target doesn't improve, and per-class results show specific classes collapsing.

Zhao et al. (2019) give the theory: marginal alignment alone is insufficient, and can be actively harmful when label distributions or class-conditional structures differ across domains.

### 3. The value of target access (Task 2 DAN ↔ Task 3 DAN-DG)

**Your cleanest cross-task comparison**, and why the λ grids were matched.

- Identical MMD implementation, kernels, bandwidth heuristic
- Identical grid {0.1, 1, 10}
- Identical ERM baseline (literally the same checkpoint)
- **Difference:** DAN aligns source↔unlabeled Sketch; DAN-DG aligns source↔source

Compare Sketch accuracy at matched λ. The gap is your estimate of what unlabeled target data bought.

**Limitations you must state** (RQ4 asks for them):
- Different *pairings*, not just target presence: source↔target vs source↔source is not one variable
- DAN's alignment signal comes from a genuinely distant distribution; DAN-DG's comes from three similar ones
- One seed, one architecture, one target domain — no error bars
- Sketch is unusually far from all sources; a milder shift might behave differently

### 4. Recognition versus rejection (Task 4 ↔ everything)

Do representations that classify better also **abstain** better?

Task 4's GCSC tests this directly: better augmentation → better CSA → better rejection? The PDF warns it need not follow.

**The deeper connection.** Tasks 2/3 face *covariate* shift — same classes, different appearance. Task 4 faces *semantic* novelty — genuinely new classes. A model can be robust to one and fragile to the other.

**The near/far split is your sharpest tool here.** Far unknowns are easy; near unknowns require the fine distinctions that also matter under domain shift. Where should a confident classifier instead abstain?

### 5. Overall synthesis

The three-part answer, grounded in your evidence:

| | Question | Evidence source |
|---|---|---|
| **Preserve** | What information must survive for classification? | Task 1 cue reliance; Tasks 2/3 class-discriminative info surviving alignment |
| **Suppress** | What variation should it become invariant to? | Tasks 2/3 separability vs accuracy — and where invariance went too far |
| **Reject** | What belongs outside the label space? | Task 4 near/far, and where proxies were insufficient |

---

## Report structure — 8 pages

Suggested organisation from the PDF: Abstract; Introduction and research questions; Experimental Setup/Methods; Results for Tasks 1–4; Cross-task Discussion; Conclusion; References.

A workable budget:

| Section | Pages | Notes |
|---|---|---|
| Abstract | 0.15 | **GitHub link goes at the end of the abstract** — required |
| Introduction + RQs | 0.6 | Frame the unifying question |
| Setup / Methods | 0.75 | Keep paper summaries **short** |
| Task 1 | 1.4 | Table + translation curve + t-SNE |
| Task 2 | 1.4 | Main table + per-class + λ study |
| Task 3 | 1.3 | Main table + diagnostics + λ study |
| Task 4 | 1.4 | Two tables + score figure + failures |
| Cross-task discussion | 0.8 | **Where the insight marks are** |
| Conclusion | 0.2 | |

References don't count. An appendix may hold extra detail, but **graders aren't required to read it** — it cannot carry required evidence.

⚠️ **The PDF's guidance on space:** *"Keep paper-method summaries concise and spend your limited space on controlled evidence, comparisons, failure analysis, and interpretation."*

Don't spend half a page explaining how MMD works. Spend it on what your MMD results showed and what they mean.

---

## Writing the analysis well

**Separate these six things explicitly** — this is the core of `01_AGENT_RULES.md` and good scientific writing:

1. **Requirement** — what the assignment mandated
2. **Implementation decision** — what you chose, and why (hue rotation over palette transfer; t-SNE over UMAP)
3. **Hypothesis** — what you expected *before* seeing results
4. **Result** — what the numbers say
5. **Interpretation** — what you think it means
6. **Limitation / alternative explanation** — what else could explain it

The pre-registered hypotheses in `dan_lambda_study.yaml` and `dan_dg_lambda_study.yaml` are already written down. Use them: *"we expected X; we observed Y"* is far stronger than an unfalsifiable post-hoc story.

**Report negative and null results.** ERM beating DAN-DG is a legitimate finding (Gulrajani & Lopez-Paz found exactly that pattern at scale). A method failing to help, honestly analysed, reads better than a marginal gain overclaimed.

**Never round a number in your favour, and never report a `--smoke` number.**

---

## Before you submit — the PDF's own checklist

- [ ] No target labels influenced Task 2 model selection
- [ ] No Sketch data consumed by Task 3 training or model selection
- [ ] Task 2 target results did not influence Task 3 settings
- [ ] No real unknowns influenced Task 4 training or threshold selection
- [ ] Every reported number traces to a saved result file or reproducible command
- [ ] Every external implementation is attributed
- [ ] GitHub link at the end of the abstract
- [ ] 8 pages of main content, NeurIPS format

**On the AI policy:** you may not use generative AI to write any part of the PDF report — the language, interpretation and analysis must be entirely your own. These documents exist to help you *understand* the experiments so you can write about them yourself. Coding assistance is permitted and is attributed in the README.

---

## Document map

| Document | Use it for |
|---|---|
| [task1_concepts.md](task1_concepts.md) | Interventions, shape bias, coverage, representation analysis |
| [task2_concepts.md](task2_concepts.md) | UDA methods, separability, negative transfer |
| [task3_concepts.md](task3_concepts.md) | DG, information boundary, SAM, sharpness |
| [task4_concepts.md](task4_concepts.md) | OSR scores, PROSER, near/far, thresholds |
| [common.md](common.md) · [shared.md](shared.md) · [task2.md](task2.md) | Code-level detail |
| [adain_weights.md](adain_weights.md) | AdaIN and the rejection rule |
| [../progress.md](../progress.md) | What was run, what was found, what's unresolved |
