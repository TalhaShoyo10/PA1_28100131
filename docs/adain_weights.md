# AdaIN Weights — What to Download and Why

Written for: Talha, covering the one external dependency in this repository.

---

## Why AdaIN needs downloaded files at all

Task 1 step 3 needs **cue-conflict images**: shape from class A, texture from class B. Ask the model what it sees, and the answer reveals which cue it used.

AdaIN (Huang & Belongie, 2017) is not a pixel formula — it's a small trained network in three parts:

| Part | What it is | Origin |
|---|---|---|
| **Encoder** | VGG-19 truncated at `relu4_1` | ImageNet-pretrained, *weight-normalised* by the AdaIN authors |
| **AdaIN layer** | Pure arithmetic, **no parameters** | Implemented in this repo — `task1/models/adain.py` |
| **Decoder** | Mirror of the encoder, features → RGB | **Trained by the authors** on MS-COCO + WikiArt |

The AdaIN operation itself is simple, and it's my own code:

$$\text{AdaIN}(c, s) = \sigma(s)\left(\frac{c - \mu(c)}{\sigma(c)}\right) + \mu(s)$$

Normalise the content feature, then rescale and reshift it by the style's per-channel mean and standard deviation. Content keeps its *spatial structure*; style supplies the *statistics*.

**The decoder is the problem.** After mixing statistics in feature space you must invert back to an image. That mapping is learned, not derivable. Training it yourself would be a project of its own.

The encoder is also not plain `torchvision.vgg19` — the authors use a weight-normalised variant with a specific truncation, so the standard checkpoint is **not** a drop-in substitute.

---

## Status on this machine

**Not downloaded.** I attempted it; the sandbox this session runs in allows read-only HTTP probes but refuses sustained downloads that write to disk (`ConnectionRefusedError` / timeout). The URLs themselves are live — a probe returned `200` with `Content-Length: 14023458`.

This is a restriction on my environment, **not** a problem with your network, the URLs, or the code. On Colab it will download normally.

## What is verified without them

| Verified now | How |
|---|---|
| AdaIN math is correct | Output statistics match style to ~1e-5; content correlation 1.000000 |
| Encoder shape | 224×224 → `(1, 512, 28, 28)` — three pooling stages |
| Decoder shape | `(1, 512, 28, 28)` → 224×224 |
| **Architecture matches the checkpoint** | Parameter shapes asserted against the reference definition |
| Stylizer plumbing | Runs end-to-end on random weights; output clamped to [0,1], deterministic |
| Error path | Missing weights → clear `FileNotFoundError` pointing at the README |

One test — `test_real_weights_load_into_the_architectures` — is **skipped** until the files exist. It is not silently passing.

---

## How to get them

### On Colab (automatic)

`AdaINStyleTransfer()` downloads on first construction. Cache to Drive so it happens once:

```python
AdaINStyleTransfer(weights_dir="/content/drive/MyDrive/AToML_PA1/adain_weights")
```

### Manually

```bash
mkdir -p task1/models/weights
curl -L -o task1/models/weights/vgg_normalised.pth \
  https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth
curl -L -o task1/models/weights/decoder.pth \
  https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth
```

Roughly 110 MB total (decoder ≈ 14 MB, VGG ≈ 80 MB). Both are gitignored — never commit them.

**Verify after downloading:**
```bash
python -m pytest tests/test_adain.py -q
```
The skipped test should now run. If the architecture didn't match, `load_state_dict` would fail loudly here.

---

## Attribution — required for your README

The assignment states: *"Clearly attribute materially reused external code in your README."*

> **AdaIN style transfer.** The AdaIN operation, VGG encoder truncation and decoder architecture in `task1/models/adain.py` are reimplemented from Huang & Belongie (2017), *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*, ICCV 2017. The **pretrained encoder and decoder weights** are downloaded from the public PyTorch port [`naoto0804/pytorch-AdaIN`](https://github.com/naoto0804/pytorch-AdaIN) (release `v0.0.0`, MIT licence) and are **not authored here**.

Be ready to say in a viva: *the AdaIN math is mine, the trained decoder is theirs.*

---

## The rejection rule — and why it's model-blind

The assignment demands a visual rejection rule **defined before model evaluation**, and forbids using model predictions to decide what to keep.

**Why that matters.** If you dropped images the model got "wrong", you'd keep only images it handles the way you expected — and shape bias would measure your filtering, not the model. The measurement would be circular.

Four criteria, all pure image statistics:

| Criterion | Catches | Threshold |
|---|---|---|
| `content_structure_lost` | Stylization destroyed the shape | V-gradient ratio < 0.30 |
| `saturation_collapse` | Washed-out grey output | mean HSV saturation < 8 |
| `style_not_applied` | Output ≈ the content image | mean abs change < 0.05 |
| `degenerate_content` | Source image had no structure | zero gradient |

`test_rejection_rule_never_sees_a_model` inspects the function signature and asserts no model or prediction argument exists — so it cannot be passed one even by accident.

Every generated image is logged to `manifest.csv` with its accept/reject status and reason, and `summary.json` records the counts the assignment asks you to report.

⚠️ **Tune these thresholds before evaluating models, never after.** If you fall short of 200 accepted conflicts, adjust and regenerate — but do it *before* any model sees the images, and record that you did.
