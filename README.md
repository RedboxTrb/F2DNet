# F2D-RetinaNet

Reference implementation of **FHAU-Net** (fractional Hessian Attention U-Net) for retinal vessel
segmentation and **F2D-RetinaNet** for diabetic retinopathy grading.

Delu, Harjule, Kumar, Gajjar and Nkomozepi — Malaviya National Institute of Technology Jaipur.

This repository contains the source code, the fractional filter coefficients, the data split
files supporting the paper.

---

## Method

Fundus images pass through three stages:

1. **FADHE** — fractional anisotropic diffusion, CLAHE, bilateral filtering and unsharp masking,
   all restricted to the fundus field of view.
2. **FHAU-Net** — an Attention U-Net taking a 10-channel input: the green channel, eight
   Atangana–Baleanu–Caputo fractional Hessian vesselness maps, and a CLAHE channel.
3. **F2D-RetinaNet** — dual-stream grading over the enhanced image and the vessel map.

### Fractional Hessian operator

Eight ABC fractional orders, **v ∈ {0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3}**, Gaussian
pre-smoothing **σ = 1.45**. With `M(v) = 1 − v + v/Γ(v)`:

```
p = (2·Γ(v)·(1−v) + 1) / (2·M·Γ(v))
q = 2^(v−1) / (M·Γ(v))
r = (3^v − 1) / (2·M·Γ(v))

kernel_x = [[ p, 0, −p]      kernel_y = [[ p,  q,  r]
            [ q, 0, −q]                  [ 0,  0,  0]
            [ r, 0, −r]]                 [−p, −q, −r]]
```

| v | M | p | q | r |
|---|---|---|---|---|
| 0.6 | 0.80290298 | 0.91636537 | 0.63383200 | 0.39023169 |
| 0.7 | 0.83926823 | 0.81641550 | 0.74558474 | 0.53132534 |
| 0.8 | 0.88714962 | 0.70954042 | 0.84286584 | 0.68172059 |
| 0.9 | 0.94220085 | 0.60272644 | 0.92667335 | 0.83818533 |
| 1.0 | 1.00000000 | 0.50000000 | 1.00000000 | 1.00000000 |
| 1.1 | 1.05625071 | 0.40290482 | 1.06658461 | 1.16850010 |
| 1.2 | 1.10694931 | 0.31127190 | 1.13020120 | 1.34655830 |
| 1.3 | 1.14851526 | 0.22387274 | 1.19440593 | 1.53826848 |

Also in machine-readable form in `coefficients/`.

The Hessian is formed by re-convolving the first derivatives — `gxx = kx*(kx*I)`,
`gxy = kx*(ky*I)`, `gyy = ky*(ky*I)` — and vesselness is `max(|λ₁|, |λ₂|)` restricted to dark
structures (`λ₁ < 0` or `λ₂ < 0`). Note that at v = 1.0 the kernel is not symmetric (p ≠ r), so
the integer-order case does not reduce to a standard centred gradient.

### FADHE parameters

α = 0.2, K = 0.05, Δt = 0.1, 20 iterations, Grünwald–Letnikov truncation depth 10, weights
`w₀ = 1`, `w_{i+1} = w_i(1 − (α+1)/(i+1))`, symmetric padding of width 10, updates applied only
inside the field-of-view mask.

---

## Repository layout

```
code/preprocessing/   generate_abc_features.py    10-channel ABC feature generation
                      abc_fractional.py           standalone ABC kernel implementation
code/models/          model_multiscale.py         FHAU-Net
code/training/        train_multiscale.py         training loop (Tversky + Focal)
                      dataset_multiscale.py       data loader
code/evaluation/      evaluate_test_set_multiscale.py
code/experiments/     scripts reproducing every reported experiment
coefficients/         ABC kernel coefficients and the training configuration
splits/               train / val / test partitions
```

Trained weights are too large for git and are deposited separately — see **Weights** below.

---

## Reproducing the results

```bash
# 1. generate the 10-channel input representation
python code/preprocessing/generate_abc_features.py \
    --dataset_dir Dataset --output_dir Dataset/multiscale_features

# 2. train
python code/training/train_multiscale.py \
    --dataset_dir  Dataset \
    --features_dir Dataset/multiscale_features \
    --splits_dir   splits \
    --output_dir   models_multiscale \
    --num_epochs   120

# 3. evaluate
python code/evaluation/evaluate_test_set_multiscale.py \
    --model models_multiscale/best_model.pth
```

Expected data layout:

```
Dataset/images/{name}.png
Dataset/ground_truth/{name}_gt.png
Dataset/masks/{name}_mask.png
Dataset/multiscale_features/{name}_features.npy     float32 [H, W, 9]
```

**Training configuration** (`coefficients/training_config.json`): AdamW, lr 1×10⁻⁴, weight decay
1×10⁻⁵, batch size 5 with gradient accumulation over 4 steps, `ReduceLROnPlateau(patience=7,
factor=0.5)`, loss `0.7·Tversky(α=0.7, β=0.3) + 0.3·Focal(α=0.25, γ=2.0)`, mixed precision,
input resolution 1024×1024.

Feature generation takes roughly 4 s per 1024×1024 image on one CPU core and produces a 36 MB
array per image, so budget about 32 GB for a 900-image corpus. It parallelises across cores.

### Additional experiments

`code/experiments/` reproduces the analyses reported in the paper:

| Script | What it produces |
|---|---|
| `lodo_pipeline.py`, `lodo_eval.py` | leave-one-dataset-out training and evaluation |
| `exp_variants.py` | matched-protocol architecture, fractional-order and seed comparisons |
| `exp_seg_eval.py` | per-dataset segmentation metrics with bootstrap confidence intervals |
| `exp_fadhe.py` | FADHE component ablation and conservative-flux-form comparison |
| `exp_order_redundancy.py` | correlation and effective dimensionality of the eight orders |
| `c20_prep.py`, `c20_train.py` | classification input-representation ablation |

---

## Weights

Trained checkpoints are not stored in git (the vessel model alone is 377 MB, and each grading
stage is 1.75 GB). They are archived separately:

> Zenodo DOI: *to be added on publication*

---

## Data

All datasets are obtained from their original providers and none are redistributed here.

**Vessel segmentation** — FIVES, DRIVE, STARE, CHASE-DB1, HRF.
**DR grading** — APTOS 2019, IDRiD, Messidor-2, SUSTech-SYSU, DeepDRiD-v1.1, Zenodo DR V03.

Clinical images acquired at SMS Hospital are **not** included and are not redistributable.

---

## Citation

```bibtex
@article{delu2026f2dretinanet,
  title   = {F2D-RetinaNet: fractional Hessian vessel segmentation and dual-path
             diabetic retinopathy grading},
  author  = {Delu, Mukesh and Harjule, Priyanka and Kumar, Rajesh and
             Gajjar, Kushal and Nkomozepi, Pilani},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Under review}
}
```

---

## Licence

*To be selected by the authors before public release — MIT or Apache-2.0 unless institutional
policy requires otherwise.*
