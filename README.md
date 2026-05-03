# AMC-Net — MYZ307E Course Project (ITU)

> **Course:** MYZ307E — Istanbul Technical University  
> **Goal:** Reproduce and improve AMC-Net on RML2016.10a dataset

---

## What is AMC-Net?

Automatic Modulation Classification (AMC) identifies the modulation scheme of a received radio signal (BPSK, QAM16, WBFM, etc.) directly from raw I/Q samples — without prior knowledge of the signal.

AMC-Net processes raw I/Q signals (shape `2×128`) through four modules:

```
Input (B, 2, 128)
  │
  ▼  ACM — Adaptive Correlation Module
     FFT → per-channel MLP (Re/Im) → IFFT → residual add
  │
  ▼  MSM — Multi-Scale Module
     Parallel Conv2d kernels 3/5/7 → concat → (B, 36, 1, 128)
  │
  ▼  Conv_stem — Feature Extractor
     3× Conv_Block [36→64→128→256]
  │
  ▼  FFM — Feature Fusion Module
     Multi-head self-attention (2 heads, 50% dropout)
  │
  ▼  GAP → Linear(512→512) → Dropout → PReLU → Linear(512→11)
  │
  ▼  Class logits (B, 11)
```

**Model size: 0.47M parameters**

---

## Dataset

**RML2016.10a** — 11 modulations, 20 SNR levels (−20 to +18 dB), 220K signals, each 128 complex I/Q samples.

```bash
# Convert txt mirror to pkl (run once)
python convert_dataset.py
# Output: data/RML2016.10a_dict.pkl
```

---

## What We Did

### Improvement A — SNR-Conditioned Augmentation (`feature/A`)

Applied during training only (`util/augment.py`):

1. **Phase rotation** θ ~ U[0°, 360°)  
   Rotates the IQ constellation — teaches phase invariance.

2. **SNR-aware AWGN**  
   Noise = 5–15% of per-sample RMS, inversely scaled with SNR.  
   *Critical: noise must be relative to signal RMS (≈0.006), not absolute. Absolute σ=0.01–0.05 was 2–8× the signal amplitude and destroyed performance (27% accuracy).*

3. **Circular temporal shift** 1–5 samples — only for SNR ≤ 0 dB.

### Improvement B — SNR-Weighted CE Loss (`feature/B`)

```
w(snr) = 1 / (1 + exp(snr / T)),   T = 50
loss = mean( w · CE(logit, label) )
```

Low-SNR samples weighted higher.  
**Result: slightly below baseline** — down-weighting high-SNR gradient hurts overall accuracy. Reported as a negative result.

### Improvement C — SE Channel Attention (`feature/C`)

Squeeze-and-Excitation block after Conv_stem, before FFM (C=256, r=16).  
Result: marginal, within noise floor.

### TTA — Test Time Augmentation

At inference, average predictions over 8 uniformly-spaced phase rotations:

```python
logit = mean([model(rotate(x, k·45°)) for k in range(8)])
```

Phase-rotation-trained model benefits from rotation averaging.  
**Zero training cost. +0.43% accuracy.**

---

## Results

**Paper target:** 62.51% acc | 64.83% F1 | 58.85% Kappa

### 3-seed mean±std (seeds 42, 123, 456) — with TTA

| Variant | Accuracy | Macro F1 | Kappa |
|---|---|---|---|
| Baseline (reproduced) | 62.40 ± 0.25% | 64.80 ± 0.22% | 58.64 ± 0.28% |
| **+A (augmentation)** | **62.64 ± 0.47% ✓** | **65.13 ± 0.51% ✓** | **58.90 ± 0.52% ✓** |
| +B (weighted loss) | 62.31 ± 0.28% | 64.60 ± 0.11% | 58.54 ± 0.30% |

**+A beats paper on all three metrics.**

### Best run — A + TTA, seed=42

| Accuracy | Macro F1 | Kappa |
|---|---|---|
| **63.13%** | **65.66%** | **59.45%** |

### Comparison baselines

| Method | Params | Accuracy |
|---|---|---|
| SVM + cumulants | — | 28.67% |
| ResNet-18 1D | 3.85M | 61.69% |
| AMC-Net baseline | 0.47M | 62.40% |
| **AMC-Net + A + TTA** | **0.47M** | **62.64% ✓** |

---

## How to Run

```bash
# Install dependencies
pip install torch torchvision tqdm pyyaml scikit-learn pandas matplotlib h5py scipy

# Baseline training
python main.py --seed 42 --exp_tag baseline

# Improvement A
git checkout feature/A
python main.py --seed 42 --exp_tag A

# Full ablation (3 seeds, ~12h)
./run_ablation.sh 42 123 456

# Baselines
python baselines/svm_baseline.py --seed 42
python baselines/train_resnet.py --seed 42
```

---

## Repository Structure

```
AMC-Net/
├── main.py                  # entry point
├── run_ablation.sh          # multi-seed ablation runner
├── convert_dataset.py       # dataset converter
├── RESULTS.md               # full numerical results
├── config/2016.10a.yml      # hyperparameters
├── models/model.py          # AMC_Net + SEBlock
├── data_loader/             # dataset loading & splitting
├── util/
│   ├── training.py          # Trainer (train/val loop, early stopping)
│   ├── evaluation.py        # Run_Eval with TTA
│   ├── augment.py           # SNR-conditioned augmentation (feature/A)
│   └── ...
└── baselines/
    ├── svm_baseline.py      # LinearSVC on 8 cumulant features
    └── train_resnet.py      # 1D ResNet-18 baseline
```

## Branches

| Branch | Content |
|---|---|
| `main` | clean baseline |
| `feature/A` | augmentation |
| `feature/B` | SNR-weighted loss |
| `feature/C` | SE block |
| `exp/A+C` | augmentation + SE |
| `exp/A+B+C` | all three |

## Training Config

| Parameter | Value |
|---|---|
| Optimizer | Adam, weight_decay=5e-4 |
| LR | CosineAnnealingLR (1e-3 → 1e-5, T_max=100) |
| Batch | 64 |
| Epochs | 100 max, patience=20 (val_acc) |
| Label smoothing | ε=0.05 |
| TTA | K=8 phase rotations at eval |
