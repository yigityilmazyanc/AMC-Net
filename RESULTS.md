# Experiment Results — MYZ307E Course Project (ITU)

## Dataset
RML2016.10a — 11 modulation classes, 20 SNR levels (−20 to +18 dB), 220K samples, 60/20/20 stratified split.

## Paper Baseline (AMC-Net, original)
| Accuracy | Macro F1 | Cohen's Kappa |
|---|---|---|
| 62.51% | 64.83% | 58.85% |

## Our Results — With TTA (K=8 phase rotations at inference)

### 3-seed mean±std (seeds: 42, 123, 456)

| Variant | Acc mean±std | F1 mean±std | Kappa mean±std |
|---|---|---|---|
| Baseline (reproduced) | 62.40±0.25% | 64.80±0.22% | 58.64±0.28% |
| **+A (augmentation)** | **62.64±0.47%** ✓ | **65.13±0.51%** ✓ | **58.90±0.52%** ✓ |
| +B (SNR-weighted loss) | 62.31±0.28% | 64.60±0.11% | 58.54±0.30% |

**+A beats paper on all three metrics (mean across 3 seeds).**

### Best Single Run (seed=42, variant A + TTA)
| Accuracy | Macro F1 | Cohen's Kappa |
|---|---|---|
| **63.13%** ✓ | **65.66%** ✓ | **59.45%** ✓ |

### Without TTA (3-seed mean, for reference)
| Variant | Acc | F1 | Kappa |
|---|---|---|---|
| Baseline | 61.92±0.26% | 64.00% | 58.11% |
| +A | 62.21±0.42% | 64.64% | 58.43% |

## Comparison Baselines
| Method | Accuracy |
|---|---|
| SVM + cumulants | 28.67% |
| ResNet-18 1D (3.85M params) | 61.69% |
| AMC-Net baseline (0.47M params) | 62.40% |
| AMC-Net + A + TTA (ours) | **62.64%** |

## Improvements

### A — SNR-Conditioned Augmentation
- Phase rotation θ ~ U[0°, 360°)
- AWGN noise: 5–15% of per-sample RMS, inversely scaled with SNR
- Circular temporal shift 1–5 samples for SNR ≤ 0 dB

### TTA — Test Time Augmentation
- K=8 uniformly-spaced phase rotations at inference
- Predictions averaged → variance reduction
- Zero training cost, +0.43% accuracy gain

### B — SNR-Weighted CE Loss
- w(snr) = 1/(1+exp(snr/T)), T=50
- Consistently slightly below baseline — down-weights high-SNR gradient signal

### C — SE Block (after Conv_stem, before FFM)
- GAP → FC(256→16) → ReLU → FC(16→256) → Sigmoid
- Marginally positive but within noise floor

## Key Findings
1. **A + TTA beats paper on all three metrics** (mean of 3 seeds)
2. **TTA adds +0.43% accuracy** with zero training cost — phase-rotation-trained models benefit from rotation averaging at inference
3. **B hurts performance** — SNR-based loss weighting suppresses high-SNR learning; not recommended
4. **AMC-Net is parameter-efficient** — 0.47M params matches ResNet-18 (3.85M) on this dataset
5. **SVM baseline confirms deep learning advantage** — 28.67% vs 62%+ for neural approaches

## Training Config
- Optimizer: Adam, weight_decay=5e-4
- LR: CosineAnnealingLR (1e-3 → 1e-5, T_max=100)
- Batch: 64, Epochs: 100 (max), patience: 20
- Label smoothing: ε=0.05, FFM dropout: 0.6
- Seeds: 42, 123, 456 (stratified 60/20/20 split per seed)
- TTA: K=8 phase rotations averaged at inference
