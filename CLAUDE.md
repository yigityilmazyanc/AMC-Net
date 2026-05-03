# AMC-Net — CLAUDE.md (source of truth for this project)

## Token Efficiency

- **Only read `*.py` and `*.yml`** unless explicitly asked otherwise.
- **Never read or open**: `.venv/`, `__pycache__/`, `data/temp_dataset/`, `checkpoint/`, `training/`, `inference/`, `data/*.pkl`, `data/*.tar.bz2`, `*.pt`, `*.pth`, `*.npy`, `*.png`, `*.jpg`, `*.pdf`, `*.tar`, `*.bz2`, `assets/`.
- Do not `cat` or `head` binary files.
- Prefer `grep`/`find` over reading whole files when looking for a specific symbol.

## Engineering Discipline

- **Minimal changes**: fix or add only what the task requires; no opportunistic refactoring.
- **Mark every improvement edit** with an inline comment: `# IMPROVEMENT A`, `# IMPROVEMENT B`, or `# IMPROVEMENT C`.
- **Branches**: all improvement work lives on `feature/A`, `feature/B`, or `feature/C`; baseline fixes on `main`.
- **Small commits**: one logical change per commit, descriptive message.
- **Ask before architectural decisions** (new modules, changes to forward signatures, split/merge files).
- Do not add error handling, fallbacks, or abstraction layers beyond what the task requires.
- Do not add comments that explain *what* code does — only add a comment when the *why* is non-obvious.

## GPU / Hardware Rules

- Hardware: RTX 4070 Laptop, **8 GB VRAM**, WSL2 Ubuntu.
- Try `batch_size=400` first; fall back to `256` if OOM.
- Always use `pin_memory=True` and `num_workers=4` in DataLoaders.
- Keep tensors on the correct device; never silently CPU-fallback.

## Session Management

- **Warn me** if the topic drifts outside this project or context approaches ~30 k tokens.
- Stop and ask for review after each of the four initial engineering steps (dataset → codebase → baseline → plan).
- Reply in English; keep replies concise.

---

## Project Goal

Beat AMC-Net baseline on **RML2016.10a** (target: >62.51% acc, >0.6483 F1, >0.5885 Kappa) with three modular improvements.

### Improvement A — SNR-Conditioned Augmentation (training only)
- Phase rotation θ ~ U[0°, 360°)
- Gaussian noise σ = 0.01–0.05, inversely scaled with SNR
- Circular temporal shift of 1–5 samples when SNR ≤ 0 dB

### Improvement B — SNR-Weighted CE Loss
- Per-sample weight: `w(snr) = 1 / (1 + exp(snr_dB / T))`, default T = 5.0
- Sweep T ∈ {3, 5, 10}

### Improvement C — SE Block (after MSM, before FFM)
- Architecture: GAP → FC(C → C/16) → ReLU → FC(C/16 → C) → Sigmoid
- Reduction ratio r = 16

### Training Config
- Optimizer: Adam, weight_decay = 1e-4
- LR schedule: cosine annealing, 1e-3 → 1e-5
- Batch: 400 (256 if OOM), epochs: 100, patience: 20, seed: 42
- Split: stratified 60/20/20 by (class × SNR)
- L2-normalize each sample before the network

### Ablation Matrix
6 model variants: baseline, +A, +B, +C, +A+B, +A+B+C
Plus SVM-cumulants and ResNet-18 baselines.

### Metrics
Overall acc, macro F1, Cohen's Kappa, low-SNR acc (≤ 0 dB), high-SNR acc (≥ 10 dB), per-class acc, confusion matrix, acc-vs-SNR curve.

---

## Key Files

| File | Role |
|---|---|
| `models/model.py` | AMC_Net definition (ACM → MSM → Conv_stem → FFM → classifier) |
| `data_loader/data_loader.py` | Load pkl, split, create DataLoaders |
| `util/training.py` | `Trainer` class (train/val loop) |
| `util/config.py` | `Config` reads `config/*.yml`, merges CLI args |
| `util/evaluation.py` | `Run_Eval` — per-SNR accuracy, F1, Kappa |
| `main.py` | Entry point |
| `config/2016.10a.yml` | Hyperparameters for this dataset |
