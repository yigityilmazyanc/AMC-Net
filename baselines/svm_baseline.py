"""
SVM-cumulants baseline for RML2016.10a.
Features: 8 higher-order cumulants per signal (C20, C21, C40, C41, C42).
Classifier: LinearSVC (scales to 132k samples; equivalent to SVM with linear kernel).

Usage:
    python baselines/svm_baseline.py --seed 42
"""
import argparse
import pickle
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from util.utils import fix_seed

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_cumulants(signals: np.ndarray) -> np.ndarray:
    """
    signals : (N, 2, 128)  float32  [I, Q]
    returns : (N, 8)       float64  feature matrix
    """
    I = signals[:, 0, :].astype(np.float64)   # (N, 128)
    Q = signals[:, 1, :].astype(np.float64)
    z = I + 1j * Q                             # (N, 128) complex

    # 2nd-order moments
    M20 = np.mean(z ** 2, axis=1)             # E[z²]        complex (N,)
    M21 = np.mean(np.abs(z) ** 2, axis=1)    # E[|z|²]      real   (N,)

    # 4th-order moments
    M40 = np.mean(z ** 4, axis=1)            # E[z⁴]        complex
    M41 = np.mean(np.abs(z) ** 2 * z ** 2, axis=1)  # E[|z|²z²]
    M42 = np.mean(np.abs(z) ** 4, axis=1)   # E[|z|⁴]      real

    # 4th-order cumulants (zero-mean assumed)
    C20 = M20
    C21 = M21
    C40 = M40 - 3.0 * M20 ** 2
    C41 = M41 - 3.0 * M20 * M21
    C42 = M42 - np.abs(M20) ** 2 - 2.0 * M21 ** 2

    # Pack into real feature vector: |C20|, ∠C20, C21, |C40|, ∠C40, |C41|, ∠C41, C42
    feats = np.column_stack([
        np.abs(C20),
        np.angle(C20),
        C21.real,
        np.abs(C40),
        np.angle(C40),
        np.abs(C41),
        np.angle(C41),
        C42.real,
    ])
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading (mirrors data_loader logic, pure numpy)
# ---------------------------------------------------------------------------

CLASSES = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
           b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9,
           b'AM-SSB': 10}


def load_data(pkl_path='./data/RML2016.10a_dict.pkl'):
    dataset = pickle.load(open(pkl_path, 'rb'), encoding='bytes')
    snrs = sorted(set(k[1] for k in dataset))
    mods = sorted(set(k[0] for k in dataset))

    signals, labels, snr_list = [], [], []
    for mod in mods:
        for snr in snrs:
            arr = dataset[(mod, snr)]         # (1000, 2, 128)
            signals.append(arr)
            labels.extend([CLASSES[mod]] * arr.shape[0])
            snr_list.extend([snr] * arr.shape[0])

    signals  = np.vstack(signals)             # (220000, 2, 128)
    labels   = np.array(labels, dtype=np.int64)
    snr_arr  = np.array(snr_list, dtype=np.int32)
    return signals, labels, snr_arr, snrs, mods


def stratified_split(n_total, n_groups, seed):
    """60/20/20 stratified split identical to Dataset_Split in data_loader."""
    rng = np.random.default_rng(seed)
    n_train = int(n_total * 0.6)
    n_per   = n_total // n_groups
    n_train_per = int(n_train / n_groups)

    train_idx, val_idx, test_idx = [], [], []
    for g in range(n_groups):
        lo, hi = g * n_per, (g + 1) * n_per
        pool   = np.arange(lo, hi)
        tr     = rng.choice(pool, size=n_train_per, replace=False)
        rest   = np.setdiff1d(pool, tr)
        n_test = len(rest) // 2
        te     = rng.choice(rest, size=n_test, replace=False)
        va     = np.setdiff1d(rest, te)
        train_idx.append(tr); val_idx.append(va); test_idx.append(te)

    return (np.concatenate(train_idx),
            np.concatenate(val_idx),
            np.concatenate(test_idx))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pkl',  type=str, default='./data/RML2016.10a_dict.pkl')
    args = parser.parse_args()

    fix_seed(args.seed)
    print('Loading dataset ...')
    signals, labels, snr_arr, snrs, mods = load_data(args.pkl)
    n_groups = len(mods) * len(snrs)             # 220 groups

    print('Extracting cumulant features ...')
    feats = extract_cumulants(signals)           # (220000, 8)

    train_idx, _, test_idx = stratified_split(len(signals), n_groups, args.seed)

    X_train, y_train = feats[train_idx],  labels[train_idx]
    X_test,  y_test  = feats[test_idx],   labels[test_idx]
    snr_test         = snr_arr[test_idx]

    print(f'Train: {X_train.shape}  Test: {X_test.shape}')

    print('Training LinearSVC ...')
    clf = Pipeline([('scaler', StandardScaler()),
                    ('svm',    LinearSVC(C=1.0, max_iter=2000, random_state=args.seed))])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Overall metrics
    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f'\n=== SVM-cumulants results ===')
    print(f'Overall accuracy : {acc:.4f}')
    print(f'Macro F1-score   : {f1:.4f}')
    print(f'Kappa            : {kappa:.4f}')

    # Per-SNR accuracy
    print('\nPer-SNR accuracy:')
    for snr in snrs:
        mask = snr_test == snr
        if mask.sum() == 0:
            continue
        acc_snr = accuracy_score(y_test[mask], y_pred[mask])
        print(f'  SNR {snr:+4d} dB : {acc_snr:.4f}  (n={mask.sum()})')

    # Low / high SNR summary
    low  = snr_test <= 0
    high = snr_test >= 10
    print(f'\nLow-SNR  (≤0 dB)  acc: {accuracy_score(y_test[low],  y_pred[low]):.4f}')
    print(f'High-SNR (≥10 dB) acc: {accuracy_score(y_test[high], y_pred[high]):.4f}')


if __name__ == '__main__':
    main()
