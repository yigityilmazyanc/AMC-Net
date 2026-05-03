"""
One-shot converter: txt mirror → RML2016.10a_dict.pkl

Keys:  (mod_bytes, snr_int)  e.g. (b'8PSK', -10)
Values: np.float32 array of shape (1000, 2, 128)
         axis-0: signals, axis-1: [I, Q], axis-2: samples
"""
import os
import pickle
import numpy as np
from pathlib import Path

TXT_DIR = Path('./data/temp_dataset/2016.10a')
OUTPUT  = Path('./data/RML2016.10a_dict.pkl')

dataset = {}

txt_files = sorted(TXT_DIR.glob('*.txt'))
assert len(txt_files) == 220, f"Expected 220 files, found {len(txt_files)}"

for txt_file in txt_files:
    stem = txt_file.stem                    # e.g. "8PSK -10"
    mod_str, snr_str = stem.rsplit(' ', 1)  # handles "AM-DSB -10" correctly
    mod = mod_str.encode('utf-8')
    snr = int(snr_str)

    signals = []
    with open(txt_file, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            assert len(tokens) == 128, \
                f"{txt_file.name}: expected 128 tokens, got {len(tokens)}"
            iq = [complex(t) for t in tokens]
            signals.append([[c.real for c in iq],   # I row
                             [c.imag for c in iq]])  # Q row

    arr = np.array(signals, dtype=np.float32)  # (1000, 2, 128)
    assert arr.shape == (1000, 2, 128), \
        f"{txt_file.name}: bad shape {arr.shape}"
    dataset[(mod, snr)] = arr

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, 'wb') as fh:
    pickle.dump(dataset, fh, protocol=4)

print(f"Wrote {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.1f} MB)")
print(f"Keys : {len(dataset)}  (expect 220)")

# --- quick verification ---
import pickle as _pkl
d = _pkl.load(open(OUTPUT, 'rb'), encoding='bytes')
mods = sorted(set(k[0] for k in d))
snrs = sorted(set(k[1] for k in d))
print(f"Mods ({len(mods)}): {[m.decode() for m in mods]}")
print(f"SNRs ({len(snrs)}): {snrs}")

sample = d[(b'8PSK', -10)]
print(f"Sample shape : {sample.shape}  dtype={sample.dtype}")
print(f"Sample I[0]  : min={sample[0,0].min():.4f}  max={sample[0,0].max():.4f}")
print(f"Sample Q[0]  : min={sample[0,1].min():.4f}  max={sample[0,1].max():.4f}")
print("Conversion OK")
