"""SNR-conditioned signal augmentation (training only). IMPROVEMENT A"""
import math
import torch


def snr_augment(sig, snr):
    """
    sig : (B, 2, 128) float32 on device
    snr : (B,)        float32 on device  [dB]

    Noise is relative to per-sample signal RMS (5-15%), not absolute.
    This keeps the added noise physically plausible across all SNR levels.
    Amplitude scaling removed — it destroys amplitude-based modulations.
    """
    B      = sig.size(0)
    device = sig.device

    # 1. Phase rotation θ ~ U[0, 2π) — IQ rotation, leaves modulation valid
    theta = torch.rand(B, device=device) * 2 * math.pi
    cos_t = torch.cos(theta).view(B, 1, 1)
    sin_t = torch.sin(theta).view(B, 1, 1)
    I, Q  = sig[:, 0:1, :], sig[:, 1:2, :]
    sig   = torch.cat([I * cos_t - Q * sin_t,
                       I * sin_t + Q * cos_t], dim=1)    # (B, 2, 128)

    # 2. AWGN: noise_frac is 5–15% of per-sample RMS, inversely scaled with SNR
    sig_rms   = sig.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()   # (B, 1, 1)
    noise_frac = (0.15 - 0.10 * (snr + 20.0) / 38.0).clamp(0.05, 0.15)
    sig        = sig + torch.randn_like(sig) * sig_rms * noise_frac.view(B, 1, 1)

    # 3. Circular temporal shift 1–5 samples — only for SNR ≤ 0 dB
    low_snr = snr <= 0.0
    if low_snr.any():
        shifts = torch.randint(1, 6, (B,), device=device)
        for s in range(1, 6):
            mask = low_snr & (shifts == s)
            if mask.any():
                sig[mask] = torch.roll(sig[mask], s, dims=-1)

    return sig
