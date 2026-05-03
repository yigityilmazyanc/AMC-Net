import math
import numpy as np
import torch

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from util.visualize import Draw_Confmat, Snr_Acc_Plot


def _tta_predict(model, sample, K=8):
    """Average logits over K uniformly-spaced phase rotations."""
    angles = [2 * math.pi * k / K for k in range(K)]
    logits = []
    with torch.no_grad():
        for theta in angles:
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            I = sample[:, 0:1, :]
            Q = sample[:, 1:2, :]
            rot = torch.cat([I * cos_t - Q * sin_t,
                             I * sin_t + Q * cos_t], dim=1)
            logits.append(model(rot))
    return torch.stack(logits).mean(0)


def Run_Eval(model,
             sig_test,
             lab_test,
             SNRs,
             test_idx,
             cfg,
             logger,
             tta=False,
             tta_k=8):

    model.eval()

    snrs = list(np.unique(SNRs))
    mods = list(cfg.classes.keys())

    Confmat_Set = np.zeros((len(snrs), len(mods), len(mods)), dtype=int)
    Accuracy_list = np.zeros(len(snrs), dtype=float)

    pre_lab_all = []
    label_all = []

    for snr_i, snr in enumerate(snrs):
        test_SNRs = map(lambda x: SNRs[x], test_idx)
        test_SNRs = list(test_SNRs)
        test_SNRs = np.array(test_SNRs).squeeze()
        test_sig_i = sig_test[np.where(np.array(test_SNRs) == snr)]
        test_lab_i = lab_test[np.where(np.array(test_SNRs) == snr)]
        Sample = torch.chunk(test_sig_i, cfg.test_batch_size, dim=0)
        Label = torch.chunk(test_lab_i, cfg.test_batch_size, dim=0)
        pred_i = []
        label_i = []
        for (sample, label) in zip(Sample, Label):
            sample = sample.to(cfg.device)
            if tta:
                logit = _tta_predict(model, sample, K=tta_k)
            else:
                with torch.no_grad():
                    logit = model(sample)
            pre_lab = torch.argmax(logit, 1).cpu()
            pred_i.append(pre_lab)
            label_i.append(label)
        pred_i = np.concatenate(pred_i)
        label_i = np.concatenate(label_i)

        pre_lab_all.append(pred_i)
        label_all.append(label_i)

        Confmat_Set[snr_i, :, :] = confusion_matrix(label_i, pred_i)
        Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)

    pre_lab_all = np.concatenate(pre_lab_all)
    label_all = np.concatenate(label_all)

    F1_score = f1_score(label_all, pre_lab_all, average='macro')
    kappa = cohen_kappa_score(label_all, pre_lab_all)
    acc = np.mean(Accuracy_list)

    logger.info(f'overall accuracy is: {acc}')
    logger.info(f'macro F1-score is: {F1_score}')
    logger.info(f'kappa coefficient is: {kappa}')

    if cfg.Draw_Confmat is True:
        Draw_Confmat(Confmat_Set, snrs, cfg)
    if cfg.Draw_Acc_Curve is True:
        Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg)


