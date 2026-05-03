"""
Train ResNet-18 baseline on RML2016.10a.
Reuses the same Trainer, DataLoader, Evaluator as the AMC-Net experiments
so the comparison is apples-to-apples.

Usage:
    python baselines/train_resnet.py --seed 42
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from baselines.resnet1d import ResNet1D18
from data_loader.data_loader import Load_Dataset, Dataset_Split, Create_Data_Loader
from util.config import Config, merge_args2cfg
from util.evaluation import Run_Eval
from util.logger import create_logger
from util.training import Trainer
from util.utils import fix_seed, log_exp_settings
from util.visualize import save_training_process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',        type=int,  default=42)
    parser.add_argument('--device',      type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int,  default=4)
    parser.add_argument('--Draw_Confmat',   type=bool, default=True)
    parser.add_argument('--Draw_Acc_Curve', type=bool, default=True)
    parser.add_argument('--exp_tag',     type=str,  default='resnet18')
    args = parser.parse_args()

    fix_seed(args.seed)

    cfg = Config('2016.10a', train=True, exp_tag=args.exp_tag)
    cfg = merge_args2cfg(cfg, vars(args))
    cfg.mode = 'train'

    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    log_exp_settings(logger, cfg)

    model = ResNet1D18(num_classes=cfg.num_classes).to(cfg.device)
    logger.info('>>> ResNet-18-1D total params: {:.2f}M'.format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    Signals, Labels, SNRs, snrs, mods = Load_Dataset(cfg.dataset, logger)
    train_set, test_set, val_set, test_idx = Dataset_Split(
        Signals, Labels, snrs, mods, logger)
    Signals_test, Labels_test = test_set

    train_loader, val_loader = Create_Data_Loader(train_set, val_set, cfg, logger)
    trainer = Trainer(model, train_loader, val_loader, cfg, logger)
    trainer.loop()

    save_training_process(trainer.epochs_stats, cfg)

    # Trainer saves as {dataset}_AMC_Net.pkl regardless of model class
    save_name = cfg.dataset + '_AMC_Net.pkl'
    model.load_state_dict(torch.load(os.path.join(cfg.model_dir, save_name)))
    Run_Eval(model, Signals_test, Labels_test, SNRs, test_idx, cfg, logger)


if __name__ == '__main__':
    main()
