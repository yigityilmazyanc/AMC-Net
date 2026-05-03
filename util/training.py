import os.path
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from tqdm import tqdm
from torch.optim import lr_scheduler

from util.augment import snr_augment                    # IMPROVEMENT A
from util.early_stop import EarlyStopping
from util.evaluation import Run_Eval
from util.logger import AverageMeter


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, logger):
        super(Trainer, self).__init__()
        self.epochs_stats = self.val_acc_list = self.val_loss_list = None
        self.train_acc_list = self.train_loss_list = None
        self.val_acc = self.val_loss = self.train_acc = None
        self.best_monitor = self.lr_list = self.train_loss = self.t_s = None
        self.criterion = self.optimizer = self.scheduler = None
        self.early_stopping = None
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.iter = 0

    def loop(self):
        self.before_train()
        for self.iter in range(0, self.cfg.epochs):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()
            if self.early_stopping.early_stop:
                self.logger.info('Early stopping')
                break

    def before_train(self):
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(self.cfg.device)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.epochs, eta_min=1e-5)
        self.lr_list = []
        self.best_monitor = 0.0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.early_stopping = EarlyStopping(self.logger, patience=self.cfg.patience)

    def before_train_step(self):
        self.model.train()
        self.t_s = time.time()
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")
    def run_train_step(self):
        with tqdm(total=len(self.train_loader),
                  desc=f'Epoch{self.iter}/{self.cfg.epochs}',
                  postfix=dict, mininterval=0.3) as pbar:
            for step, (sig_batch, lab_batch, snr_batch) in enumerate(self.train_loader):
                sig_batch = sig_batch.to(self.cfg.device)
                lab_batch = lab_batch.to(self.cfg.device)
                snr_batch = snr_batch.to(self.cfg.device)
                sig_batch = snr_augment(sig_batch, snr_batch)  # IMPROVEMENT A

                logit = self.model(sig_batch)
                # IMPROVEMENT B: SNR-weighted CE loss
                w    = 1.0 / (1.0 + torch.exp(snr_batch / self.cfg.snr_loss_T))
                w    = w / w.mean()
                loss = (F.cross_entropy(logit, lab_batch, reduction='none',
                                        label_smoothing=0.05) * w).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pre_lab = torch.argmax(logit, 1)
                acc = torch.sum(pre_lab == lab_batch.data).double().item() / lab_batch.size(0)
                self.train_loss.update(loss.item())
                self.train_acc.update(acc)
                pbar.set_postfix(**{'train_loss': self.train_loss.avg,
                                    'train_acc': self.train_acc.avg})
                pbar.update(1)
    def after_train_step(self):
        self.lr_list.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} Train acc: {} lr: {:.5f}'.format(
                self.iter, time.time() - self.t_s,
                self.train_loss.avg, self.train_acc.avg, self.lr_list[-1]))
        self.train_loss_list.append(self.train_loss.avg)
        self.train_acc_list.append(self.train_acc.avg)

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_loss = AverageMeter()
        self.val_acc = AverageMeter()
        self.logger.info(f"Starting validation epoch {self.iter}:")

    def run_val_step(self):
        with tqdm(total=len(self.val_loader),
                  desc=f'Epoch{self.iter}/{self.cfg.epochs}',
                  postfix=dict, mininterval=0.3, colour='blue') as pbar:
            for step, (sig_batch, lab_batch) in enumerate(self.val_loader):
                with torch.no_grad():
                    sig_batch = sig_batch.to(self.cfg.device)
                    lab_batch = lab_batch.to(self.cfg.device)
                    logit = self.model(sig_batch)
                    loss = self.criterion(logit, lab_batch)
                    pre_lab = torch.argmax(logit, 1)
                    acc = torch.sum(pre_lab == lab_batch.data).double().item() / lab_batch.size(0)
                    self.val_loss.update(loss.item())
                    self.val_acc.update(acc)
                    pbar.set_postfix(**{'val_loss': self.val_loss.avg,
                                        'val_acc': self.val_acc.avg})
                    pbar.update(1)

    def after_val_step(self):
        gap = self.train_acc.avg - self.val_acc.avg
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Val Loss: {:.4f} Val acc: {:.4f} '
            '[gap train-val: {:.3f}]'.format(
                self.iter, time.time() - self.t_s,
                self.val_loss.avg, self.val_acc.avg, gap))
        if self.cfg.monitor == 'acc':
            if self.val_acc.avg >= self.best_monitor:
                self.best_monitor = self.val_acc.avg
                self.logger.info(
                    f'>>> Best val_acc: {self.best_monitor:.4f} at epoch {self.iter}')
                save_model_name = self.cfg.dataset + '_AMC_Net.pkl'
                torch.save(self.model.state_dict(),
                           os.path.join(self.cfg.model_dir, save_model_name))
        else:
            raise NotImplementedError(f'Not Implement monitor: {self.cfg.monitor}')

        self.early_stopping(-self.val_acc.avg, self.model)  # monitor val_acc
        self.scheduler.step()

        self.val_loss_list.append(self.val_loss.avg)
        self.val_acc_list.append(self.val_acc.avg)
        self.epochs_stats = pd.DataFrame(
            data={"epoch": range(self.iter + 1),
                  "lr_list": self.lr_list,
                  "train_loss": self.train_loss_list,
                  "val_loss": self.val_loss_list,
                  "train_acc": self.train_acc_list,
                  "val_acc": self.val_acc_list}
        )
