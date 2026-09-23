# coding=utf-8


import gc
import torch
import torch.nn as nn
from log.mylogger import MyLog
from network.MoTFormer import MoTFormer
from utils.losses import InstanceLevelContrastiveLoss, ClassLevelContrastiveLoss
from utils.meter import AverageLossMeter, BinaryMetricsAggregator


class BrainCLModel(object):
    def __init__(self, lr, weight_decay, epoch_num, repeated_cv_dataloader):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch_num = epoch_num
        self.repeated_cv_dataloader = repeated_cv_dataloader
        self.n_splits = self.repeated_cv_dataloader.n_splits
        self.n_repeats = self.repeated_cv_dataloader.n_repeats
        self.lr_min = lr * 0.1
        self.device = 'cuda'
        self.cont_start = 100

    @staticmethod
    def release_gpu_resources(*objs):
        """Release GPU and host memory resources."""
        for obj in objs:
            del obj
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    def train(self):
        """Train the model and report train, validation, and test metrics."""
        for repeat_idx in range(self.n_repeats):
            for fold_idx in range(self.n_splits):
                train_dataloader, val_dataloader, test_dataloader = (
                    self.repeated_cv_dataloader.get_cur_dataloader(repeat_idx, fold_idx)
                )
                net = MoTFormer().to(self.device)
                cls_criterion = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
                instance_cont_criterion = InstanceLevelContrastiveLoss().to(self.device)
                class_cont_criterion = ClassLevelContrastiveLoss().to(self.device)
                optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                train_loss_meter = AverageLossMeter()
                train_metrics_meter = BinaryMetricsAggregator()
                for cur_epoch in range(self.epoch_num):
                    train_loss_meter.reset()
                    train_metrics_meter.reset()

                    net.train()
                    # Train
                    for data in train_dataloader:
                        fcn, label = data
                        fcn = fcn.to(self.device)
                        label = label.to(self.device)
                        output, masked_output, fs, fm = net(fcn, cur_epoch >= self.cont_start,
                                                        cur_epoch - self.cont_start, self.epoch_num - self.cont_start)
                        if cur_epoch < self.cont_start:
                            loss = cls_criterion(output, label)
                        else:
                            cls_loss = 0.5 * cls_criterion(output, label) + 0.5 * cls_criterion(masked_output, label)
                            instance_cont_loss = instance_cont_criterion(fs, fm)
                            class_cont_loss = class_cont_criterion(fs, fm, label, output)
                            loss = cls_loss + 0.6 * instance_cont_loss + 0.4 * class_cont_loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss_meter.update_with_weight(loss.item(), label.size(0))
                        train_metrics_meter.update(output, label)

                    val_loss, val_metrics = self.evaluate(net, val_dataloader, cls_criterion, cur_epoch)
                    test_loss, test_metrics = self.evaluate(net, test_dataloader, cls_criterion, cur_epoch)
                    MyLog.print(f'**Rep{repeat_idx}-Fold{fold_idx}**'.center(118, ' '))
                    MyLog.print(f'Epoch{cur_epoch:03d}'.center(118, '-'))
                    self.print_metrics('Train', train_loss_meter.avg, train_metrics_meter.compute())
                    self.print_metrics('Val', val_loss, val_metrics)
                    self.print_metrics('Test', test_loss, test_metrics)
                BrainCLModel.release_gpu_resources(net, train_dataloader, val_dataloader, test_dataloader,
                                                   cls_criterion, instance_cont_criterion,
                                                   class_cont_criterion, optimizer)

    def evaluate(self, net, dataloader, cls_criterion, cur_epoch):
        """Evaluate validation and test data."""
        loss_meter = AverageLossMeter()
        metrics_meter = BinaryMetricsAggregator()
        net.eval()
        with torch.no_grad():
            for fcn, label in dataloader:
                fcn = fcn.to(self.device)
                label = label.to(self.device)
                output, _, _, _ = net(fcn, cur_epoch >= self.cont_start,
                                      cur_epoch - self.cont_start, self.epoch_num - self.cont_start)
                loss = cls_criterion(output, label)
                loss_meter.update_with_weight(loss.item(), label.size(0))
                metrics_meter.update(output, label)
        return loss_meter.avg, metrics_meter.compute()

    @staticmethod
    def print_metrics(split, loss, metrics):
        MyLog.print(
            f'{split:<5} Loss:{loss:07.4f}      {split:<5} AUC:{metrics["AUC"] * 100:06.2f}(%)      '
            f'{split:<5} ACC:{metrics["ACC"] * 100:06.2f}(%)      '
            f'{split:<5} SEN:{metrics["SEN"] * 100:06.2f}(%)      '
            f'{split:<5} SPE:{metrics["SPE"] * 100:06.2f}(%)'
        )
