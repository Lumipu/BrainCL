# coding=utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceLevelContrastiveLoss(nn.Module):
    """Contrastive loss between the original and masked views."""
    def __init__(self, temperature: float = 0.7):
        super().__init__()
        self.temperature = temperature

    def forward(self, fs, fm):
        # Normalize features to unit vectors.
        fs = F.normalize(fs, dim=1)
        fm = F.normalize(fm, dim=1)

        sim_ms = torch.matmul(fm, fs.T) / self.temperature  # [B, B]
        sim_sm = torch.matmul(fs, fm.T) / self.temperature  # [B, B]

        pos_ms = sim_ms.diag()  # [B]
        pos_sm = sim_sm.diag()  # [B]

        denom_ms = torch.logsumexp(sim_ms, dim=1)  # [B]
        denom_sm = torch.logsumexp(sim_sm, dim=1)  # [B]

        loss_ms = -pos_ms + denom_ms
        loss_sm = -pos_sm + denom_sm

        loss = (loss_ms + loss_sm).mean() * 0.5
        return loss


class ClassLevelContrastiveLoss(nn.Module):
    """class-level contrastive loss"""
    def __init__(self, momentum=0.9, feature_dim=400, warm_start_batches=5):
        super().__init__()
        self.momentum = momentum
        self.temperature = 0.7
        self.register_buffer('prototypes', torch.zeros(2, feature_dim))  # Binary classification
        self.initialized = False
        self.warm_start_batches = warm_start_batches
        self._warm_fs = []
        self._warm_y = []

    def forward(self, fs: torch.Tensor, fm: torch.Tensor, y: torch.Tensor, logits: torch.Tensor):
        y = y.long()
        y_pred = (logits.sigmoid() > 0.5).long()

        # === Warm Start ===
        if not self.initialized:
            self._warm_fs.append(fs.detach())
            self._warm_y.append(y.detach())
            if len(self._warm_fs) >= self.warm_start_batches:
                all_fs = torch.cat(self._warm_fs, dim=0)
                all_y = torch.cat(self._warm_y, dim=0)
                self._warm_start_prototypes(all_fs, all_y)
                self.initialized = True
            return torch.tensor(0.0, device=fs.device)

        feats = torch.cat([fs, fm], dim=0)  # (2B, D)
        feats = F.normalize(feats, dim=1)
        labels = y.repeat(2)
        prototypes = F.normalize(self.prototypes, dim=1)
        logits = (feats @ prototypes.T) / self.temperature
        loss = F.cross_entropy(logits, labels)

        # === EMA Update Prototypes ===
        self._update_prototypes(fs, fm, y, y_pred)
        return loss


    @torch.no_grad()
    def _warm_start_prototypes(self, fs: torch.Tensor, y: torch.Tensor):
        for c in range(2):
            mask = (y == c)
            if torch.any(mask):
                self.prototypes[c] = fs[mask].mean(dim=0)

    @torch.no_grad()
    def _update_prototypes(self, fs: torch.Tensor, fm: torch.tensor, y: torch.Tensor, y_pred: torch.Tensor):
        fs_norm = F.normalize(fs, dim=1)
        fm_norm = F.normalize(fm, dim=1)
        cos_sim = (fs_norm * fm_norm).sum(dim=1)
        alpha = (1.0 + cos_sim) / 2.0
        for c in range(2):
            mask = (y == c) & (y_pred == c)
            if torch.any(mask):
                alpha_c = alpha[mask]
                fs_c = fs[mask]
                weight = alpha_c / (alpha_c.sum() + 1e-9)
                class_mean = (weight.unsqueeze(1) * fs_c).sum(dim=0)
                self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * class_mean
