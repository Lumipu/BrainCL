# coding=utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F


class SalienceMasker(nn.Module):
    def __init__(self, feature_dim=200, mask_ratio=0.3, temperature=0.7):
        """Mask nodes using attention guidance."""
        super().__init__()
        self.mask_ratio = mask_ratio
        self.alpha = 0.5
        self.temperature = temperature
        self.mask_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.2)
        self.warm_up_ratio = 0.3


    def forward(self, semantic_attn_weights, trans_batch, hop_attn_weights, features, cur_epoch, max_epoch):
        batch_size, num_node, _ = features.shape
        device = features.device

        T0 = int(max_epoch * self.warm_up_ratio)
        T0 = max(T0, 1)
        t = cur_epoch
        if t <= T0:
            k = self.mask_ratio * num_node * ((t / T0) ** 0.5)
        else:
            k = self.mask_ratio * num_node
        k = int(round(k))

        # Compute semantic scores.
        semantic_scores = F.softmax(semantic_attn_weights.sum(dim=1), dim=-1)  # (B, N)

        # Compute topological scores.
        hop_attn_weights = hop_attn_weights.squeeze(-1)  # (B, N, H)

        weighted_sum = torch.einsum(
            "bih,bhij->bj",
            hop_attn_weights,
            trans_batch
        )  # (B, N)

        topo_scores = F.softmax(weighted_sum, dim=-1)

        salience_scores = self.alpha * semantic_scores + (1.0 - self.alpha) * topo_scores
        salience_probs = F.softmax(salience_scores / self.temperature, dim=-1)
        # Gumbel-Max sampling
        u = torch.rand(batch_size, num_node, device=device)
        gumbel_noise = -torch.log(-torch.log(u + 1e-9) + 1e-9)
        gumbel_sample = torch.log(salience_probs + 1e-9) + gumbel_noise
        mask_indices = gumbel_sample.topk(k, dim=-1).indices

        # Mask nodes.
        features_masked = features.clone()
        batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1).expand(-1, k)
        features_masked[batch_indices, mask_indices] = torch.tanh(self.mask_token)

        return features_masked
