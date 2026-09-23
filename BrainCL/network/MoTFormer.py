# coding=utf-8


import copy
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from utils.masker import SalienceMasker


class NodeWiseHopAttention(nn.Module):
    def __init__(self, feature_dim=200, hidden_dim=128):
        super().__init__()

        self.q_linear = nn.Linear(feature_dim, hidden_dim)
        self.k_linear = nn.Linear(feature_dim, hidden_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        self.o_linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, hop_transition, fcn):
        q = self.q_linear(fcn)
        q = q.unsqueeze(2)

        hop_transition = hop_transition.permute(0, 2, 1, 3)
        k = self.k_linear(hop_transition)
        v = self.v_linear(hop_transition)

        attn_score = (q * k).sum(dim=-1) / (q.size(-1) ** 0.5)
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = attn_prob.unsqueeze(-1)
        output =  (attn_prob * v).sum(dim=-2)
        output = self.o_linear(output)

        return output, attn_prob


class TopologicalEmbedding(nn.Module):
    """Embed topology at multiple hop distances into the FCN."""
    def __init__(self, walk_length=32):
        super(TopologicalEmbedding, self).__init__()
        self.walk_length = walk_length
        self.hop_attention = NodeWiseHopAttention()
        self.thresh = 0.3
        self.lambda_rw = 1.0 # Self-loop

    @staticmethod
    def calcu_communication_strength(fcn):
        device = fcn.device
        dtype = fcn.dtype
        num_nodes = fcn.size(0)
        eye = torch.eye(num_nodes, device=device, dtype=dtype)

        return fcn * (1.0 - eye)

    def get_sprw_trans(self, fcn):
        device = fcn.device
        num_nodes = fcn.shape[0]
        dtype = fcn.dtype

        eye = torch.eye(num_nodes, device=device, dtype=dtype)
        C = self.calcu_communication_strength(fcn)
        adj = (fcn > self.thresh).to(dtype)
        W = adj * C + self.lambda_rw * eye

        row_sum = W.sum(dim=1, keepdim=True)  # (N, 1)
        Pi = W.clone()

        # Normalize rows with positive total weight.
        mask = row_sum.squeeze(-1) > 0
        Pi[mask] = W[mask] / row_sum[mask]
        if (~mask).any():
            Pi[~mask] = eye[~mask]
        tran_list = [Pi]
        out = Pi
        if self.walk_length > 1:
            for j in range(1, self.walk_length):
                out = out @ Pi
                tran_list.append(out)
        trans = torch.stack(tran_list, dim=0).to(device)
        return trans

    def forward(self, fcn_batch):
        """Compute long-range embeddings for a batch of brain networks."""
        batch_size = fcn_batch.shape[0]
        trans_batch = []
        for i in range(batch_size):
            fcn = fcn_batch[i].squeeze()
            trans = self.get_sprw_trans(fcn)
            trans_batch.append(trans)
        trans_batch = torch.stack(trans_batch)
        topological_batch, hop_attn_weights = self.hop_attention(trans_batch, fcn_batch)
        feature = fcn_batch + topological_batch
        return feature, trans_batch, hop_attn_weights


class ClusterAssignment(nn.Module):
    """Cluster nodes into supernodes."""
    def __init__(self, cluster_number: int, embedding_dimension: int, alpha: float = 1.0,
                 cluster_centers: Optional[torch.Tensor] = None, orthogonal=True, freeze_center=True,
                 project_assignment=True) -> None:
        super(ClusterAssignment, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.project_assignment = project_assignment

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers

        if orthogonal:
            orthogonal_cluster_centers = torch.zeros(self.cluster_number, self.embedding_dimension, dtype=torch.float)
            orthogonal_cluster_centers[0] = initial_cluster_centers[0]
            for i in range(1, cluster_number):
                project = 0
                for j in range(i):
                    project += self.project(
                        initial_cluster_centers[j], initial_cluster_centers[i])
                initial_cluster_centers[i] -= project
                orthogonal_cluster_centers[i] = initial_cluster_centers[i] / torch.norm(initial_cluster_centers[i], p=2)
            initial_cluster_centers = orthogonal_cluster_centers

        self.cluster_centers = nn.Parameter(
            initial_cluster_centers, requires_grad=(not freeze_center))

    @staticmethod
    def project(u, v):
        return (torch.dot(u, v)/torch.dot(u, u))*u

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.project_assignment:
            assignment = batch@self.cluster_centers.T
            assignment = torch.pow(assignment, 2)
            norm = torch.norm(self.cluster_centers, p=2, dim=-1)
            soft_assign = assignment/norm
            return F.softmax(soft_assign, dim=-1)
        else:
            norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
            numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
            power = float(self.alpha + 1) / 2
            numerator = numerator ** power
            return numerator / torch.sum(numerator, dim=1, keepdim=True)


class DEC(nn.Module):
    """Deep embedded clustering."""
    def __init__(self, cluster_number: int, hidden_dimension: int, encoder: torch.nn.Module, alpha: float=1.0,
        orthogonal=True, freeze_center=True, project_assignment=True):
        super(DEC, self).__init__()

        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha, orthogonal=orthogonal, freeze_center=freeze_center,
            project_assignment=project_assignment)

    def forward(self, batch: torch.Tensor):
        node_num = batch.size(1)
        batch_size = batch.size(0)

        flattened_batch = batch.view(batch_size, -1)
        encoded = self.encoder(flattened_batch)
        encoded = encoded.view(batch_size * node_num, -1)
        assignment = self.assignment(encoded)
        assignment = assignment.view(batch_size, node_num, -1)
        encoded = encoded.view(batch_size, node_num, -1)
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)
        return node_repr


class ClusterReadout(nn.Module):
    """Readout based on orthogonal clustering."""
    def __init__(self, input_node_num=200, output_node_num=50, input_feature_dim=200, output_feature_dim=8):
        super(ClusterReadout, self).__init__()

        hidden_dim = 32

        encoder = nn.Sequential(
            nn.Linear(input_node_num * input_feature_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_node_num * input_feature_dim)
        )

        self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_dim, encoder=encoder)
        self.dim_reduction = nn.Sequential(
            nn.Linear(input_feature_dim, output_feature_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_dec= self.dec(x)
        x_reduction = self.dim_reduction(x_dec)
        x_out = x_reduction.reshape((batch_size, -1))
        return x_out


class AttentionReadout(nn.Module):
    """Attention readout."""
    def __init__(self, feature_dim, readout_dim=1):
        super(AttentionReadout, self).__init__()
        self.readout_dim = readout_dim
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        scores = self.attention(x)
        weights = F.softmax(scores, dim=self.readout_dim)
        x_out = (weights * x).sum(dim=self.readout_dim)
        return x_out


class MeanReadout(nn.Module):
    """Mean readout."""
    def __init__(self, readout_dim=1):
        super(MeanReadout, self).__init__()
        self.readout_dim = readout_dim

    def forward(self, x):
        return x.mean(dim=self.readout_dim)


class MaxReadout(nn.Module):
    """Max readout."""
    def __init__(self, readout_dim=1):
        super(MaxReadout, self).__init__()
        self.readout_dim = readout_dim

    def forward(self, x):
        return x.max(dim=self.readout_dim).values


class MeanMaxReadout(nn.Module):
    """Mean-max readout."""
    def __init__(self, readout_dim=1):
        super(MeanMaxReadout, self).__init__()
        self.readout_dim = readout_dim

    def forward(self, x):
        x_mean = x.mean(dim=self.readout_dim)
        x_max = x.max(dim=self.readout_dim).values
        return x_mean + x_max


class LearnableMeanMaxReadout(nn.Module):
    """Learnable mean-max readout."""
    def __init__(self, feature_dim, readout_dim=1):
        super(LearnableMeanMaxReadout, self).__init__()
        self.readout_dim = readout_dim
        self.fc = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, x):
        x_mean = x.mean(dim=self.readout_dim)
        x_max = x.max(dim=self.readout_dim).values
        combined = torch.cat([x_mean, x_max], dim=-1)
        return self.fc(combined)


class InterpretableTransformerEncoderLayer(TransformerEncoderLayer):
    """Return attention weights averaged across heads."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first) -> None:
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                         batch_first=batch_first)
        self.attention_weights: Optional[torch.Tensor] = None

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x, weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
            is_causal=is_causal,
        )
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self.attention_weights


class MoTFormer(nn.Module):
    """Multi-Order Topology-Aware Transformer"""
    def __init__(self, roi_num=200, walk_length=16, num_head=4, dim_ffn=1024):
        super(MoTFormer, self).__init__()

        feature_dim = roi_num
        readout_node_num = 50
        readout_node_feature_dim = 8
        readout_dim = readout_node_num * readout_node_feature_dim  # Readout dimension: 50 x 8 = 400

        self.embedding = TopologicalEmbedding(walk_length)
        self.encoder = MoTFormer.build_encoder(feature_dim, num_head,dim_ffn)
        self.readout = ClusterReadout(roi_num, readout_node_num, feature_dim, readout_node_feature_dim)
        # self.projector = nn.Sequential(
        #     nn.Linear(readout_dim, readout_dim),
        #     nn.ReLU(),
        #     nn.Linear(readout_dim, readout_dim)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self.masker = SalienceMasker(feature_dim)

    @staticmethod
    def build_encoder(feature_dim, num_head, dim_ffn, dropout=0.2):
        """Build the encoder."""
        encoder_layer = InterpretableTransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_head,
            dim_feedforward=dim_ffn,
            dropout=dropout,
            batch_first=True,
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        return encoder

    def forward(self, x, mask_start, cur_epoch, max_epoch):
        x_embedding, trans_batch, hop_attn_weights = self.embedding(x)
        x_encoding = self.encoder(x_embedding)
        x_readout = self.readout(x_encoding)
        out = self.classifier(x_readout).squeeze(-1)

        if self.training and mask_start:
            # fs = self.projector(x_readout)
            fs = x_readout
            attention_weight = self.get_attention_weights()[-1]
            x_masked = self.masker(attention_weight, trans_batch, hop_attn_weights, x, cur_epoch, max_epoch)
            x_masked_embedding, _, _ = self.embedding(x_masked)
            x_masked_encoding = self.encoder(x_masked_embedding)
            x_masked_readout = self.readout(x_masked_encoding)
            # fm = self.projector(x_masked_readout)
            fm = x_masked_readout
            masked_out = self.classifier(x_masked_readout).squeeze(-1)
            return out, masked_out, fs, fm
        else:
            return out, None, None, None

    def get_attention_weights(self):
        """Return attention weights from all encoder layers."""
        attention_weights = []
        for layer in self.encoder.layers:
            attention_weights.append(layer.attention_weights)
        return attention_weights
