# coding=utf-8


import torch
from sklearn.metrics import roc_auc_score


class AverageLossMeter(object):
    """Track the mean loss."""
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update_with_weight(self, val: float, n: int):
        """Accumulate the weighted value and sample count."""
        self.sum += val * n
        self.count += n

    def reset(self):
        """Reset the accumulated statistics."""
        self.__init__()

    @property
    def avg(self) -> float:
        """Return the mean, or -1 if no values have been added."""
        if self.count == 0:
            return -1
        return self.sum / self.count


class BinaryMetricsAggregator(object):
    """Aggregate binary classification metrics."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.total_correct = 0
        self.total_samples = 0

        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        self.all_logits = []
        self.all_labels = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Update statistics for one batch.
        Args:
            logits: Raw model logits before sigmoid.
            labels: Ground-truth labels as a float tensor containing 0 or 1.
        """
        logits = logits.detach().cpu()
        labels = labels.detach().cpu().int()
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).int()

        # Accuracy
        correct = (preds == labels).sum().item()
        self.total_correct += correct
        self.total_samples += labels.size(0)

        # Confusion matrix components
        self.tp += ((preds == 1) & (labels == 1)).sum().item()
        self.fp += ((preds == 1) & (labels == 0)).sum().item()
        self.tn += ((preds == 0) & (labels == 0)).sum().item()
        self.fn += ((preds == 0) & (labels == 1)).sum().item()

        # For AUC
        self.all_logits.append(logits)
        self.all_labels.append(labels)

    def compute(self) -> dict:
        """
        Compute all accumulated metrics.
        Returns:
            dict: Contains ACC, AUC, SEN, and SPE.
        """
        if self.total_samples == 0:
            raise ValueError("No data has been added. Call update() first.")

        acc = self.total_correct / self.total_samples
        sen = self.tp / (self.tp + self.fn + 1e-8)
        spe = self.tn / (self.tn + self.fp + 1e-8)

        # AUC
        try:
            logits = torch.cat(self.all_logits)
            labels = torch.cat(self.all_labels)
            probs = torch.sigmoid(logits)
            auc = roc_auc_score(labels.numpy(), probs.numpy())
        except ValueError:
            auc = float('nan')

        return {
            'ACC': acc,
            'AUC': auc,
            'SEN': sen,
            'SPE': spe
        }

    def reset(self):
        self.__init__()
