"""Evaluation of the predictions."""

import torch
import torchmetrics
from torch import nn


class Evaluation(nn.Module):
    """Evaluation module."""

    def __init__(self, num_classes=2, logger=None):
        """init."""
        super().__init__()
        self.logger = logger
        self.num_classes = num_classes
        task = "binary" if num_classes == 2 else "multilabel"
        average = None if num_classes == 2 else "macro"
        metric_params = ({
            "task": task,
            "num_classes": num_classes,
            "average": average,
            "top_k": 1,
        } if num_classes == 2 else {
            "task": task,
            "num_labels": num_classes,  # 多标签任务用 num_labels
            "average": average,
            "top_k": 1,
            })
        
        metric_params_2 = ({
            "task": "multiclass",
            "num_classes": num_classes,
            "average": "weighted",
            "top_k": 1,
        } if num_classes == 2 else {
            "task": task,
            "num_labels": num_classes,  # 多标签任务用 num_labels
            "average": "weighted",
            "top_k": 1,
            })
        
        metric_params_3 = ({
            "task": "multiclass",
            "num_classes": num_classes,
            "average": "macro",
            "top_k": 1,
        } if num_classes == 2 else {
            "task": task,
            "num_labels": num_classes,  # 多标签任务用 num_labels
            "average": "macro",
            "top_k": 1,
            })
        # print(num_classes, type(num_classes))
        self.accuracy = torchmetrics.Accuracy(**metric_params)
        if num_classes == 2:
            self.roc = torchmetrics.ROC(task=task, num_classes=num_classes)
            self.auc = torchmetrics.AUROC(task=task, num_classes=num_classes)
            self.recall = torchmetrics.Recall(**{"task": task})
            self.specifity = torchmetrics.Specificity(**{"task": task})
        else:
            self.roc = torchmetrics.ROC(task=task, num_labels=num_classes)
            self.auc = torchmetrics.AUROC(task=task, num_labels=num_classes)
            self.recall = torchmetrics.Recall(task=task, num_labels=num_classes)
            self.specifity = torchmetrics.Specificity(task=task, num_labels=num_classes)

        self.f1 = torchmetrics.F1Score(**metric_params)
        self.f1_weighted = torchmetrics.F1Score(**metric_params_2)
        
        self.balanced_accuracy = torchmetrics.Accuracy(**metric_params_3)

    def forward(self, y_pred, y_true, prefix):
        """forward method.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions.
        y_true : torch.Tensor
            True labels.
        prefix : str
            Prefix for the metrics.

        Returns
        -------
        metrics : dict
            Dictionary with the metrics.
        """
        metrics = {}
        auc_score = self.auc(y_pred, y_true)
        metrics[f"{prefix}auc"] = auc_score

        if self.num_classes == 2:   
            y_pred = torch.argmax(y_pred, dim=1)
            y_true = torch.argmax(y_true, dim=1)

        metrics["acc"] = self.accuracy(y_pred, y_true)
        metrics[f"{prefix}test_f1"] = self.f1(y_pred, y_true)
        metrics[f"{prefix}recall"] = self.recall(y_pred, y_true)
        metrics[f"{prefix}specifity"] = self.specifity(y_pred, y_true)
        metrics[f"{prefix}balanced_accuracy"] = self.balanced_accuracy(y_pred, y_true)
        metrics[f"{prefix}f1_weighted"] = self.f1_weighted(y_pred, y_true)
        return metrics
