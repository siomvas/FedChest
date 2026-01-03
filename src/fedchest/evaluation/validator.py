"""
Validation and metrics for multi-label chest X-ray classification.

Computes AUROC, F1, accuracy, sensitivity, specificity with optimal thresholds.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


with open(Path(__file__).parent.parent.parent / "configs" / "preprocessing.yaml", "r") as f:
    PREPROCESSING_CONFIGS = yaml.safe_load(f)
CLASS_NAMES = PREPROCESSING_CONFIGS["common_diseases"]


class Validator:
    """
    Validator for multi-label chest X-ray classification.
    
    Computes comprehensive metrics including per-class AUROC,
    optimal thresholds via Youden's J statistic, and macro-averaged
    F1, accuracy, sensitivity, specificity, and precision.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Args:
            class_names: List of class names for metrics logging.
                        Defaults to standard FedChest classes.
        """
        self.class_names = class_names or CLASS_NAMES
        
        # Lazy import to avoid requiring monai for basic usage
        try:
            from monai.transforms import Activations
            self.transform_post = Activations(sigmoid=True)
        except ImportError:
            logger.warning("MONAI not installed, using torch.sigmoid instead")
            self.transform_post = None
    
    def _apply_sigmoid(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid activation to logits."""
        if self.transform_post is not None:
            return self.transform_post(logits)
        return torch.sigmoid(logits)
    
    def run(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: str = "cuda:0",
    ) -> Dict[str, Any]:
        """
        Run validation and compute metrics.
        
        Args:
            model: PyTorch model to evaluate
            data_loader: DataLoader with test/validation data
            device: Device to run inference on
        
        Returns:
            Dict containing:
                - auroc_{class}: Per-class AUROC
                - mean_auroc: Mean AUROC across classes
                - macro_f1_score: Macro-averaged F1
                - macro_accuracy: Macro-averaged accuracy
                - macro_specificity: Macro-averaged specificity
                - macro_sensitivity: Macro-averaged sensitivity (recall)
                - macro_precision: Macro-averaged precision
                - optimal_thresholds: List of optimal thresholds per class
        """
        model.eval()
        metrics: Dict[str, Any] = {}
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
                for batch in tqdm(
                    data_loader,
                    desc="Validation",
                    dynamic_ncols=True,
                ):
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    
                    logits = model(images)
                    preds = self._apply_sigmoid(logits)
                    
                    all_labels.append(labels)
                    all_preds.append(preds)
        
        # Concatenate all batches
        y = torch.cat(all_labels)
        y_pred = torch.cat(all_preds)
        
        # Check for NaN values
        if torch.isnan(y).any() or torch.isnan(y_pred).any():
            logger.error("NaN values detected in labels or predictions")
            return self._get_zero_metrics()
        
        # Convert to numpy for sklearn
        y_np = y.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        # Compute per-class AUROC
        try:
            auroc_scores = roc_auc_score(y_np, y_pred_np, average=None)
        except ValueError as e:
            logger.error(f"AUROC computation failed: {e}")
            return self._get_zero_metrics()
        
        mean_auroc = 0.0
        for i, score in enumerate(auroc_scores):
            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            metrics[f"auroc_{class_name}"] = float(score)
            mean_auroc += score
        
        metrics["mean_auroc"] = mean_auroc / len(auroc_scores)
        
        # Find optimal thresholds using Youden's J statistic
        num_classes = y_np.shape[1]
        optimal_thresholds = np.zeros(num_classes)
        
        for class_idx in range(num_classes):
            fpr, tpr, thresholds = roc_curve(y_np[:, class_idx], y_pred_np[:, class_idx])
            # Youden's J = sensitivity + specificity - 1 = TPR - FPR
            j_scores = tpr - fpr
            optimal_thresholds[class_idx] = thresholds[np.argmax(j_scores)]
        
        # Binarize predictions using optimal thresholds
        y_pred_binary = (y_pred_np > optimal_thresholds).astype(np.int32)
        
        # Compute confusion matrices for each class
        confusion_matrices = multilabel_confusion_matrix(y_true=y_np, y_pred=y_pred_binary)
        
        # Compute macro-averaged metrics
        eps = 1e-7
        f1_scores = []
        accuracies = []
        specificities = []
        sensitivities = []
        precisions = []
        
        for cm in confusion_matrices:
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            
            f1_scores.append(2 * tp / (2 * tp + fn + fp + eps))
            accuracies.append((tp + tn) / (tp + tn + fp + fn + eps))
            specificities.append(tn / (tn + fp + eps))
            sensitivities.append(tp / (tp + fn + eps))
            precisions.append(tp / (tp + fp + eps))
        
        metrics["macro_f1_score"] = float(np.mean(f1_scores))
        metrics["macro_accuracy"] = float(np.mean(accuracies))
        metrics["macro_specificity"] = float(np.mean(specificities))
        metrics["macro_sensitivity"] = float(np.mean(sensitivities))
        metrics["macro_precision"] = float(np.mean(precisions))
        metrics["optimal_thresholds"] = optimal_thresholds.tolist()
        
        return metrics
    
    def _get_zero_metrics(self) -> Dict[str, Any]:
        """Return zero metrics for error cases."""
        metrics = {}
        for class_name in self.class_names:
            metrics[f"auroc_{class_name}"] = 0.0
        metrics["mean_auroc"] = 0.0
        metrics["macro_f1_score"] = 0.0
        metrics["macro_accuracy"] = 0.0
        metrics["macro_specificity"] = 0.0
        metrics["macro_sensitivity"] = 0.0
        metrics["macro_precision"] = 0.0
        metrics["optimal_thresholds"] = [0.5] * len(self.class_names)
        return metrics
