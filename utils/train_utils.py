import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, filename, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Saved model to {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

def compute_metrics(gt_labels, pred_probs):
    """
    Compute metrics (accuracy, AUC, sensitivity, specificity, composite score).
    Code from the handout's evaluate.py.
    """
    metrics = {}

    # Convert to one-hot encoding
    y_true = label_binarize(gt_labels, classes=[0, 1, 2])
    y_pred = pred_probs

    # Compute micro-averaged ROC curve
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel(), drop_intermediate=False)
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")

    # Sensitivity at 90% specificity
    specificity_threshold = 0.90
    fpr_threshold = 1 - specificity_threshold
    sensitivity_at_90_specificity = np.interp(fpr_threshold, fpr, tpr)

    # Specificity at 90% sensitivity
    sensitivity_threshold = 0.90
    fpr_at_90_sensitivity = np.interp(sensitivity_threshold, tpr, fpr)
    specificity_at_90_sensitivity = 1 - fpr_at_90_sensitivity

    amalgamated_results = [roc_auc, specificity_at_90_sensitivity, sensitivity_at_90_specificity]
    averaged_results = np.mean(amalgamated_results)

    metrics["results"] = {
        "AUC": roc_auc,
        "Specificity": specificity_at_90_sensitivity,
        "Sensitivity": sensitivity_at_90_specificity,
        "Score": averaged_results
    }

    return metrics
