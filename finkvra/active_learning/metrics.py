import numpy as np
import pandas as pd
from sklearn.metrics import auc


def recall_at_k_auc(y_real_true, y_gal_true, y_real_proba, y_gal_proba, max_k=None):
    """
    Computes recall@k and area under recall curve (normalized) for 'real and not galactic' samples.

    Parameters:
    - y_real_true: array-like, true binary labels for 'real'
    - y_gal_true: array-like, true binary labels for 'galactic'
    - y_real_proba: array-like, predicted probability for 'real'
    - y_gal_proba: array-like, predicted probability for 'galactic'
    - max_k: optional int, maximum rank K (default: len(y))

    Returns:
    - recall_at_k: list of recall@k values
    - auc_recall: float, normalized area under the recall curve
    """
    y_real_true = np.array(y_real_true)
    y_gal_true = np.array(y_gal_true)
    y_real_proba = np.array(y_real_proba)
    y_gal_proba = np.array(y_gal_proba)

    dist = np.sqrt((1 - y_real_proba) ** 2 + y_gal_proba ** 2)
    sorted_idx = np.argsort(dist)

    is_relevant = (y_real_true == 1) & (y_gal_true == 0)
    n_relevant = is_relevant.sum()

    if n_relevant == 0:
        return [0.0] * (max_k or len(y_real_true)), 0.0

    if max_k is None:
        max_k = len(y_real_true)

    recall_at_k = []
    for k in range(1, max_k + 1):
        top_k_idx = sorted_idx[:k]
        relevant_found = is_relevant[top_k_idx].sum()
        recall_at_k.append(relevant_found / n_relevant)

    auc_recall = auc(np.arange(1, max_k + 1), recall_at_k) / max_k
    return recall_at_k, auc_recall


def compute_class_balance(y_real_pool: pd.Series, y_gal_pool: pd.Series, selected_ids: list):
    """
    Returns class distribution for a given batch of selected ATLAS_IDs.

    Class labels:
    - 'not_real'
    - 'real_galactic'
    - 'real_not_galactic'

    Returns:
    - dict: {class_name: count}
    """
    y_real_sel = y_real_pool.loc[selected_ids]
    y_gal_sel = y_gal_pool.loc[selected_ids]

    counts = {
        "not_real": ((y_real_sel == 0)).sum(),
        "real_galactic": ((y_real_sel == 1) & (y_gal_sel == 1)).sum(),
        "real_not_galactic": ((y_real_sel == 1) & (y_gal_sel == 0)).sum()
    }
    return counts


def compute_dayn_distribution(X_pool: pd.DataFrame, selected_ids: list):
    """
    Returns a dictionary: {dayN_value: count} for selected dayN samples.
    """
    selected = X_pool.loc[selected_ids]
    return selected["dayN"].value_counts().to_dict()