import numpy as np
import pandas as pd
from sklearn.metrics import auc


def rank_function(real_pred, gal_pred, fudge_factor=0.5, max_rank=10):
    """
    Computes a VRA-style ranking score from real and galactic probabilities.
    Higher = better (closer to (1,0)).
    """
    distance = np.sqrt((1 - real_pred) ** 2 + (fudge_factor * gal_pred) ** 2)
    max_distance = np.sqrt(1 + fudge_factor ** 2)
    return (max_distance - distance) * max_rank / max_distance


def fast_recall_curve(labels, target_class='good'):
    """
    Vectorized recall@k curve for a specific class label.

    Parameters:
        labels (pd.Series): categorical labels sorted by predicted rank (high → low)
        target_class (str): label to count as relevant (e.g. 'good')

    Returns:
        recall_at_k (np.ndarray): recall@k from k=1 to N
    """
    is_relevant = (labels == target_class).to_numpy()
    total_relevant = is_relevant.sum()

    if total_relevant == 0:
        return np.zeros(len(labels))

    cum_relevant = np.cumsum(is_relevant)
    return cum_relevant / total_relevant


def recall_at_k_auc(y_type_true,
                    y_real_proba,
                    y_gal_proba, fudge_factor=0.5, max_rank=10):
    """
    Compute recall@k curve and AUC for a VRA-style rank ordering.

    Parameters:
        y_type_true (pd.Series): categorical labels (e.g. 'good', 'galactic', etc.)
        y_real_proba (np.ndarray): predicted probability of 'real'
        y_gal_proba (np.ndarray): predicted probability of 'galactic'
        fudge_factor (float): scalar applied to galactic axis in rank
        max_rank (int): scaling factor for output rank values (not needed for ordering)

    Returns:
        recall_at_k (list of float)
        auc_recall (float)
    """
    ranks = rank_function(y_real_proba, y_gal_proba, fudge_factor, max_rank)
    df = y_type_true.to_frame(name="type").copy()
    df["rank"] = ranks
    df_sorted = df.sort_values("rank", ascending=False)

    recall = fast_recall_curve(df_sorted["type"], target_class="good")
    x = np.linspace(0, 1, len(recall))
    auc_val = auc(x, recall)

    return recall.tolist(), auc_val


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