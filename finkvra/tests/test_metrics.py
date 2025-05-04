import pytest
import pandas as pd
import numpy as np
from finkvra.active_learning import metrics


def test_recall_at_k_auc_basic():
    # 4 samples, one relevant at index 1 (closest to (1,0))
    y_real = [0, 1, 1, 0]
    y_gal = [0, 0, 1, 1]
    y_real_proba = [0.2, 0.9, 0.7, 0.4]
    y_gal_proba = [0.1, 0.05, 0.6, 0.8]

    recall_curve, auc_val = metrics.recall_at_k_auc(y_real,
                                                    y_gal,
                                                    y_real_proba,
                                                    y_gal_proba)

    assert isinstance(recall_curve, list)
    assert 0 <= auc_val <= 1
    assert recall_curve[-1] == 1.0  # Eventually all relevant items are found


def test_recall_at_k_auc_no_relevant():
    y_real = [0, 0, 0]
    y_gal = [1, 1, 1]
    y_real_proba = [0.1, 0.2, 0.3]
    y_gal_proba = [0.8, 0.9, 0.7]

    recall_curve, auc_val = metrics.recall_at_k_auc(y_real,
                                                    y_gal,
                                                    y_real_proba,
                                                    y_gal_proba)

    assert all(r == 0.0 for r in recall_curve)
    assert auc_val == 0.0


def test_compute_class_balance():
    idx = [1001, 1002, 1003, 1004, 1005]
    y_real_pool = pd.Series([0, 1, 1, 1, 0], index=idx)
    y_gal_pool = pd.Series([np.nan, 1, 0, 1, np.nan], index=idx)

    selected_ids = [1001, 1002, 1003, 1004]

    result = metrics.compute_class_balance(y_real_pool, y_gal_pool, selected_ids)
    assert result == {
        "not_real": 1,
        "real_galactic": 2,
        "real_not_galactic": 1
    }


def test_compute_dayn_distribution():
    df = pd.DataFrame({
        "dayN": [1, 1, 2, 3, 3, 3]
    }, index=[100, 101, 102, 103, 104, 105])

    selected_ids = [100, 102, 103, 105]
    result = metrics.compute_dayn_distribution(df, selected_ids)

    assert result == {1: 1, 2: 1, 3: 2}