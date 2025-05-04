import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from finkvra.active_learning import strategy


# Create a small dummy dataset
@pytest.fixture
def dummy_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, index=np.arange(100))
    y_series = pd.Series(y, index=X_df.index)
    return X_df, y_series


# Fit a simple model
@pytest.fixture
def trained_model(dummy_data):
    X_df, y_series = dummy_data
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_df, y_series)
    return model


def test_random_sampling(dummy_data):
    X_df, _ = dummy_data
    selected = strategy.random_sampling(X_df, batch_size=10, random_state=0)
    assert isinstance(selected, list)
    assert len(selected) == 10
    # line below just checks selected indexes are valid indexes of X_df
    assert all(idx in X_df.index for idx in selected)


def test_uncertainty_sampling(dummy_data, trained_model):
    X_df, _ = dummy_data
    selected = strategy.uncertainty_sampling(trained_model, X_df, batch_size=10)
    assert isinstance(selected, list)
    assert len(selected) == 10
    assert all(idx in X_df.index for idx in selected)


def test_entropy_sampling(dummy_data, trained_model):
    X_df, _ = dummy_data
    selected = strategy.entropy_sampling(trained_model, X_df, batch_size=10)
    assert isinstance(selected, list)
    assert len(selected) == 10
    assert all(idx in X_df.index for idx in selected)


def test_strategies_select_different_samples(dummy_data, trained_model):
    X_df, _ = dummy_data
    rand = set(strategy.random_sampling(X_df, batch_size=10, random_state=0))
    unc = set(strategy.uncertainty_sampling(trained_model, X_df, batch_size=10))
    ent = set(strategy.entropy_sampling(trained_model, X_df, batch_size=10))

    # Sanity check: at least one strategy is different from another
    assert not (rand == unc == ent)


def test_select_batch_dispatcher(dummy_data, trained_model):
    X_df, _ = dummy_data

    # Test each valid strategy
    for strat in ["random", "uncertainty", "entropy"]:
        selected = strategy.select_batch(
            strategy=strat,
            model=trained_model,
            X_pool=X_df,
            batch_size=5,
            random_state=0
        )
        assert isinstance(selected, list)
        assert len(selected) == 5
        assert all(idx in X_df.index for idx in selected)

    # Test invalid strategy
    with pytest.raises(ValueError, match="Unknown strategy"):
        strategy.select_batch(
            strategy="not_a_real_strategy",
            model=trained_model,
            X_pool=X_df,
            batch_size=5
        )