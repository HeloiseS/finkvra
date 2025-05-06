import pandas as pd
import importlib.resources as pkg_resources
import finkvra.data  # new location of your data files
import pytest
from finkvra.active_learning.learner import QueryAL
import os
from finkvra.utils.labels import preprocess_labels

def load_csv_from_package(filename):
    with pkg_resources.files(finkvra.data).joinpath(filename).open("r") as f:
        return pd.read_csv(f, index_col=0)

@pytest.fixture
def duck_day1_data():
    X_train = load_csv_from_package("X_train_day1_duck1000rows.csv")
    y_train_raw = load_csv_from_package("y_train_day1_duck1000rows.csv")
    X_val = load_csv_from_package("X_val_day1_duck1000rows.csv")
    y_val_raw = load_csv_from_package("y_val_day1_duck1000rows.csv")

    y_train_real, y_train_gal = preprocess_labels(y_train_raw)
    y_val_real, y_val_gal = preprocess_labels(y_val_raw)

    return X_train, y_train_real, y_train_gal, X_val, y_val_real, y_val_gal

@pytest.fixture
def duck_dayn_data():
    X_train = load_csv_from_package("X_train_dayN_duck1000rows.csv")
    y_train_raw = load_csv_from_package("y_train_dayN_duck1000rows.csv")
    X_val = load_csv_from_package("X_val_dayN_duck1000rows.csv")
    y_val_raw = load_csv_from_package("y_val_dayN_duck1000rows.csv")

    y_train_real, y_train_gal = preprocess_labels(y_train_raw)
    y_val_real, y_val_gal = preprocess_labels(y_val_raw)

    return X_train, y_train_real, y_train_gal, X_val, y_val_real, y_val_gal


def test_queryal_smoke_dayn(tmp_path, duck_dayn_data):
    X_train, y_train_real, y_train_gal, X_val, y_val_real, y_val_gal = duck_dayn_data

    learner = QueryAL(
        X_pool=X_train,
        y_real_pool=y_train_real,
        y_gal_pool=y_train_gal,
        strategy="entropy",
        batch_size=5,
        n_iterations=2,
        random_state=123,
    )

    model_real, model_gal = learner.run(X_val, y_val_real, y_val_gal, verbose=False)

    assert len(learner.metric_history) == 2
    assert len(learner.selected_ids_log) == 2
    assert len(learner.class_balance_log) == 2
    assert len(learner.dayn_log) == 2  # ✅ additional check for Day N logging

    prefix = tmp_path / "dayn_test"
    learner.export_logs(prefix=prefix)

    # Check logs written with run ID
    expected = [
        f"{prefix}_{learner.run_id}_metric_history.csv",
        f"{prefix}_{learner.run_id}_selected_ids.csv",
        f"{prefix}_{learner.run_id}_class_balance.csv",
        f"{prefix}_{learner.run_id}_dayn_distribution.csv",
        f"{prefix}_{learner.run_id}_metadata.json",
    ]
    for path in expected:
        assert os.path.exists(path)

def test_queryal_smoke_day1(tmp_path, duck_day1_data):
    X_train, y_train_real, y_train_gal, X_val, y_val_real, y_val_gal = duck_day1_data

    learner = QueryAL(
        X_pool=X_train,
        y_real_pool=y_train_real,
        y_gal_pool=y_train_gal,  # day 1: gal is not used but required
        strategy="random",
        batch_size=5,
        n_iterations=2,
        random_state=123,
    )

    model_real, model_gal = learner.run(X_val, y_val_real, y_val_gal, verbose=False)

    # Simple assertions
    assert len(learner.metric_history) == 2
    assert len(learner.selected_ids_log) == 2
    assert len(learner.class_balance_log) == 2

    # Export logs
    prefix = tmp_path / "test_run"
    learner.export_logs(prefix=prefix)

    # Check output files exist
    expected = [
        f"{prefix}_{learner.run_id}_metric_history.csv",
        f"{prefix}_{learner.run_id}_selected_ids.csv",
        f"{prefix}_{learner.run_id}_class_balance.csv",
        f"{prefix}_{learner.run_id}_metadata.json",
    ]
    for path in expected:
        assert os.path.exists(path)

def test_queryal_early_stop_warning(duck_day1_data):
    X_train, y_train_real, y_train_gal, X_val, y_val_real, y_val_gal = duck_day1_data

    # This should deplete the pool (1000 rows total)
    learner = QueryAL(
        X_pool=X_train,
        y_real_pool=y_train_real,
        y_gal_pool=y_train_gal,
        strategy="random",
        batch_size=500,
        n_iterations=3,
        random_state=0,
    )

    with pytest.warns(RuntimeWarning, match="No more unlabeled samples"):
        learner.run(X_val, y_val_real, y_val_gal, verbose=False)

    # Assert that only two iterations ran (3rd would require >1000 samples)
    assert len(learner.metric_history) == 1