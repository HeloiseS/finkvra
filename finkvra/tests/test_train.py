import os
import pytest
import yaml
from finkvra.train import main as train_main
import importlib.resources as pkg_resources
import finkvra.data

def get_full_path(name):
    return str(pkg_resources.files(finkvra.data).joinpath(name))

# Helper to write a temporary config file
@pytest.fixture
def temp_config_path(tmp_path):
    config = {
        "day1": {
            "X_train": get_full_path("X_train_day1_duck1000rows.csv"),
            "y_train": get_full_path("y_train_day1_duck1000rows.csv"),
            "X_val": get_full_path("X_val_day1_duck1000rows.csv"),
            "y_val": get_full_path("y_val_day1_duck1000rows.csv"),
        },
        "dayn": {
            "X_train": get_full_path("X_train_dayN_duck1000rows.csv"),
            "y_train": get_full_path("y_train_dayN_duck1000rows.csv"),
            "X_val": get_full_path("X_val_dayN_duck1000rows.csv"),
            "y_val": get_full_path("y_val_dayN_duck1000rows.csv"),
        },
        "strategy": "entropy",
        "batch_size": 5,
        "n_iterations": 2,
        "random_seed": 42,
        "prefix": str(tmp_path / "testrun")
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

def test_train_day1_runs(monkeypatch, temp_config_path):
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--config", str(temp_config_path),
        "--day1"
    ])
    train_main()

    files = os.listdir(temp_config_path.parent)
    assert any("metric_history.csv" in f for f in files)
    assert any("metadata.json" in f for f in files)

def test_train_dayn_runs(monkeypatch, temp_config_path):
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--config", str(temp_config_path),
        "--dayn"
    ])
    train_main()

    files = os.listdir(temp_config_path.parent)
    assert any("metric_history.csv" in f for f in files)
    assert any("metadata.json" in f for f in files)