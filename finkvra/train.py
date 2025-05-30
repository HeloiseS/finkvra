import argparse
import yaml
import pandas as pd
from finkvra.active_learning.learner import QueryAL
from finkvra.utils.labels import preprocess_labels

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def override_config(config, args):
    # Loop over the arguments that can be overridden
    for key in ["strategy", "batch_size", "n_iterations", "random_seed", "prefix"]:
        # gets the CLI value
        val = getattr(args, key)
        if val is not None:
            config[key] = val
    return config


def run_experiment(dataset, config):
    """
    Runs a single QueryAL experiment for 'day1' or 'dayn' using paths in the config.

    Parameters:
        dataset (str): 'day1' or 'dayn'
        config (dict): Full config dictionary (after CLI overrides)
    """
    paths = config[dataset]  # e.g. config['day1'] or config['dayn']

    # Load data
    X_train = pd.read_csv(paths["X_train"], index_col=0)
    y_train_raw = pd.read_csv(paths["y_train"], index_col=0)
    X_val = pd.read_csv(paths["X_val"], index_col=0)
    y_val_raw = pd.read_csv(paths["y_val"], index_col=0)

    # Preprocess labels
    y_train_real, y_train_gal = preprocess_labels(y_train_raw)
    y_val_real, y_val_gal = preprocess_labels(y_val_raw)

    # Run active learning
    learner = QueryAL(
        X_pool=X_train,
        y_real_pool=y_train_real,
        y_gal_pool=y_train_gal,
        strategy=config["strategy"],
        batch_size=config["batch_size"],
        n_iterations=config["n_iterations"],
        random_state=config["random_seed"]
    )

    print(f"Running experiment on {dataset.upper()} data...")
    learner.run(X_val, y_val_raw,y_val_real, y_val_gal)
    learner.export_logs(config.get("prefix"))

def main():
    # CLI overrides of the config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--strategy", type=str, help="Override sampling strategy")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--n-iterations", type=int, help="Override number of iterations")
    parser.add_argument("--random-seed", type=int, help="Override random seed")
    parser.add_argument("--prefix", type=str, help="Optional prefix for output files")
    parser.add_argument("--day1", action="store_true", help="Run Day 1 experiment")
    parser.add_argument("--dayn", action="store_true", help="Run Day N experiment")
    args = parser.parse_args()

    # Combine default config with CLI overrides
    config = override_config(load_config(args.config), args)

    # If no flags are set, run both
    if not args.day1 and not args.dayn:
        run_experiment("day1", config)
        run_experiment("dayn", config)
    else:
        if args.day1:
            run_experiment("day1", config)
        if args.dayn:
            run_experiment("dayn", config)

if __name__ == "__main__":
    main()