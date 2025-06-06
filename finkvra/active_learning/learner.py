import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import finkvra.active_learning.strategy as al_strategy
import finkvra.active_learning.metrics as al_metrics
import uuid
import json
import warnings


class QueryAL:
    """
    Class to perform the querying strategies of our Active Learning loops.

    This version trains the real and galactic classifiers independently and logs their selections separately.
    """
    def __init__(
        self,
        X_pool: pd.DataFrame,
        y_real_pool: pd.Series,
        y_gal_pool: pd.Series,
        strategy: str = "entropy",
        batch_size: int = 10,
        n_iterations: int = 10,
        random_state: int = 42,
    ):
        import uuid
        import numpy as np

        self.X_pool = X_pool.copy()
        self.y_real_pool = y_real_pool.copy()
        self.y_gal_pool = y_gal_pool.copy()

        self.strategy = strategy
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.run_id = uuid.uuid4().hex[:8]
        self.metric_history = []
        self.class_balance_log = {"real": [], "gal": []}
        self.selected_ids_log = {"real": [], "gal": []}
        self.dayn_log = []
        self.recall_curves = []

        np.random.seed(self.random_state)

        # Independent initialization for real and gal models
        self.unlabeled_idx = {
            "real": list(self.X_pool.index),
            "gal": list(self.y_gal_pool.dropna().index)
        }

        self.labeled_idx = {
            "real": list(np.random.choice(self.unlabeled_idx["real"], size=self.batch_size, replace=False)),
            "gal": list(np.random.choice(self.unlabeled_idx["gal"], size=self.batch_size, replace=False))
        }

        self.unlabeled_idx["real"] = [i for i in self.unlabeled_idx["real"] if i not in self.labeled_idx["real"]]
        self.unlabeled_idx["gal"] = [i for i in self.unlabeled_idx["gal"] if i not in self.labeled_idx["gal"]]

    def run(self, X_val, y_val_raw, y_real_val, y_gal_val, verbose=True, real_model_kwargs=None, gal_model_kwargs=None):

        for i in range(self.n_iterations):
            # Train REAL model
            X_train_real = self.X_pool.loc[self.labeled_idx["real"]]
            y_train_real = self.y_real_pool.loc[self.labeled_idx["real"]]
            self.real_model_kwargs = real_model_kwargs or {
                "learning_rate": 0.1,
                "l2_regularization": 10,
                "class_weight": "balanced",
                "random_state": self.random_state,
            }
            real_model = HistGradientBoostingClassifier(**self.real_model_kwargs).fit(X_train_real, y_train_real)

            # Train GALACTIC model
            X_train_gal = self.X_pool.loc[self.labeled_idx["gal"]]
            y_train_gal = self.y_gal_pool.loc[self.labeled_idx["gal"]]
            self.gal_model_kwargs = gal_model_kwargs or {
                "learning_rate": 0.2,
                "l2_regularization": 10,
                "class_weight": "balanced",
                "random_state": self.random_state,
            }
            gal_model = HistGradientBoostingClassifier(**self.gal_model_kwargs).fit(X_train_gal, y_train_gal)

            # Evaluate Recall@K AUC
            recall_curve, auc_val = al_metrics.recall_at_k_auc(
                y_val_raw["type"],
                real_model.predict_proba(X_val)[:, 1],
                gal_model.predict_proba(X_val)[:, 1],
            )
            self.metric_history.append(auc_val)
            self.recall_curves.append(recall_curve)

            # REAL model selection
            if len(self.unlabeled_idx["real"]) > 0:
                X_pool_real = self.X_pool.loc[self.unlabeled_idx["real"]]
                y_pool_real = self.y_real_pool.loc[self.unlabeled_idx["real"]]
                selected_real = al_strategy.select_batch(
                    self.strategy, real_model, X_pool_real,  y_pool_real,
                    self.batch_size
                )
                self.labeled_idx["real"] += selected_real
                self.unlabeled_idx["real"] = [idx for idx in self.unlabeled_idx["real"] if idx not in selected_real]
                self.selected_ids_log["real"].append(selected_real)
                self.class_balance_log["real"].append(
                    al_metrics.compute_class_balance(self.y_real_pool, self.y_gal_pool, selected_real)
                )

            # GALACTIC model selection
            if len(self.unlabeled_idx["gal"]) > 0:
                X_pool_gal = self.X_pool.loc[self.unlabeled_idx["gal"]]
                y_pool_gal = self.y_gal_pool.loc[self.unlabeled_idx["gal"]]
                selected_gal = al_strategy.select_batch(
                    self.strategy, gal_model, X_pool_gal, y_pool_gal,
                    self.batch_size
                )
                self.labeled_idx["gal"] += selected_gal
                self.unlabeled_idx["gal"] = [idx for idx in self.unlabeled_idx["gal"] if idx not in selected_gal]
                self.selected_ids_log["gal"].append(selected_gal)
                self.class_balance_log["gal"].append(
                    al_metrics.compute_class_balance(self.y_real_pool, self.y_gal_pool, selected_gal)
                )

            # Track dayN if present
            if "dayN" in self.X_pool.columns:
                all_selected = list(set(self.selected_ids_log["real"][-1] + self.selected_ids_log["gal"][-1]))
                self.dayn_log.append(al_metrics.compute_dayn_distribution(self.X_pool, all_selected))

            if verbose:
                print(
                    f"Iter {i+1:02d} | Recall@K AUC: {auc_val:.4f} | "
                    f"Selected real: {len(selected_real)} | gal: {len(selected_gal)} | "
                    f"Remaining real: {len(self.unlabeled_idx['real'])} | gal: {len(self.unlabeled_idx['gal'])}"
                )

            if len(self.unlabeled_idx["real"]) == 0 and len(self.unlabeled_idx["gal"]) == 0:
                warnings.warn("No more unlabeled samples to select. AL loop terminated early.", RuntimeWarning)
                break

        return real_model, gal_model


    def export_logs(self, prefix=None):

        tag = f"{prefix}_" if prefix else ""
        base = f"{tag}"

        # Metric history
        pd.Series(self.metric_history, name="recall_at_k_auc").to_csv(f"{base}metric_history.csv", index=False)
        pd.DataFrame(self.recall_curves).to_csv(f"{base}recall_curves.csv", index=False)

        # Selected samples
        for model in ["real", "gal"]:
            flat_rows = []
            for i, batch in enumerate(self.selected_ids_log[model]):
                for idx in batch:
                    flat_rows.append({"iteration": i + 1, "ATLAS_ID": idx})
            pd.DataFrame(flat_rows).to_csv(f"{base}selected_ids_{model}.csv", index=False)

        # Class balance logs
        for model in ["real", "gal"]:
            pd.DataFrame(self.class_balance_log[model]).to_csv(f"{base}class_balance_{model}.csv", index=False)

        # DayN distribution (optional)
        if self.dayn_log:
            pd.DataFrame(self.dayn_log).fillna(0).to_csv(f"{base}dayn_distribution.csv", index=False)

        # Metadata
        metadata = {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "batch_size": self.batch_size,
            "n_iterations": self.n_iterations,
            "random_state": self.random_state,
            "has_dayN": "dayN" in self.X_pool.columns,
            "real_model_kwargs": self.real_model_kwargs,
            "gal_model_kwargs": self.gal_model_kwargs,
        }
        with open(f"{base}metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

###############################################3



class QueryAL_old:
    """
    Class to perform the querying strategies of our Active Learning loops.

    Attributes:
        X_pool: pd.DataFrame
            DataFrame containing the features of the training pool of data that has not yet been selected for training.
        y_real_pool: pd.Series
            Series containing the REAL = [0,1] labels for X_pool.
        y_gal_pool: pd.Series
            Series containing the GALACTIC = [0,1] labels for X_pool.
        strategy: str
            The strategy to use for selecting the next batch of samples. Options are: "entropy", "uncertainty", "random".
        batch_size: int
            The number of samples to select in each iteration.
        n_iterations: int
            The number of iterations to run the active learning loop.
        random_state: int
            Random seed for reproducibility.
        metric_history: list
            List to store the history of the custom metric (AUC of recall@k) for each iteration.
        class_balance_log: list
            List to store the class balance.
        selected_ids_log: list
            List to store the selected sample IDs
        dayn_log: list
            List to store the dayN distribution.
    """
    def __init__(
        self,
        X_pool: pd.DataFrame,
        y_real_pool: pd.Series,
        y_gal_pool: pd.Series,
        strategy: str = "entropy",
        batch_size: int = 10,
        n_iterations: int = 10,
        random_state: int = 42,
    ):
        """
        Parameters:
            X_pool: pd.DataFrame
                DF containing the features of the training pool of data that has not yet been selected for training
            y_real_pool: pd.Series
                Series containing the REAL = [0,1] labels for X_pool. This is extracted before calling QueryAL from y_train which contains the categorical labels "garbage", "pm", "galactic", "good".
            y_gal_pool: pd.Series
                Series containing the GALACTIC = [0,1] labels for X_pool. This is extracted before calling QueryAL from y_train which contains the categorical labels "garbage", "pm", "galactic", "good".
            strategy: str
                The strategy to use for selecting the next batch of samples.
                Options are: "entropy", "uncertainty", "random".
            batch_size: int
                The number of samples to select in each iteration.
            n_iterations: int
                The number of iterations to run the active learning loop. The total number of samples will be n_iterations * batch_size.
            random_state: int
                Random seed for reproducibility. Default is 42.
        """
        #### ATTRIBUTE INITIALIZATION ####
        # We make copies of the data frames so as not to modify the original ones
        self.X_pool = X_pool.copy()
        self.y_real_pool = y_real_pool.copy()
        self.y_gal_pool = y_gal_pool.copy()

        # Store our parameters into attributes
        self.strategy = strategy
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.random_state = random_state

        # Generate a run ID
        self.run_id = uuid.uuid4().hex[:8]

        # Set up attributes that will store important metadata about the training
        # such as the values of our chosen metric, the balance of classes and  dayN,
        # and of course the ATLAS_IDs selected for training
        self.metric_history = []
        self.class_balance_log = []
        self.selected_ids_log = []
        self.dayn_log = []

        ####### SETUP THE EXPERIMENTS #######
        np.random.seed(self.random_state)

        # Initialize labeled set randomly
        # NOTE: the "unlabeled" set is the pool of data we can train on but haven't used yet
        # This is because in a classic case you don't know the labels yet and you only bother
        # to label data that has been selected in your AL strategy (that avoids labelling too much)
        # In some of our test cases like with the ATLAS data the labels are already known
        # but we keep the nomenclature anyway. (Partially because ChatGPT wrote the code and
        # and I don't want to refactor, and partially because in future it WILL run on unlabeled data).
        self.unlabeled_idx = list(self.X_pool.index)
        self.labeled_idx = list(
            np.random.choice(self.unlabeled_idx, size=self.batch_size, replace=False)
        )
        self.unlabeled_idx = [idx for idx in self.unlabeled_idx if idx not in self.labeled_idx]

    def run(self,
            X_val: pd.DataFrame,
            y_val: pd.Series,
            y_real_val: pd.Series,
            y_gal_val: pd.Series,
            verbose: bool =True,
            real_model_kwargs: dict = None,
            gal_model_kwargs: dict = None,
            ):
        """
        Parameters:
            X_val: pd.DataFrame
                DataFrame containing the features of the validation set.
            y_real_val: pd.Series
                Series containing the REAL = [0,1] labels for X_val.
            y_gal_val: pd.Series
                Series containing the GALACTIC = [0,1] labels for X_val.
            verbose: bool
                If True, print progress messages. Default is True.
        """

        # For each iteration i
        for i in range(self.n_iterations):
            # The training data is the "labeled" or "selected" indexes
            X_train = self.X_pool.loc[self.labeled_idx]
            y_train_real = self.y_real_pool.loc[self.labeled_idx]
            #y_train_gal = self.y_gal_pool.loc[self.labeled_idx].dropna()
            #X_train_gal = X_train.loc[y_train_gal.index]

            # We train the REAL classifier using the same hyperparmaters as
            # in the ATLAS VRA paper.
            self.real_model_kwargs = real_model_kwargs or {
                "learning_rate": 0.1,
                "l2_regularization": 10,
                "class_weight": "balanced",
                "random_state": self.random_state,
            }
            real_model = HistGradientBoostingClassifier(**self.real_model_kwargs
            ).fit(X_train, y_train_real)

            # Same for the GALACTIC classifier
            self.gal_model_kwargs = gal_model_kwargs or { "learning_rate": 0.2,
                "l2_regularization": 10,
                "class_weight": "balanced",
                "random_state": self.random_state,
                                                          }

            gal_model = HistGradientBoostingClassifier(**self.gal_model_kwargs
                                                       ).fit(X_train, y_train_real)

            # Evaluate custom metric on the validation set


            recall_curve, auc_val = al_metrics.recall_at_k_auc(
                y_val["type"],
                real_model.predict_proba(X_val)[:, 1],
                gal_model.predict_proba(X_val)[:, 1],
            )

            self.metric_history.append(auc_val)

            # Next batch to "label" or select.
            X_unlabeled = self.X_pool.loc[self.unlabeled_idx]

            selected_atlas_ids = al_strategy.select_batch(
                strategy=self.strategy,
                model=real_model,
                X_pool=X_unlabeled,
                y_true=y_val,
                batch_size=min(self.batch_size, len(X_unlabeled)),
                random_state=self.random_state,
            )
            # store the selected ATLAS_IDs
            # and regenerate the list of ATLAS_IDs that are still unlabeled/unselected
            self.labeled_idx += selected_atlas_ids
            self.unlabeled_idx = [idx for idx in self.unlabeled_idx if idx not in selected_atlas_ids]
            # Log the ATLAS_IDs selected this batch
            self.selected_ids_log.append(selected_atlas_ids)

            # Log class balance
            class_counts = al_metrics.compute_class_balance(
                self.y_real_pool,
                self.y_gal_pool,
                selected_atlas_ids
            )
            class_counts["iteration"] = i + 1
            self.class_balance_log.append(class_counts)

            # Log dayN distribution if present
            if "dayN" in self.X_pool.columns:
                dayn_counts = al_metrics.compute_dayn_distribution(self.X_pool, selected_atlas_ids)
                dayn_counts["iteration"] = i + 1
                self.dayn_log.append(dayn_counts)

            if verbose:
                print(
                    f"Iter {i+1:02d} | Recall@K AUC: {auc_val:.4f} | "
                    f"Selected: {len(selected_atlas_ids)} | Remaining: {len(self.unlabeled_idx)}"
                )

            if len(self.unlabeled_idx) == 0:
                warnings.warn("No more unlabeled samples to select. AL loop terminated early.", RuntimeWarning)
                break

        return real_model, gal_model

    def export_logs(self, prefix=None):
        """
        Exports logs with filenames prefixed by run ID and optional user-defined prefix.
        Also writes metadata to JSON file .
        """
        tag = f"{prefix}_" if prefix else ""
        base = f"{tag}"#{self.run_id}"

        # Metric history
        pd.Series(self.metric_history, name="recall_at_k_auc").to_csv(f"{base}metric_history.csv", index=False)

        # Selected samples
        flat_rows = []
        for i, batch in enumerate(self.selected_ids_log):
            for idx in batch:
                flat_rows.append({"iteration": i + 1, "ATLAS_ID": idx})
        pd.DataFrame(flat_rows).to_csv(f"{base}selected_ids.csv", index=False)

        # Class balance
        pd.DataFrame(self.class_balance_log).to_csv(f"{base}class_balance.csv", index=False)

        # DayN distribution (if available)
        if self.dayn_log:
            pd.DataFrame(self.dayn_log).fillna(0).to_csv(f"{base}dayn_distribution.csv", index=False)

        # Metadata
        metadata = {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "batch_size": self.batch_size,
            "n_iterations": self.n_iterations,
            "random_state": self.random_state,
            "has_dayN": "dayN" in self.X_pool.columns,
            "real_model_kwargs": self.real_model_kwargs,
            "gal_model_kwargs": self.gal_model_kwargs,
        }
        json.dump(metadata, open(f"{base}metadata.json", "w"), indent=2)
