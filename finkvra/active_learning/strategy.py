import numpy as np
import pandas as pd

def random_sampling(X_pool: pd.DataFrame, batch_size: int, random_state: int = 42) -> list:
    """Random sampling from the pool of "unlabeled" data.
    (when testing with ATLAS VRA data it's just the training data that's not yet been used)
    """
    return list(X_pool.sample(n=batch_size, random_state=random_state).index)


def uncertainty_sampling(model, X_pool: pd.DataFrame, batch_size: int) -> list:
    """Sampling based on how close to 0.5 the predicted probabilities are."""
    probs = model.predict_proba(X_pool)[:, 1]
    scores = -np.abs(probs - 0.5)  # highest uncertainty = closest to 0.5
    return list(pd.Series(scores, index=X_pool.index).nlargest(batch_size).index)


def entropy_sampling(model,
                     X_pool: pd.DataFrame,
                     y_true: pd.Series,
                     batch_size: int) -> list:
    """Sampling the highest entropy predictions. (That is, those with the most CONFUSION)"""
    probs = model.predict_proba(X_pool)[:, 1]
    # we need to avoid values too close to 0 or 1 in the probabilities to avoid getting infs
    # np.clip does not CLIP the array itself it clips the VALUES to the boundaries we defined,
    probs = np.clip(probs, 1e-5, 1 - 1e-5)
    entropy = -y_true.values * np.log2(probs) - (np.abs(y_true.values - probs)) * np.log2(np.abs(y_true.values - probs))

    return list(pd.Series(entropy, index=X_pool.index).nlargest(batch_size).index)


def select_batch(
    strategy: str,
    model,
    X_pool: pd.DataFrame,
    y_true: pd.Series,
    batch_size: int,
    random_state: int = 42
) -> list:
    """Select a batch of samples from the pool based on the specified strategy."""
    if strategy == "random":
        return random_sampling(X_pool, batch_size, random_state)
    elif strategy == "uncertainty":
        return uncertainty_sampling(model, X_pool, batch_size)
    elif strategy == "entropy":
        return entropy_sampling(model, X_pool, y_true, batch_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")