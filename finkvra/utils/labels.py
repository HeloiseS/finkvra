import numpy as np
import pandas as pd
import webbrowser
import time
from datetime import datetime
import logging
import os
# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s")



# Maps from original label to binary "real"
dict_type_to_preal = {
    'auto-garbage': 0,
    'garbage': 0,
    'pm': 0,
    'galactic': 1,
    'good': 1,
    'AGN': 1,
}

# Maps from original label to binary "galactic"
dict_type_to_pgal = {
    'auto-garbage': np.nan,
    'garbage': np.nan,
    'pm': 1,
    'galactic': 1,
    'good': 0,
    'AGN': np.nan,
}

def preprocess_labels(y_df):
    """
    Converts a 'type' column of categorical labels into binary real and galactic Series.

    Parameters:
        y_df (pd.DataFrame): must contain a 'type' column and ATLAS_ID index.

    Returns:
        (pd.Series, pd.Series): (y_real, y_gal), both indexed by ATLAS_ID
    """
    y_real = y_df['type'].map(lambda x: dict_type_to_preal[x])
    y_gal = y_df['type'].map(lambda x: dict_type_to_pgal[x])
    return y_real, y_gal


def cli_label_alerts(
    df: pd.DataFrame,
    output: str = "~/Data/FinkZTFStream/labeled.csv",
    allowed_labels: dict = None,
    resume: bool = True,
    sleep: float = 1.0,
) -> None:
    """Command-line loop for labeling alerts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'objectId' and 'candid' columns.
    output : str
        Path to CSV file where labels will be stored.
    allowed_labels : dict
        Dictionary mapping shortcuts to label names.
    resume : bool
        If True, skip rows already labeled in the output file.
    sleep : float
        Time in seconds to wait after opening a URL.
    """
    output = os.path.expanduser(output)
    df = df.copy().reset_index()
    df["candid"] = df["candid"].astype(str)
    df["url"] = df["objectId"].apply(lambda oid: f"https://lasair-ztf.lsst.ac.uk/objects/{oid}")

    if allowed_labels is None:
        allowed_labels = {
            "r": "real",
            "x": "extragal",
            "g": "gal",
            "a": "agn",
            "b": "bogus",
            "v": "varstar",
        }

    # Load and normalize existing labels
    if resume and os.path.exists(output):
        existing = pd.read_csv(output, dtype={"candid": str})
        labeled_ids = set(existing["candid"])
        df = df[~df["candid"].isin(labeled_ids)]
        print(f"Resuming from previous file. {len(df)} unlabeled samples remaining.")
    else:
        existing = pd.DataFrame(columns=["candid", "objectId", "label", "timestamp"])

    i = 0
    while i < len(df):
        row = df.iloc[i]
        print(f"\n{i+1}/{len(df)} — {row['objectId']} (candid {row['candid']})")

        opened = False
        while True:
            if not opened:
                print("Opening in browser...")
                webbrowser.open(row["url"])
                time.sleep(sleep)
                opened = True

            inp = input("Label [r/x/g/a/b/v] (s=skip, z=undo, q=quit): ").strip().lower()

            if inp == "q":
                existing.to_csv(output, index=False)
                print("Exiting. Progress saved.")
                return

            elif inp == "s":
                print("Skipped.")
                i += 1
                break

            elif inp == "z":
                if len(existing) > 0:
                    last = existing.iloc[-1]
                    print(f"Undoing last label: {last['objectId']} as '{last['label']}'")
                    existing = existing.iloc[:-1]
                    existing.to_csv(output, index=False)
                    i = max(i - 1, 0)
                    break
                else:
                    print("Nothing to undo.")

            elif inp in allowed_labels:
                label = allowed_labels[inp]
                timestamp = datetime.utcnow().isoformat()
                new_row = {
                    "candid": row["candid"],
                    "objectId": row["objectId"],
                    "label": label,
                    "timestamp": timestamp,
                }
                existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
                existing.to_csv(output, index=False)
                print(f"Labeled as '{label}'.")
                i += 1
                break

            else:
                print(f"Invalid input '{inp}'. Try: r/x/g/a/b/v/s/z/q")

    logger.info(f"Complete. Saved {len(existing)} labeled entries to {output}.")