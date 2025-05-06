import numpy as np

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