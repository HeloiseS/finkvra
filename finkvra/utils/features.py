"""Data processing module to make features from data saved by consumer"""
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s")


# Lc features from fink I've decided to keep for now
fink_lc_features_to_keep = ['amplitude', 
                       'linear_fit_reduced_chi2', 
                       'linear_fit_slope', 
                       'linear_fit_slope_sigma', 
                       'median', 
                       'median_absolute_deviation']

def vra_lc_features(row):
    """Function to compute light curve features on each row of a dataframe.
    To be applied to data saved by the poll_n_alerts function in consumer.py.
    """
    # need to make sure I ignore the negative diffs
    
    pos_diffs = (row.isdiffpos == 't')

    # NUMBER OF DETECTIONS
    try:    
        ndets = sum(pos_diffs)
    except TypeError:
        if pos_diffs is True:
            ndets = 1
        else:
            # it's a positive diff or if it's None (to detection) we set ndets to 0 
            ndets = 0

    # NUMBER OF NON-DETECTIONS
    nnondets = sum(pd.isna(row['mag']))

    if ndets == 0:
        return 0, nnondets, np.nan, np.nan
    
    dets_median = np.nanmedian(row['mag'][pos_diffs])
    dets_std = np.nanstd(row['mag'][pos_diffs])

    return ndets, nnondets, dets_median, dets_std

def make_features(clean_data: pd.DataFrame, 
                  fink_lc_features: list = fink_lc_features_to_keep
                  ) -> pd.DataFrame:
    """Make features from the clean data DataFrame. 
    The clean data is created by the consumer module and saved as parquet files. 
    They should be loaded into dataframes first before being given to this function.
    
    Parameters:
        clean_data (pd.DataFrame): DataFrame with columns 'candid', 'objectId', 'ra', 'dec', 'drb',
                                   'mjd', 'mag', 'maglim', 'fid', 'sep_arcsec' and light curve features.
        fink_lc_features (list): List of light curve features to keep from Fink.
    """
    if fink_lc_features is None:
        raise ValueError("fink_lc_features must be provided and not None.")
    
    # Create the VRA light curve features
    vra_feat_df = clean_data.apply(lambda row: vra_lc_features(row), axis=1, result_type='expand')
    vra_feat_df.columns = ['ndets', 'nnondets', 'dets_median', 'dets_std']
    clean_data_copy = clean_data.join(vra_feat_df) # add them to the clean data

    # Our clean data has cells that contains scalars, lists and dictionaries
    # here we clean this up to have a neat features data frame X where each row
    # corresponds to a sample (alert)

    # this is where we store the fink light curve features we want to keep
    lc_features_g_series = []
    lc_features_r_series = []
    vra_features = [] # ra, dec, drb, 'ndets', 'nnondets', 'dets_median', 'dets_std'
    # We iterate over each row of the clean data (each row is an alert)
    for i in range(clean_data_copy.shape[0]):
        # first we grab the "VRA" features, some are basic context from ZTF or sherlock,
        # others are the lightcurve features we computed above
        vra_features.append(clean_data_copy.iloc[i][['candid',
                                          'objectId',
                                          'ra', 
                                          'dec', 
                                          'drb', 
                                          'ndets', 
                                          'nnondets', 
                                          'dets_median', 
                                          'dets_std', 
                                          'sep_arcsec',
                                          ]])
        # Now we grab the Fink lightcurve features for g and r bands
        lc_features_g_series.append(pd.Series(clean_data_copy.iloc[i]['lc_features_g'
                                                                      ])[fink_lc_features])
        lc_features_r_series.append(pd.Series(clean_data_copy.iloc[i]['lc_features_r'
                                                                      ])[fink_lc_features])

    # Now we create a DataFrame X that contains all the features
    X = pd.DataFrame(vra_features).join(pd.DataFrame(lc_features_g_series)
                                        ).join(pd.DataFrame(lc_features_g_series), 
                                               rsuffix='r_')
    # We use candid as a in index because it is UNIQUE
    X.set_index('candid', inplace=True)

    # Finally we separate the features and the object names
    meta = X[['objectId']]
    X = X.drop(['objectId'], axis=1)

    return X, meta