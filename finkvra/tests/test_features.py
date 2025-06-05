import pandas as pd
import numpy as np
import pytest
from finkvra.utils import features
from dustmaps.config import config

config.reset()

@pytest.fixture
def dummy_clean_data():
    # Simulate 3 alerts with mixed detections and mock features
    return pd.DataFrame({
        'candid': ['1', '2', '3'],
        'objectId': ['ZTF1', 'ZTF2', 'ZTF3'],
        'ra': [10.1, 20.2, 30.3],
        'dec': [-10.1, -20.2, -30.3],
        'drb': [0.9, 0.8, 0.95],
        'mag': [[18.5, 18.7, np.nan], [17.9, np.nan, np.nan], [19.1, 19.2, 19.0]],
        'maglim': [[20.1]*3]*3,
        'mjd': [[60000, 60001, 60002]]*3,
        'fid': [[1, 2, 1]]*3,
        'isdiffpos': ['t', 'f', np.nan],
        'sep_arcsec': [0.4, 0.6, 0.2],
        'lc_features_g': [{
            'amplitude': 1.2, 'linear_fit_reduced_chi2': 0.5, 'linear_fit_slope': 0.3,
            'linear_fit_slope_sigma': 0.05, 'median': 18.6, 'median_absolute_deviation': 0.1
        }] * 3,
        'lc_features_r': [{
            'amplitude': 1.0, 'linear_fit_reduced_chi2': 0.4, 'linear_fit_slope': 0.25,
            'linear_fit_slope_sigma': 0.04, 'median': 18.5, 'median_absolute_deviation': 0.09
        }] * 3
    })

def test_get_ebv(ra=180.0, dec=45.0):
    """Test the get_ebv function with fixed RA and Dec."""
    ebv = features.get_ebv(ra, dec)
    assert np.isclose(ebv, 0.01418992, atol=1e-5), "E(B-V) value does not match expected value."

def test_vra_lc_features_counts(dummy_clean_data):
    ndets, nnondets, median, std = features.vra_lc_features(dummy_clean_data.iloc[0])
    assert ndets == 1
    assert nnondets == 1
    assert np.isclose(median, 18.7)
    assert np.isclose(std, 0.0, atol=0.01)

def test_make_features_shapes(dummy_clean_data):
    X, meta = features.make_features(dummy_clean_data)
    assert X.shape[0] == 3
    assert meta.shape[0] == 3
    assert 'objectId' in meta.columns
    assert 'ra' in X.columns
    assert 'amplitude' in X.columns
    assert 'amplituder_' in X.columns  # r-band
    assert 'ebv' in X.columns  # E(B-V) feature