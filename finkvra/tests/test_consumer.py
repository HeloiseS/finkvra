# pytest --cov=finkvra --cov-report=html

import pandas as pd
import importlib.resources as pkg_resources
from finkvra.utils import consumer
import pytest
import numpy as np


@pytest.fixture
def alerts():  # safer inside the function
    from finkvra import data  # safer inside the function
    path = pkg_resources.files(data).joinpath('test_alerts.npz')
    return np.load(path, allow_pickle=True)['arr_0']

def test_process_alerts_minimal(alerts):

    df = consumer.process_alerts(alerts)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 10