import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from finkvra.utils import consumer


def make_mock_sherlock_response(obj_type):
    return {
        'classifications': {'transient_00000': [obj_type]},
        'crossmatches': [{'separationArcsec': 1.23}]
    }


@patch('finkvra.utils.consumer.lasair.lasair_client')
def test_run_sherlock_filters_classes(mock_lasair):
    # Setup mock Lasair client
    mock_client = MagicMock()
    mock_client.sherlock_position.side_effect = [
        make_mock_sherlock_response('SN'),
        make_mock_sherlock_response('AGN'),
        make_mock_sherlock_response('VS')
    ]
    mock_lasair.return_value = mock_client

    # Dummy DataFrame
    df = pd.DataFrame({
        'candid': ['1', '2', '3'],
        'objectId': ['a', 'b', 'c'],
        'ra': [10.0, 11.0, 12.0],
        'dec': [-10.0, -11.0, -12.0],
        'drb': [0.99, 0.95, 0.98],
        'mjd': [60000.0, 60001.0, 60002.0],
        'mag': [18.5, 18.7, 19.0],
        'maglim': [20.0, 20.2, 20.1],
        'fid': [1, 2, 1],
        'lc_features_g': [None]*3,
        'lc_features_r': [None]*3,
    })

    clean = consumer.run_sherlock(df)
    assert clean.shape[0] == 1
    assert clean['objectId'].iloc[0] == 'a'


def test_process_alerts_missing_candidate():
    alert = {
        'candid': 1,
        'objectId': 'ZTF1',
        'lc_features_g': None,
        'lc_features_r': None
        # 'candidate' key is missing
    }
    with pytest.raises(KeyError):
        consumer.process_alerts([alert])


def test_process_alerts_valid_minimal():
    alert = {
        'candid': 1,
        'objectId': 'ZTF1',
        'candidate': {
            'ra': 10.0,
            'dec': -10.0,
            'drb': 0.9,
            'jd': 60000.0 + 2400000.5,
            'fid': 1,
            'magpsf': 18.5,
            'diffmaglim': 20.0
        },
        'lc_features_g': None,
        'lc_features_r': None,
        'prv_candidates': []
    }
    df = consumer.process_alerts([alert])
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert df['ra'].iloc[0] == 10.0


@patch('finkvra.utils.consumer.AlertConsumer')
@patch('finkvra.utils.consumer.run_sherlock')
@patch('finkvra.utils.consumer.process_alerts')
def test_poll_n_alerts_pipeline(mock_process_alerts, mock_run_sherlock, mock_alert_consumer):
    # Setup mock consumer
    fake_alert = {
        'candidate': {'ra': 1.0, 'dec': 2.0, 'drb': 0.9, 'jd': 2450000.5, 'fid': 1},
        'candid': 1,
        'objectId': 'ZTF1',
        'lc_features_g': None,
        'lc_features_r': None
    }
    mock_consumer_instance = MagicMock()
    mock_consumer_instance.consume.return_value = [['topic', fake_alert, 'key']]
    mock_alert_consumer.return_value = mock_consumer_instance

    # Mock processing functions
    dummy_df = pd.DataFrame({'dummy': [1]})
    mock_process_alerts.return_value = dummy_df
    mock_run_sherlock.return_value = dummy_df

    # Patch the DataFrame save to_parquet method
    with patch.object(pd.DataFrame, 'to_parquet') as mock_save:
        consumer.poll_n_alerts(
            {'bootstrap.servers': 'localhost:1234', 'group.id': 'test'},
            ['fink_vra_ztf'],
            n=1,
            outidr='/tmp/'
        )
        assert mock_consumer_instance.consume.called
        assert mock_process_alerts.called
        assert mock_run_sherlock.called
        assert mock_save.called