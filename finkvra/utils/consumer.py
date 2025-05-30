""" Poll the Fink servers only once at a time """
from fink_client.consumer import AlertConsumer
from fink_client.configuration import load_credentials
from fink_client.visualisation import extract_field

from astropy.time import Time
import pandas as pd
import numpy as np
import time
from datetime import datetime
import lasair
import os
import json

def process_alerts(alerts: list) -> pd.DataFrame:
    """Process raw alerts into a cleaned DataFrame."""
    dat = pd.DataFrame.from_records(alerts)
    dat['mag'] = dat.apply(lambda alert: extract_field(alert, 'magpsf'), axis=1)
    dat['maglim'] = dat.apply(lambda alert: extract_field(alert, 'diffmaglim'), axis=1)
    dat['mjd'] = dat.apply(lambda alert: extract_field(alert, 'jd'), axis=1) - 2400000.5
    dat['fid'] = dat.apply(lambda alert: extract_field(alert, 'fid'), axis=1)
    dat['ra'] = dat.apply(lambda row: row['candidate']['ra'], axis=1)
    dat['dec'] = dat.apply(lambda row: row['candidate']['dec'], axis=1)
    dat['drb'] = dat.apply(lambda row: row['candidate']['drb'], axis=1)

    return dat[[
        'candid', 'objectId', 'ra', 'dec', 'drb',
        'mjd', 'mag', 'maglim', 'fid',
        'lc_features_g', 'lc_features_r'
    ]]


def run_sherlock(alert_data:pd.DataFrame):
    """Run Sherlock on the alert data processed by process_data."""
    # Load the Sherlock configuration
    sherlock_config = json.load(open(os.path.join(os.path.dirname(__file__), 'sherlock_config.json')))
    
    # Initialize the Sherlock client
    client = lasair.SherlockClient(sherlock_config['url'], token=sherlock_config['token'])
    
    # Process each alert
    for _, row in alert_data.iterrows():
        response = client.submit_alert(row.to_dict())
        print(f"Submitted alert {row['candid']} with response: {response}")
    # the lasair client will be used for fetching Sherlock data
    L = lasair.lasair_client(os.environ.get('LASAIR_TOKEN'), 
                             endpoint='https://lasair-ztf.lsst.ac.uk/api')
    
    sherl_class = []
    sherl_separcsec = []

    for i in range(alert_data.shape[0]):
        _sherl = L.sherlock_position(alert_data.iloc[i]['ra'], alert_data.iloc[i]['dec'], lite=False)
        sherl_class.append(_sherl['classifications']['transient_00000'][0])
        try:
            sherl_separcsec.append(_sherl['crossmatches'][0]['separationArcsec'])
        except IndexError:
            # If orphan will get no match 
            sherl_separcsec.append(np.nan)

    data_w_sherl = alert_data.join(pd.DataFrame(np.atleast_2d([sherl_class, sherl_separcsec]).T, 
                                                columns=['sherl_class', 'sep_arcsec']))
    
    # remove AGNs and Variable Stars
    mask = (data_w_sherl['sherl_class'] == 'AGN') | (data_w_sherl['sherl_class'] == 'VS')
    clean_data = data_w_sherl[~mask]
    return clean_data

def poll_n_alerts(myconfig, topics, n=10, outidr = '~/Data/FinkZTFStream/') -> None:
    """ Connect to and poll fink servers once.

    Parameters
    ----------
    myconfig: dic
        python dictionnary containing credentials
    topics: list of str
        List of string with topic names
    """
    maxtimeout = 5
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Instantiate a consumer
    consumer = AlertConsumer(topics, myconfig)

    # Poll the servers
    out = consumer.consume(n, maxtimeout)
    if len(out) > 0:
        topics, alerts, keys = np.transpose(out) 
    else:
        topics, alerts, keys = [None], [None], [None]

    # Analyse output - we just print some values for example
    if not np.all([i is None for i in alerts]):
        alert_data = process_alerts(alerts)
        clean_dat = run_sherlock(alert_data)
        # if clean_dat is empty, we do not save it but log it
        clean_dat.to_parquet(outidr +f'{prefix}_alerts.parquet')


    else:
        print(
            'No alerts received in the last {} seconds'.format(
                maxtimeout
            )
        )

    # Close the connection to the servers
    consumer.close()


if __name__ == "__main__":
    """ Poll the servers only once at a time """

    # load user configuration
    # to fill
    myconfig = {
        'bootstrap.servers': 'kafka-ztf.fink-broker.org:24499',
        'group.id': 'heloise_test6'
    }

    topics = ['fink_vra_ztf']

    poll_n_alerts(myconfig, topics)