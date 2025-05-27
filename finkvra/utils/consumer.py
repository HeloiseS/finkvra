""" Poll the Fink servers only once at a time """
from fink_client.consumer import AlertConsumer
from fink_client.configuration import load_credentials
from fink_client.visualisation import extract_field

from astropy.time import Time
import pandas as pd
import numpy as np
import time

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
        clean_dat = process_alerts(alerts)
        clean_dat.to_csv(outidr + 'alerts.csv', mode='a', header=False)


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