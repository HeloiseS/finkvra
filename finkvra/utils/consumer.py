""" Poll the Fink servers only once at a time """
import logging
from fink_client.consumer import AlertConsumer
from fink_client.visualisation import extract_field
import pandas as pd
import numpy as np
from datetime import datetime
import lasair
import os

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s")



def process_alerts(alerts: list) -> pd.DataFrame:
    """Process raw alerts into a cleaned DataFrame."""

    logger.info("Running alert processing")

    dat = pd.DataFrame.from_records(alerts)
    dat['mag'] = dat.apply(lambda alert: extract_field(alert, 'magpsf'), axis=1)
    dat['maglim'] = dat.apply(lambda alert: extract_field(alert, 'diffmaglim'), axis=1)
    dat['mjd'] = dat.apply(lambda alert: extract_field(alert, 'jd'), axis=1) - 2400000.5
    dat['fid'] = dat.apply(lambda alert: extract_field(alert, 'fid'), axis=1)
    dat['ra'] = dat.apply(lambda row: row['candidate']['ra'], axis=1)
    dat['dec'] = dat.apply(lambda row: row['candidate']['dec'], axis=1)
    dat['drb'] = dat.apply(lambda row: row['candidate']['drb'], axis=1)


    logger.info("Alerts processed into dataframe: %d", len(dat))

    return dat[[
        'candid', 'objectId', 'ra', 'dec', 'drb',
        'mjd', 'mag', 'maglim', 'fid',
        'lc_features_g', 'lc_features_r'
    ]]


def run_sherlock(alert_data:pd.DataFrame):
    """Run Sherlock on the alert data processed by process_data."""

    logger.info("Running Sherlock classification on alerts.")
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

    logger.info("Successfully retrieved Sherlock classifications for %d alerts.", len(sherl_class))
    data_w_sherl = alert_data.join(pd.DataFrame(np.atleast_2d([sherl_class, sherl_separcsec]).T, 
                                                columns=['sherl_class', 'sep_arcsec']))
    
    # remove AGNs and Variable Stars
    mask = (data_w_sherl['sherl_class'] == 'AGN') | (data_w_sherl['sherl_class'] == 'VS')
    clean_data = data_w_sherl[~mask]
    logger.info("After removing AGNs and Variable Stars, %d alerts remain.", clean_data.shape[0])
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

    logger.info(f"Polling {n} alerts from topics: {topics}")

    maxtimeout = 5
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Instantiate a consumer
    try:
        consumer = AlertConsumer(topics, myconfig)
        out = consumer.consume(n, maxtimeout)
    except Exception as e:
        logger.error(f"Failed to connect or consume alerts: {e}")
        return

    if len(out) > 0:
        topics, alerts, keys = np.transpose(out) 
        logger.info(f"Received {len(alerts)} alerts.")
    else:
        logger.info(f"No alerts received in the last {maxtimeout} seconds.")
        consumer.close()
        return
    
    try:
        alert_data = process_alerts(alerts)
        clean_dat = run_sherlock(alert_data)
        if clean_dat.empty:
            logger.info("No alerts left after filtering.")
        else:
            outpath = os.path.expanduser(outidr + f'{prefix}_alerts.parquet')
            clean_dat.to_parquet(outpath)
            logger.info(f"Saved {len(clean_dat)} cleaned alerts to {outpath}")
    except Exception as e:
        logger.error(f"Error during alert processing: {e}")


    consumer.close()
    logger.info("Consumer connection closed.")
    return 

if __name__ == "__main__":
    """ Poll the servers only once at a time """

    # load user configuration
    # to fill
    myconfig = {
        'bootstrap.servers': 'kafka-ztf.fink-broker.org:24499',
        'group.id': 'heloise_test6'
    }

    topics = ['fink_vra_ztf']

    n_alerts = 50
    poll_n_alerts(myconfig, topics, n=n_alerts)

