from glob import glob
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
from datetime import datetime
import os
from finkvra.utils.features import make_features as fvra_make_features
from finkvra.utils.labels import cli_label_one_object as fvra_cli_label_one_object
import json
from mlflow.tracking import MlflowClient
import logging
from mlflow.models.signature import infer_signature
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, 
                    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s")


class ALPipeline(object):
    """Active Learning Pipeline to Run Experiments on the Fink ZTF Stream."""

    # Class attributes

    label2galclass = {'real': np.nan,
                  'extragal': 0,
                  'gal': 1,
                  'agn': 0,
                  'bogus': np.nan,
                  'varstar': 1,
                  None: np.nan,
                  np.nan: np.nan,
                 }

    label2realclass = {'real': 1,
                    'extragal': 1,
                    'gal': 1,
                    'agn': 1,
                    'bogus': 0,
                    'varstar': 1,
                    None: np.nan,
                    np.nan: np.nan,
                    }

    label2transclass = {'real': np.nan,
                    'extragal': 1,
                    'gal': 1,
                    'agn': 0,
                    'bogus': np.nan,
                    'varstar': 0,
                    None: np.nan,
                    np.nan: np.nan,
                    }

    label2class = {'gal': label2galclass,
                'real': label2realclass,
                'trans': label2transclass
                }
    
    def __init__(self, 
                 configfile: str,
                 parquet_glob_path: str = None,
                 batch_size: int = None,
                 ):
        """Initialize our AL Pipeline

        Parameters:
            configfile (str): Path to the YAML configuration file.
        """

        self.config = self._read_config(configfile)
        self.parquet_glob_path = parquet_glob_path
        self.BATCH_SIZE = batch_size

        self.__config() # Load the parameters from the config file and do some checks
        logger.info(f"Settings loaded from {configfile}")

        self.__mlflow_setup()  # Setup MLflow tracking server
        logger.info(f"ML Flow set-up complete.")

        self.__load_parquet()

        self.__make_features()  
        logger.info(f"Features made for {self.X.shape[0]} samples (unique candid) - {self.meta.objectId.unique().shape[0]} unique objects")

        ### NOW GONNA HAVE TO ADD LOGIC FOR WHEN ROUND IS AND ISN'T 0 

    def _read_config(self, configfile: str) -> dict:
        """Read the configuration file."""
        with open(configfile, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def __config(self):
        """Load the parameters from the config file and do some checks."""
        self.EXPERIMENT = self.config['EXPERIMENT']
        self.SAMPLING_STRATEGY = self.config['SAMPLING_STRATEGY']
        self.HOST = self.config['HOST']
        self.PORT = self.config['PORT']
        self.PARQUET_GLOB_PATH = self.config['PARQUET_GLOB_PATH'] 
        if self.parquet_glob_path is not None:
            self.PARQUET_GLOB_PATH = self.parquet_glob_path
            logger.info(f"Using custom parquet glob path: {self.PARQUET_GLOB_PATH}")


        self.MODEL_TYPE = self.config['MODEL_TYPE']
        if self.MODEL_TYPE not in ['gal', 'real', 'trans']:
            raise ValueError(f"MODEL_TYPE must be one of ['gal', 'real', 'trans'], got {self.MODEL_TYPE} instead.")
        
        self.LABELS_PATH = self.config['LABELS_PATH']
        self.OUTPUT_ROOT = self.config['OUTPUT_ROOT']

        self.N_batch_0 = self.config['N_batch_0']  # initial batch size
        self.N_batch_i = self.config['N_batch_i']  # batch size for subsequent iterations
        
        if self.BATCH_SIZE is not None:
            logger.info(f"Using custom batch size: {self.BATCH_SIZE}")
        else:
            self.BATCH_SIZE = None

        self.PARAMS = self.config['PARAMS']

        # Check if the {OUTPUT_ROOT}{EXPERIMENT} directory exists, if not create it
        if not os.path.exists(f"{self.OUTPUT_ROOT}{self.EXPERIMENT}"):
            # log a warning and then create the directory
            logger.warning(f"Output directory {self.OUTPUT_ROOT}{self.EXPERIMENT} does not exist. Creating it.")
            os.makedirs(f"{self.OUTPUT_ROOT}{self.EXPERIMENT}")

        # +++++++++++++++++++++
        # Constants derived from Config
        # +++++++++++++++++++++
        self.model_subpath = f"{self.MODEL_TYPE}_model"
        self.training_ids_path = f"{self.OUTPUT_ROOT}{self.EXPERIMENT}/{self.SAMPLING_STRATEGY}_{self.MODEL_TYPE}_training_ids.csv"
        self.training_ids_artifact_path = f"{self.SAMPLING_STRATEGY}_{self.MODEL_TYPE}_training_ids.csv"
        self.labels = pd.read_csv(self.LABELS_PATH, index_col=0)


    def __mlflow_setup(self):
        """Setup MLflow tracking server."""
        mlflow.set_tracking_uri(f"http://{self.HOST}:{self.PORT}")
        mlflow.set_experiment(self.EXPERIMENT)

        self.client = MlflowClient()
        experiment = self.client.get_experiment_by_name(self.EXPERIMENT)

        # Get the run idea of the last SUCCESSFUL run
        experiment_id = experiment.experiment_id

        runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        mlflow_uri = mlflow.get_tracking_uri()

        if not runs:
            self.CURRENT_ROUND = 0
            if self.BATCH_SIZE is None:
                self.BATCH_SIZE = self.config['N_batch_0']
            logger.info(f"This is ROUND 0 of EXPERIMENT {self.EXPERIMENT}. Batch size is {self.BATCH_SIZE}")
        else:
            last_run = runs[0]
            self.prev_run_id = last_run.info.run_id
            BATCH_SIZE = self.config['N_batch_i']
            logger.info(f"Found Successful run - ID: {self.prev_run_id}. Batch size is {BATCH_SIZE}")
            
            
            logger.info(f"Loading artifacts from previous run: Previous training IDs")
            previous_ids_path = self.client.download_artifacts(self.prev_run_id,
                                                               self.training_ids_artifact_path)
            previous_ids_df = pd.read_csv(previous_ids_path)
            self.CURRENT_ROUND= previous_ids_df.iloc[-1]['round'] + 1
            logger.info(f"Previous trainings IDs found - CURRENT ROUND: {self.CURRENT_ROUND}")
                        
            logger.info(f"Loading artifacts from previous run: Model")
            model_uri = f"runs:/{self.prev_run_id}/{self.model_subpath}"
            self.clf = mlflow.sklearn.load_model(model_uri)
    
    def __load_parquet(self):
        """Load the parquet files from the glob path."""
        if self.PARQUET_GLOB_PATH is None:
            raise ValueError("PARQUET_GLOB_PATH is not set. Please set it in the config file.")
        
        logger.info(f"Loading parquet files from {self.PARQUET_GLOB_PATH}")
        files = sorted(glob(self.PARQUET_GLOB_PATH))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.PARQUET_GLOB_PATH}")
        
        df_list = [pd.read_parquet(file) for file in files]
        self.data = pd.concat(df_list, ignore_index=True)
        logger.info(f"{len(files)} .parquet files loaded from {self.PARQUET_GLOB_PATH}")
        
    def __make_features(self):
        X, meta = fvra_make_features(self.data)
        # remove samples that have no positive detections
        valid_candid = list(X[X.ndets > 0].index.values)
        self.X = X.loc[valid_candid]
        self.meta = meta.loc[valid_candid]
