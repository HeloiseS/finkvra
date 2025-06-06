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
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, 
                    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s")

def uncertainty_sampling_score(prediction):
    return np.abs(prediction.astype(float) - 0.5) 


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
        logger.info("ML Flow set-up complete.")

        self.__load_parquet()

        self.__make_features()  
        logger.info(f"Features made for {self.X.shape[0]} samples (unique candid) - {self.meta.objectId.unique().shape[0]} unique objects")

        ### NOW GONNA HAVE TO ADD LOGIC FOR WHEN ROUND IS AND ISN'T 0 
        if self.CURRENT_ROUND == 0:
            # could scramble them if you wanted more randomness
            self.candid_loop = self.X.index.astype(np.int64)

        elif self.CURRENT_ROUND > 0:
            self.__candid_loop() 

        # Now that we have candid_loop can do the labeling. 
        self.new_ids_df, self.updated_labels = self.__labeling()

        if self.CURRENT_ROUND > 0:
            self.train_ids_df = pd.concat([self.previous_ids_df, 
                                           self.new_ids_df]).reset_index(drop=True)
        else:
            train_ids_df = self.new_ids_df
            
        train_ids_df.to_csv(self.training_ids_path, index=False)
        logging.info(f'Saved training ids locally to {self.training_ids_path}')
        
        self.y_train = self.updated_labels.loc[train_ids_df.candid].label.map(self.label2class[self.MODEL_TYPE])
        self.X_train = self.X.loc[train_ids_df.candid]

        self.y_train.to_csv(f"{self.OUTPUT_ROOT}{self.EXPERIMENT}/y_train.csv")
        self.X_train.to_csv(f"{self.OUTPUT_ROOT}{self.EXPERIMENT}/X_train.csv")

        logger.info(f"Made y_train and X_train and saved to {self.OUTPUT_ROOT}{self.EXPERIMENT}")


        # -------------------
        # TRAIN
        # -------------------

        MODEL_TAG = f"{self.model_subpath}_round_{self.CURRENT_ROUND}"

        with mlflow.start_run(run_name=f"round_{self.CURRENT_ROUND}_{self.SAMPLING_STRATEGY}"):

            # Log metadata
            meta_info = {
                "round": int(self.CURRENT_ROUND),
                "timestamp": datetime.utcnow().isoformat(),
                "n_train": int(self.X_train.shape[0]),
                "sampling_strategy": str(self.SAMPLING_STRATEGY),
                "model_tag": str(MODEL_TAG)
            }

            with open("meta.json", "w") as f:
                json.dump(meta_info, f, indent=2)
            mlflow.log_artifact("meta.json")

            # Train model
            mlflow.log_params(self.PARAMS)
            clf_new = HistGradientBoostingClassifier(**self.PARAMS)
            clf_new.fit(self.X_train.values, self.y_train.values)
            y_pred_new = clf_new.predict(self.X_train.values)

            # Evaluate on training set
            acc = accuracy_score(self.y_train, y_pred_new)
            mlflow.log_metric("accuracy", acc)
            
            prec = precision_score(self.y_train, y_pred_new)
            mlflow.log_metric("precision", prec)
            
            recall = recall_score(self.y_train, y_pred_new)
            mlflow.log_metric("recall", recall)
            
            f1 = f1_score(self.y_train, y_pred_new)
            mlflow.log_metric("f1-score", f1)

            # Log model
            signature = infer_signature(self.X_train, y_pred_new)
            mlflow.sklearn.log_model(
                clf_new,
                artifact_path=self.model_subpath,
                signature=signature,
                input_example=self.X_train.iloc[:2]
            )

            # Save training state
            mlflow.log_artifact(self.training_ids_path)
            mlflow.log_artifact(f"{self.OUTPUT_ROOT}{self.EXPERIMENT}/y_train.csv")
            mlflow.log_artifact(f"{self.OUTPUT_ROOT}{self.EXPERIMENT}/X_train.csv")



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
            
            
            logger.info("Loading artifacts from previous run: Previous training IDs")
            previous_ids_path = self.client.download_artifacts(self.prev_run_id,
                                                               self.training_ids_artifact_path)
            previous_ids_df = pd.read_csv(previous_ids_path)
            self.CURRENT_ROUND= previous_ids_df.iloc[-1]['round'] + 1
            logger.info(f"Previous trainings IDs found - CURRENT ROUND: {self.CURRENT_ROUND}")
                        
            logger.info("Loading artifacts from previous run: Model")
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

   
    def __candid_loop(self):

        # First we have to define the pool of samples we can sample from
        # that is those we haven't trained on yet.
        self.train_ids = self.previous_ids_df["candid"].tolist()
        self.X_pool = self.X.drop(index=self.train_ids, 
                                  errors='ignore')
        
        self.meta_pool = self.meta.drop(index=self.train_ids, 
                                        errors='ignore')
        logger.info(f"Training pool contains {self.X_pool.shape[0]} samples ({self.meta_pool.objectId.unique().shape[0]} unique)")
        
        logger.info("Predicting class for training pool using previous model")
        self.y_pred = self.clf.predict_proba(self.X_pool)[:, 1]  
        self.y_pred_pool = pd.DataFrame(np.vstack((self.X_pool.index.values.astype(str), 
                                                    self.y_pred)).T, columns= ['candid', 
                                                                               'pred']).set_index('candid')
        
        ### here need a function that gives back sampling score: low is "good" (we want)
        self.y_pred_pool['sampling_score'] = self.y_pred_pool.apply(lambda row: uncertainty_sampling_score(row['pred']), axis=1)
        self.y_pred_pool.sort_values('sampling_score', ascending=True, inplace=True)
        self.candid_loop = self.y_pred_pool.index.astype(np.int64)



    def __labeling(self):
        new_labels = []
        new_label_candid = []
        new_sample_candid = []

        N_i = 0

        for _candid in self.candid_loop:
            try: 
                try:
                    classification = self.label2galclass[self.labels.loc[_candid].label]
                except TypeError:
                    logging.error(f"Shit! You got duplicate labels in {self.LABELS_PATH}")
                    logging.error(_candid)
                    exit()
                if not np.isnan(classification):
                    new_sample_candid.append(_candid)
                    N_i += 1
            except KeyError: 
                # if _candid not in labels then labels.loc[_candid] will throw a KeyError 
                # and we can activate the logic below
                _objectId = self.meta.loc[_candid].objectId
                
                # this is where we need the labeling 
                label = fvra_cli_label_one_object(_objectId)
                
                if label is None: 
                    continue
                new_labels.append(label)
                new_label_candid.append(_candid)
                classification = self.label2galclass[label]
                if not np.isnan(classification):
                    new_sample_candid.append(_candid)
                    N_i += 1
                
            if N_i == self.BATCH_SIZE:
                logging.info(f'Batch size ({self.BATCH_SIZE}) reached.')
                break

        if N_i < self.BATCH_SIZE:
            logging.warning(f'Batch size ({self.BATCH_SIZE}) not reached: {N_i}')       

        # ----------------------------------------------------
        # Record (or update) labels
        # ----------------------------------------------------
        timestamp = datetime.utcnow().isoformat()
        new_labels_df = pd.DataFrame.from_dict({
                                                'objectId': self.meta.loc[np.array(new_label_candid
                                                                                   ).astype(np.int64)].objectId,
                                                'label':  new_labels,
                                                'timestamp': timestamp
                                            })
        updated_labels = pd.concat((self.labels, 
                                    new_labels_df))
        updated_labels.to_csv(self.LABELS_PATH)
        logging.info(f"Updated the labels at: {self.LABELS_PATH}")

        new_ids_df = pd.DataFrame({'candid': np.array(new_sample_candid).astype(np.int64),
                                'round': self.CURRENT_ROUND,
                                })
        return new_ids_df, updated_labels