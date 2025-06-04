Workflows and pipelines
=========================

`Currently everything is run locally on my laptop`

|:electric_plug:| Plugging into the stream
-------------------------------------------------
First step is to be able to plug into the Kafka stream, grab the alert 
data and save a clean version of the data locally. 
The full packet, which I need to get the historical (30 days) lighcurve data, also has 
the three image stamps - it is _massive_. 
So I don't save the whole thing, I clean it on the fly and save that locally. 

|:radio:| Listening
+++++++++++++++++++++++++++++++

The ``finkvra`` package has a polling utility (``utils.consumer.poll_n_alerts``)
use to "listen" to the stream. 
It is called **every hour** in a cron job. 

- The bash: ``listen.sh``

.. code-block:: bash

    #!/bin/bash
    # Activate conda base and run listener

    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate base 
    export LASAIR_TOKEN="MYTOKEN"
    export PYTHONPATH="/home/stevance/software/finkvra:$PYTHONPATH"

    python /home/stevance/Data/FinkZTFStream/listen.py

- The python: ``listen.py`` polls **1000 alerts** from the ``fink_vra_ztf`` topic (filter)

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from finkvra.utils import consumer as finkconsumer
    from glob import glob


    myconfig = {
        'bootstrap.servers': 'kafka-ztf.fink-broker.org:24499',
        'group.id': 'heloise_finkvra'
    }

    topics = ['fink_vra_ztf'] # the fink VRA filter I made with Julien in May 2025

    finkconsumer.poll_n_alerts(myconfig, topics, n=1000)


The ``fink_vra_ztf`` topic is directly implemented in the Fink data broker. 
It has the following criteria:

-  ``candidate.magpsf > 19.5``
- ``candidate.drb >0.5``   (deep learning RB score - we picked 0.5 because they filter on rb score 0.55)
- ``cdsxmatch='Unknown'`` (is it known in simbad)
- ``roid<3``  (not a asteroid)


|:detective:| Sherlock
+++++++++++++++++++++++++
In the ``poll_n_alerts`` function also calls a ``run_sherlock`` function which calls
the Lasair API to get the Sherlock classification and some additional features like 
the separation from the host match. 

This was added because I found that most alerts returned in the stream were AGNs or Variable Stars.
These can be filtered out during the cleaning step of the polling function with Sherlock. 

.. important::

    In the context of LSST, through the ACME project, Sherlock will be ran further up stream and I won't have to worry about this step. So it's okay if it's a bit inefficient.


|:card_file_box:| Alert Data
+++++++++++++++++++++++++++++++++
The alerts data are saved as ``.parquet`` files in my ``~Data/FinkZTFStream/`` folder,
with the format ``YYYYMMDD_HHMMSS_alerts.parquet``.

    
|:mortar_board:| Training the models
-------------------------------------

|:ocean:| ML Flow
++++++++++++++++++++
We are logging our models and "experiments" using `ML Flow <https://mlflow.org/>`_.
The first thing to do is to start the ML Flow server **inside the ``~Science/fink_vra_notebooks/`` directory**.:

.. code-block:: bash

    mlflow server --host 127.0.0.1 --port 6969

.. warning::

    If you start the server in a different location it won't find the artifacts and logs.

|:checkered_flag:| Initialising the active learning loop
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
So far I'm doing this in a jupyter notebook (see e.g. ``2.First_training.ipynb`` in `fink-vra-notebooks <https://github.com/HeloiseS/fink-vra-notebooks>`_)
In this first round of training we define a ``EXPERIMENT`` name 
which will be used in subsequent runs to find past models. 
The logic for that first round is similar to the other loops described below,
apart from the fact that we chose **effectively randomly** the first set of alerts. 

The number of alerts used for the first batch set in that notebook 
is not necessarily the same as the number of alerts used in subsequent loops, 
and we will test the best instantiating and follow-up strategies. 


|:runner:| Running subsequent loops
+++++++++++++++++++++++++++++++++++++
Here I provide the pseudo-code but details of the step by step can be seen in
``3.Testing_AL_loop.ipynb`` in `fink-vra-notebooks <https://github.com/HeloiseS/fink-vra-notebooks>`_)

On the day-to-day the code is run in a script rather than cell by cell though. 

|:scroll:| **Pseudo-code**

* Set up (ML flow experiment name, linking to server)
* Get the last successful run ID. This is where we find the previous ML model that we'll use to predict and sample.
* Load the ``.parquet`` data from the directory where I save the Fink-ZTF data I get from my cron job
* Make features using ``finkvra.utils.features.make_features`` (+ remove objects with no postive diff)
* Load the `candid` we've used for training before from the `training_ids.csv` file, and create the ``CURRENT_ROUND`` number
* Create `X_pool` the features for the pool of samples that **have not yet been used for training**. 
* Load previous model from our previous run id 
* Predict the classification for all `X_pool`
* Create the `uncertainty_score` column - for now using **uncertainty sampling**
* Order the list of candids from our pool by that `uncertainty_score` column
* **Active Learning loop with dynamic labelling**

   +  Load existing labels from the `labels.csv` file
   + set up the variable for the loop

    .. code:: python

        new_labels = [] # where we'll store new labels that dont already exist
        new_label_candid = [] # where we store candid for the new labels we made
        new_sample_candid = [] # where we store the candid for the alerts we've sampled for our AL loop

        N_to_sample = 10 # our target
        N_i = 0


   + For each candid (ordered from most to least uncertain):

     - if there is an existing label in ``labels.csv`` 

        * turn the label to a classification (1, 0 or np.nan)
        * if classification is not NaN: record candid to `new_sampled_candid` and N_1 += 1

     - if not:

       * get ``objectId`` from ``meta.loc[candid]``
       * use the ``finkvra.utils.labels.cli_label_one_object`` (input = ``objectId``, output = label)
       * turn the label to a classification (1, 0 or np.nan)
       * if classification is not NaN: record candid to ``new_sampled_candid`` and N_1 += 1 

     - Check if N_i == N_to_sample 

* Make an updated label data frame and write out to ``labels.csv``
* concat the previous training candid and the new sample candid to make our **training ids**
* Make ``X_train`` and ``y_train`` from X and labels and the training ids
* Start the ML flow run:

.. code:: python 

    with mlflow.start_run(run_name=f"round_{CURRENT_ROUND}_{SAMPLING_STRATEGY}"):

        # Log metadata
        meta_info = {
            "round": int(CURRENT_ROUND),
            "timestamp": datetime.utcnow().isoformat(),
            "n_train": int(X_train.shape[0]),
            "sampling_strategy": str(SAMPLING_STRATEGY),
            "model_tag": str(MODEL_TAG)
        }

        with open("meta.json", "w") as f:
            json.dump(meta_info, f, indent=2)
        mlflow.log_artifact("meta.json")

        # Train model
        clf_new = HistGradientBoostingClassifier(max_iter=100, 
                                                l2_regularization=10,
                                                random_state=42,
                                                learning_rate=0.1)
        clf_new.fit(X_train.values, y_train.values)

        # Evaluate on training set
        acc = accuracy_score(y_train, clf_new.predict(X_train.values))
        mlflow.log_metric("train_accuracy", acc)

        # Log model
        signature = infer_signature(X_train, clf_new.predict(X_train))
        mlflow.sklearn.log_model(
            clf_new,
            artifact_path=ARTIFACT_PATH,
            signature=signature,
            input_example=X_train.iloc[:2]
        )

        # Save training state
        mlflow.log_artifact(f"{EXPERIMENT}_training_ids.csv")


Labelling data
+++++++++++++++++
The labels created through our labeling step are saved in the same directory as the ``.parquet`` files
in ``labeld.csv`` with columns ``candid``, ``objectId``, ``label``, ``timestamp``.

.. attention::

    The ``candid`` and ``objectId`` columns are not the same. The ``candid`` is the unique identifier of the alert, while the ``objectId`` is the unique identifier of the object in ZTF. 
    
The labels are indexed on ``candid`` not ``objectId``, and generally speaking when sampling
data we go by ``candid`` not ``objectId``. This means that a given object may be given
different labels if I eyeball it on different dates. At this stage I think this is a 
good thing because I am still working with the mindset of reproducing human classification. 
If we want to do **better than human** classification later, this may have to be reviewed. 

There are two ways to label the alerts:

1. In bulk using the ``finkvra.utils.labels.cli_label_alerts`` command line utility. 
    This is useful for the first round of training, where we want to label a large number of alerts.
    It will create a ``labeled.csv`` file in the same directory as the ``.parquet`` files.

2. One at a time using the ``finkvra.utils.labels.cli_label_one_object`` command line utility.
    This is useful for the active learning loop, where we want to label a small number of alerts at a time.
    It will return the label and the ``objectId`` of the alert.

