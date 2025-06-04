Fink stream prototypes
===============================================

The goal here is not to have a performant model but to create the codes 
and utilities to run AL training loops **directly on the Fink ZTF stream**.

There are jupyter notebooks in the
`fink-vra-notebooks <https://github.com/HeloiseS/fink-vra-notebooks>`_
repo (separate so it doesn't clog the core python utilities) which 
show:

1. How the features are extracted from the ``.parquet`` files we get from the consumer
2. The first training round, including ML Flow logging
3. An AL loop for a follow-up round.

The latter code is put in a script which is easier to run: ``gal_model_AL_loop.py``.

The features
------------------
The ``.parquet`` files are put into ``pandas.dataframes`` and then 
passed to the `finkvra.utils.features.make_features` function which 
creates the ``X`` and ``meta`` dataframes.
In ``1.Finkdata_to_X`` (see `fink-vra-notebooks <https://github.com/HeloiseS/fink-vra-notebooks>`_)
you can see it done bit by bit. 

The ``meta`` dataframe contains two columns: ``candid`` and ``objectId``.
It is used to create the links to the Lasair webpages for eyeballing, as the API
requires ``objectId``, not ``candid``. 

The features in the ``X`` dataframe are indexed on ``candid``. 
The current list is as follows | `[Fink features ref] <https://github.com/astrolabsoftware/fink-science/tree/master/fink_science/ztf/ad_features>`_
:

+--------------------------------+----------------------------------------+--------------+
| Column                         | Description                            | Who Made it  |
+================================+========================================+==============+
| ``ra``                         | RA deg                                 | ZTF          |
+--------------------------------+----------------------------------------+--------------+
| ``dec``                        | Dec deg                                | ZTF          |
+--------------------------------+----------------------------------------+--------------+
| ``drb``                        | Deep RealBogus score                   | ZTF          |
+--------------------------------+----------------------------------------+--------------+
| ``ndets``                      | Number detections                      | Me           |
+--------------------------------+----------------------------------------+--------------+
| ``nnondets``                   | Number non detections                  | Me           |
+--------------------------------+----------------------------------------+--------------+
| ``dets_median``                | Median mag detections                  | Me           |
+--------------------------------+----------------------------------------+--------------+
| ``dets_std``                   | Standard deviation mag detections      | Me           |
+--------------------------------+----------------------------------------+--------------+
| ``sep_arcsec``                 | Separation in Arcseconds               | Sherlock     |
+--------------------------------+----------------------------------------+--------------+
| ``amplitude``                  | The half amplitude of the LC           | Fink         |
+--------------------------------+----------------------------------------+--------------+
| ``linear_fit_reduced_chi2``    |                                        | Fink         |
+--------------------------------+----------------------------------------+--------------+
| ``linear_fit_slope``           |                                        | Fink         |
+--------------------------------+----------------------------------------+--------------+
| ``linear_fit_slope_sigma``     | The linear fit has an error term       | Fink         |
+--------------------------------+----------------------------------------+--------------+
| ``median``                     |                                        | Fink         |
+--------------------------------+----------------------------------------+--------------+
| ``median_absolute_deviation``  |                                        | Fink         |
+--------------------------------+----------------------------------------+--------------+
| ``amplituder_``                | Same as above but in g instead of r    | Fink         |
+--------------------------------+----------------------------------------+--------------+
| ``linear_fit_reduced_chi2r_``  |                                        | Fink         |
+--------------------------------+----------------------------------------+--------------+
| ``linear_fit_sloper_``         |                                        | Fink         |
+--------------------------------+----------------------------------------+--------------+
| ``linear_fit_slope_sigmar_``   |                                        | Fink         |
+--------------------------------+----------------------------------------+--------------+
| ``medianr_``                   |                                        | Fink         |
+--------------------------------+----------------------------------------+--------------+
| ``median_absolute_deviationr_``|                                        | Fink         |
+--------------------------------+----------------------------------------+--------------+


.. important::

    Amplitude probs mostly about periodic LC but maybe useful for us?
    Will have to do **permutation importance** to see what's useful and what isn't


.. attention::

    I really should add E(B-V) to the features!



The "first" AL loop
----------------------
Because of development and extensive debugging I ran many "first" loops but 
now I think I can call it done. Here's what I've got.

Here I am focusing on galactic models (Galactic Vs Not galactic) because 
there aren't as many bogus alerts as in the ATLAS data set I was working with. 
This is in part due to the the real/bogus constraints set by the Fink VRA filter (``drb > 0.5``)
and those set by Fink (``rb > 0.55``).

I initialised training with data from **the same day** (June 2nd 2025). 
The idea is to mimick training a model from scratch every time. 

I used **30 samples for the first batch** - it feels like an easy enough number to eyeball at the start. 
Emille recommended against large batches at first because in the early days the model is `very` bad 
and it's essentially random sampling. 
But the first batch being larger has the advantage of the model not being completely useless from the 
get-go (in my ATLAS tests, the models trained with 3 or 5 at a time crawled at the start - I think I 
wouldn't go lower than 10. I picked 30 to kick start it, but **it is worth investigating these strategies**. 

I used the ``gal_model_AL_loop.py`` script to run the subsequent loops,
although there is a jupyter notebook, it was mostly for developement. 
I ran four additional rounds of training with each step **adding 10 new samples**,
using the uncertainty sampling method.

For this first test I only recorded accuracy as a metric, but we 
can see how the models improve with each round. 

+---------------------+------------------+
| # Labeled Samples   | Accuracy Score   |
+=====================+==================+
| 30                  | 0.63             |
+---------------------+------------------+
| 40                  | 0.75             |
+---------------------+------------------+
| 50                  | 0.88             |
+---------------------+------------------+
| 60                  | 0.87             |
+---------------------+------------------+
| 70                  | 0.93             |
+---------------------+------------------+


