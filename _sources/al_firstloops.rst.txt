2. First AL training loops on Fink ZTF stream
===========================================

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

2.1 The features
------------------


