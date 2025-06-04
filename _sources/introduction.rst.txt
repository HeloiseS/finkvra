Introduction
==============

.. important::

    These are my **personnal research notes** for project that is **under development**.
    Feel free to loiter. 

|:checkered_flag:| Motivations
--------------------------------

LSST is coming and with it a new stream with potentially regular 
changes to the schema, the data, or changes in science goals. 

This project is aimed at leveraging the resources in the Fink data broker
to create "Vritual Research Assistants" (VRAs) **quickly** and **dynamically**.
If the data changes or the goal changes, we want to be able to 
train a VRA **directly on the stream** with **very little data**. 

To do this there are two key points of methodology we are harnessing:

1. **Active Learning**: We want to be able to train a model on a small amount of data, and then iteratively improve it with new data.

2. **Lightweight models**: That can be trained on a few samples and need little compute.

3. **Feature engineering**: Although feature-based machine learning is not popular anymore, it is very powerful because it is easier to interpret and understand, which is essential for science especially in a context where we might be creating or updating models quickly. 


.. danger::

    You need to think carefully through some science cases to drive the design. For example how does this work for someone who cares about transients that happen on average once a month? In this case you'll need historical data. Here there are **different kinds of use cases** you need to get clear in your head (after you've done your first large set of AL experiments). 


|:mortar_board:| Active Learning
-------------------------------------
Active Learning (AL) is an area of Machine Learning (ML) concerned about 
optimizing the samples used to train our ML models by using the power of 
**Information Theory**, which is a branch of mathematics that 
tries to quantify how much information is in a given bit of data. 

The idea of AL is that if we are smart about which samples we add 
to our training set, we can train a model with less data (or make a more representative model). 
This is useful in our context because we are using light-weight ML models which are 
**supervised** - meaning they require labels alongside the data to learn.
But **labeling data is expensive** - either because you have to take a spectrum or because 
a human needs to sit down and eyeball. 
With AL we can be picky about the data and train with less labour/extra resources. 

Also the idea of using AL is to train iteratively.
We start with very little data, the model is bad, we add a little bit of data at a time, 
and over time we make a better model. It takes more steps but we can make a better 
model with a fraction of the data. Training this way **makes sense when we have a constant data stream**. 


.. admonition:: To Do 

    I could try to quantify that with the ATLAS VRA dataset as it's a closed experiment. 


Uncertainty sampling
+++++++++++++++++++++

In the initial tests of the Fink VRA project we are going to use the most 
basic AL method: **uncertainty sampling**. The concept is simple - 
the samples where the model is most unsure are the ones which are going to 
teach the model the most. 
In uncertainty sampling we take our model from yesterday, predict 
the scores for today's data, and then we select the top X samples that 
were the most unclear and ask the human for a label. 

* **Pros**: It's simple and it's been shown to be very competitve with other fancier methods `(Kennamer et al. 2020) <https://arxiv.org/abs/2010.05941>`_
* **Cons**: It might be easy to end up with samples that have just very little information and even a human eyeballer can't do much with that. Also if a model is **confidently wrong** we'll miss it. 

Entropy
+++++++++
Entropy is not only a concept in physics, it's also often used in ML, stats and information theory. 
There are different ways to calculate it, always with a probablity being shoved into a natural log, then there 
can be extra steps involved. 

In our present case entropy can be used for two things: **confusion** and **diversity**. 

Confusion
~~~~~~~~~~ 

Since we're doing binary classification the formula is 


.. math::

   H(X) = -y \ln{p} - \Big (\mid y  - p \mid \times \ln \mid y - p \mid \Big )

where ``y`` is the **true label** and ``p`` is the **predicted probability**.

.. danger:: 

    I need to double check my maths - something doesn't look right

  
Instead of **sampling our next batch of data** based on our uncertain we are, 
we select based on how far from the "true" label the predictions are. 
The problem is that **this requires knowing the labels**.

.. admonition:: To Do 

    Try to add an entropy sampling step every week or two when labels are made available through, e.g. TNS. 


Diversity
~~~~~~~~~~ 
The other way we can use entropy is to quantify how **diverse** the data in our batch is. 
If there are quite a few samples (and there will be quite a few alerts in LSST),
then the top, say 10, samples in the list might be **very similar**.
This slows down the learning process and could lead to ovefitting (although HGBDT - see below - are prety robust). 


.. admonition:: To Do 

    Try to understand how to implement this (see `Kennamer et al. 2020 <https://arxiv.org/abs/2010.05941>`_).


|:deciduous_tree:| Histogram Based Gradient Boosted Decision Trees
--------------------------------------------------------------------

**To Write**

see sklearn documentation 