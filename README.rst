Benchmarking nearest neighbors
------------------------------

This project contains some tools to benchmark various implementations of approximate nearest neighbor (ANN) search.

Evaluated
---------

 * Annoy
 * FLANN
 * scikit-learn
 * PANNS
 * NearPy
 * KGraph

Data sets
---------

 * GloVe
 * SIFT

Motivation
----------

Doing fast searching of nearest neighbors in high dimensional spaces is an increasingly important problem, but with little attempt at objectively comparing methods.

Princinples
-----------

 * Everyone is welcome to submit pull requests with tweaks and changes to how each library is being used.
 * This is meant to be an ongoing project and represent the current state.
 * Make everything easy to replicate, including installing and preparing the datasets.
 * To make it simpler, look only at the precision-performance tradeoff.
 * Try many different values of parameters for each library and ignore the points that are not on the precision-performance frontier.
 * High-dimensional datasets with approximately 100-1000 dimensions. This is challenging but also realistic. Not more than 1000 dimensions because those problems should probably be solved by doing dimensionality reduction separately.

Results
-------

This is very much a work in progress... more results coming later!

.. figure:: https://raw.github.com/erikbern/ann-benchmarks/master/plot.png
   :align: center
