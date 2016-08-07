.. image:: https://img.shields.io/travis/erikbern/ann-benchmarks/master.svg?style=flat
    :target: https://travis-ci.org/erikbern/ann-benchmarks

Benchmarking nearest neighbors
------------------------------

This project contains some tools to benchmark various implementations of approximate nearest neighbor (ANN) search.

Evaluated
---------

* `Annoy <https://github.com/spotify/annoy>`__
* `FLANN <http://www.cs.ubc.ca/research/flann/>`__
* `scikit-learn <http://scikit-learn.org/stable/modules/neighbors.html>`__: LSHForest, KDTree, BallTree
* `PANNS <https://github.com/ryanrhymes/panns>`__
* `NearPy <http://nearpy.io>`__
* `KGraph <https://github.com/aaalgo/kgraph>`__
* `NMSLIB (Non-Metric Space Library) <https://github.com/searchivarius/nmslib>`__
* `RPForest <https://github.com/lyst/rpforest>`__
* `FALCONN <http://falconn-lib.org/>`__

Data sets
---------

* `GloVe <http://nlp.stanford.edu/projects/glove/>`__
* `SIFT <http://corpus-texmex.irisa.fr/>`__

Motivation
----------

Doing fast searching of nearest neighbors in high dimensional spaces is an increasingly important problem, but with little attempt at objectively comparing methods.

Install
-------

Clone the repo and run ``bash install.sh``. This will install all libraries. It could take a while. It has been tested in Ubuntu 12.04 and 14.04.

To download and preprocess the data sets, run ``bash install/glove.sh`` and ``bash install/sift.sh``.

Principles
----------

* Everyone is welcome to submit pull requests with tweaks and changes to how each library is being used.
* In particular: if you are the author of any of these libraries, and you think the benchmark can be improved, consider making the improvement and submitting a pull request.
* This is meant to be an ongoing project and represent the current state.
* Make everything easy to replicate, including installing and preparing the datasets.
* To make it simpler, look only at the precision-performance tradeoff.
* Try many different values of parameters for each library and ignore the points that are not on the precision-performance frontier.
* High-dimensional datasets with approximately 100-1000 dimensions. This is challenging but also realistic. Not more than 1000 dimensions because those problems should probably be solved by doing dimensionality reduction separately.
* No batching of queries, use single queries by default. ann-benchmarks saturates CPU cores by using a thread pool.
* Avoid extremely costly index building (more than several hours).
* Focus on datasets that fit in RAM. Out of core ANN could be the topic of a later comparison.
* Do proper train/test set of index data and query points.

Results
-------

1.19M vectors from GloVe (100 dimensions, trained from tweets), cosine similarity, run on an c4.2xlarge instance on EC2.

.. figure:: https://raw.github.com/erikbern/ann-benchmarks/master/results/glove.png
   :align: center

1M SIFT features (128 dimensions), Euclidean distance, run on an c4.2xlarge:

.. figure:: https://raw.github.com/erikbern/ann-benchmarks/master/results/sift.png
   :align: center

Starting 2016-07-20 these results now reflect multi-threaded benchmarks so the results are not consistent with earlier results.

Note that FALCONN doesn't support multiple threads so the benchmark is affected by that.

Also note that NMSLIB saves indices in the directory indices. 
If the tests are re-run using a different seed and/or a different number of queries, the
content of this directory should be deleted.


Testing
-------

The project is fully tested using Travis, with unit tests run for all different libraries and algorithms.

References
----------

* `sim-shootout <https://github.com/piskvorky/sim-shootout>`__ by Radim Řehůřek
* This `blog post <http://maheshakya.github.io/gsoc/2014/08/17/performance-comparison-among-lsh-forest-annoy-and-flann.html>`__
