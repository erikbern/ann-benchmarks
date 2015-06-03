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

Data sets
---------

* `GloVe <http://nlp.stanford.edu/projects/glove/>`__
* `SIFT <http://corpus-texmex.irisa.fr/>`__

Motivation
----------

Doing fast searching of nearest neighbors in high dimensional spaces is an increasingly important problem, but with little attempt at objectively comparing methods.

Install
-------

Clone the repo and run ``bash install.sh``. This will install all libraries as well as downloading and preprocessing all data sets. It could take a while. It has been tested in Ubuntu 14.04. 

Princniples
-----------

* Everyone is welcome to submit pull requests with tweaks and changes to how each library is being used.
* In particular: if you are the author of any of these libraries, and you think the benchmark can be improved, consider making the improvement and submitting a pull request.
* This is meant to be an ongoing project and represent the current state.
* Make everything easy to replicate, including installing and preparing the datasets.
* To make it simpler, look only at the precision-performance tradeoff.
* Try many different values of parameters for each library and ignore the points that are not on the precision-performance frontier.
* High-dimensional datasets with approximately 100-1000 dimensions. This is challenging but also realistic. Not more than 1000 dimensions because those problems should probably be solved by doing dimensionality reduction separately.
* Use cosine similarity. For libraries where this is not supported, this is trivially achieved by normalizing all vectors.
* Use single core benchmarks. I believe most real world scenarios could be parallelized in other ways (eg. do multiple queries in parallel).
* Avoid extremely costly index building (more than several hours).
* Focus on datasets that fit in RAM. Out of core ANN could be the topic of a later comparison.
* Do proper train/test set of index data and query points.

Results
-------

Currently this is on the top 100k vectors from GloVe (27B Twitter version with 100 dimensions).
This is very much a work in progress... more results coming later!

.. figure:: https://raw.github.com/erikbern/ann-benchmarks/master/plot.png
   :align: center

References
----------

* `sim-shootout <https://github.com/piskvorky/sim-shootout>`__ by Radim Řehůřek
* `NonMetricSpaceLib <https://github.com/searchivarius/NonMetricSpaceLib>`__
* This `blog post <http://maheshakya.github.io/gsoc/2014/08/17/performance-comparison-among-lsh-forest-annoy-and-flann.html>`__
