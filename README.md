Benchmarking nearest neighbors
==============================

This project contains some tools to benchmark various implementations of approximate nearest neighbor (ANN) search for different metrics.

See [the results of this benchmark](http://sss.projects.itu.dk/ann-benchmarks).

Evaluated
=========

Euclidean space
---------------

* [Annoy](https://github.com/spotify/annoy)
* [FLANN](http://www.cs.ubc.ca/research/flann/)
* [scikit-learn](http://scikit-learn.org/stable/modules/neighbors.html): LSHForest, KDTree, BallTree
* [PANNS](https://github.com/ryanrhymes/panns)
* [NearPy](http://nearpy.io)
* [KGraph](https://github.com/aaalgo/kgraph)
* [NMSLIB (Non-Metric Space Library)](https://github.com/searchivarius/nmslib): SWGraph, HNSW, BallTree, MPLSH
* [RPForest](https://github.com/lyst/rpforest)
* [FALCONN](http://falconn-lib.org/)
* [FAISS](https://github.com/facebookresearch/faiss.git)
* [DolphinnPy](https://github.com/ipsarros/DolphinnPy)

Set similarity
--------------
* [Datasketch](https://github.com/ekzhu/datasketch)

Motivation
==========

Doing fast searching of nearest neighbors in high dimensional spaces is an increasingly important problem, but with little attempt at objectively comparing methods.

Data sets
=========

We have a number of precomputed data sets for this. All data sets are pre-split into train/test and come with ground truth data in the form of the top 100 neighbors. We store them in a HDF5 format:

| Dataset                                                           | Dimensions | Train size | Test size | Neighbors | Distance  | Download                                                                     |
| ----------------------------------------------------------------- | ---------: | ---------: | --------: | --------: | --------- | ---------------------------------------------------------------------------- |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) |        784 |     60,000 |    10,000 |       100 | Euclidean | [HDF5](http://vectors.erikbern.com/fashion-mnist-784-euclidean.hdf5) (217MB) |
| [GIST](http://corpus-texmex.irisa.fr/)                            |        960 |  1,000,000 |     1,000 |       100 | Euclidean | N/A (coming shortly)                                                         |
| [GloVe](http://nlp.stanford.edu/projects/glove/)                  |         25 |  1,133,628 |    59,886 |       100 | Angular   | [HDF5](http://vectors.erikbern.com/glove-25-angular.hdf5) (121MB)            |
| Glove                                                             |         50 |  1,133,628 |    59,886 |       100 | Angular   | [HDF5](http://vectors.erikbern.com/glove-50-angular.hdf5) (235MB)            |
| Glove                                                             |        100 |  1,133,628 |    59,886 |       100 | Angular   | [HDF5](http://vectors.erikbern.com/glove-100-angular.hdf5) (463MB)           |
| Glove                                                             |        200 |  1,133,628 |    59,886 |       100 | Angular   | [HDF5](http://vectors.erikbern.com/glove-200-angular.hdf5) (918MB)           |
| [MNIST](http://yann.lecun.com/exdb/mnist/)                        |        784 |     60,000 |    10,000 |       100 | Euclidean | [HDF5](http://vectors.erikbern.com/mnist-784-euclidean.hdf5) (217MB)         |
| [SIFT](https://corpus-texmex.irisa.fr/)                           |        128 |  1,000,000 |    10,000 |       100 | Euclidean | [HDF5](http://vectors.erikbern.com/sift-128-euclidean.hdf5) (501MB)          |

Note that a few other datasets were used previously, in particular for Hamming and set similarity. We are going to add them back shortly in the more convenient HDF5 format.

Install
=======

Clone the repo and run `bash install.sh`. This will install all libraries. It could take a while. It has been tested in Ubuntu 16.04. We advice to run it only in a VM.

Experiment Setup
================

Running a set of algorithms with specific parameters works:

* Check that `algos.yaml` contains the parameter settings that you want to test
* To run experiments on SIFT, invoke `python run.py --dataset glove-100-angular`. See `python run.py --help` for more information on possible settings. Note that experiments can take a long time. 
* To process the results, either use `python plot.py --dataset glove-100-angular` or `python createwebsite.py`. An example call: `python createwebsite.py --plottype recall/time --latex --scatter --outputdir website/`. 

Including Your Algorithm
========================
You have two choices to include your own algorithm. If your algorithm has a Python wrapper (or is entirely written in Python), then all you need to do is to add your algorithm into `ann_benchmarks/algorithms` by providing a small wrapper. 

Principles
==========

* Everyone is welcome to submit pull requests with tweaks and changes to how each library is being used.
* In particular: if you are the author of any of these libraries, and you think the benchmark can be improved, consider making the improvement and submitting a pull request.
* This is meant to be an ongoing project and represent the current state.
* Make everything easy to replicate, including installing and preparing the datasets.
* Try many different values of parameters for each library and ignore the points that are not on the precision-performance frontier.
* High-dimensional datasets with approximately 100-1000 dimensions. This is challenging but also realistic. Not more than 1000 dimensions because those problems should probably be solved by doing dimensionality reduction separately.
* No batching of queries, use single queries by default. ANN-Benchmarks saturates CPU cores by using a thread pool.
* Avoid extremely costly index building (more than several hours).
* Focus on datasets that fit in RAM. Out of core ANN could be the topic of a later comparison.
* We currently support CPU-based ANN algorithms. GPU support is planned as future work.
* Do proper train/test set of index data and query points.

Results
=======
See http://sss.projects.itu.dk/ann-benchmarks.

Note that NMSLIB saves indices in the directory indices. 
If the tests are re-run using a different seed and/or a different number of queries, the
content of this directory should be deleted.

Testing
=======

The project is fully tested using Travis, with unit tests run for all different libraries and algorithms.

References
==========

* [sim-shootout](https://github.com/piskvorky/sim-shootout) by Radim Řehůřek
* This [blog post](http://maheshakya.github.io/gsoc/2014/08/17/performance-comparison-among-lsh-forest-annoy-and-flann.html)
