Benchmarking nearest neighbors
==============================

[![Build Status](https://img.shields.io/github/workflow/status/erikbern/ann-benchmarks/ANN%20benchmarks?style=flat-square)](https://github.com/erikbern/ann-benchmarks/actions?query=workflow:benchmarks)

Doing fast searching of nearest neighbors in high dimensional spaces is an increasingly important problem, but so far there has not been a lot of empirical attempts at comparing approaches in an objective way.

This project contains some tools to benchmark various implementations of approximate nearest neighbor (ANN) search for different metrics. We have pregenerated datasets (in HDF5) formats and we also have Docker containers for each algorithm. There's a [test suite](https://travis-ci.org/erikbern/ann-benchmarks) that makes sure every algorithm works.

Evaluated
=========

* [Annoy](https://github.com/spotify/annoy)
* [FLANN](http://www.cs.ubc.ca/research/flann/)
* [scikit-learn](http://scikit-learn.org/stable/modules/neighbors.html): LSHForest, KDTree, BallTree
* [PANNS](https://github.com/ryanrhymes/panns)
* [NearPy](http://pixelogik.github.io/NearPy/)
* [KGraph](https://github.com/aaalgo/kgraph)
* [NMSLIB (Non-Metric Space Library)](https://github.com/nmslib/nmslib): SWGraph, HNSW, BallTree, MPLSH
* [hnswlib (a part of nmslib project)](https://github.com/nmslib/hnsw)
* [RPForest](https://github.com/lyst/rpforest)
* [FAISS](https://github.com/facebookresearch/faiss.git)
* [DolphinnPy](https://github.com/ipsarros/DolphinnPy)
* [Datasketch](https://github.com/ekzhu/datasketch)
* [PyNNDescent](https://github.com/lmcinnes/pynndescent)
* [MRPT](https://github.com/teemupitkanen/mrpt)
* [NGT](https://github.com/yahoojapan/NGT): ONNG, PANNG, QG
* [SPTAG](https://github.com/microsoft/SPTAG)
* [PUFFINN](https://github.com/puffinn/puffinn)
* [N2](https://github.com/kakao/n2)
* [ScaNN](https://github.com/google-research/google-research/tree/master/scann)
* [Elastiknn](https://github.com/alexklibisz/elastiknn)
* [OpenSearch KNN](https://github.com/opensearch-project/k-NN)
* [DiskANN](https://github.com/microsoft/diskann): Vamana, Vamana-PQ
* [Vespa](https://github.com/vespa-engine/vespa)
* [scipy](https://docs.scipy.org/doc/scipy/reference/spatial.html): cKDTree
* [vald](https://github.com/vdaas/vald)

Data sets
=========

We have a number of precomputed data sets for this. All data sets are pre-split into train/test and come with ground truth data in the form of the top 100 neighbors. We store them in a HDF5 format:

| Dataset                                                           | Dimensions | Train size | Test size | Neighbors | Distance  | Download                                                                   |
| ----------------------------------------------------------------- | ---------: | ---------: | --------: | --------: | --------- | -------------------------------------------------------------------------- |
| [DEEP1B](http://sites.skoltech.ru/compvision/noimi/)              |         96 |  9,990,000 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/deep-image-96-angular.hdf5) (3.6GB)
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) |        784 |     60,000 |    10,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5) (217MB) |
| [GIST](http://corpus-texmex.irisa.fr/)                            |        960 |  1,000,000 |     1,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/gist-960-euclidean.hdf5) (3.6GB)          |
| [GloVe](http://nlp.stanford.edu/projects/glove/)                  |         25 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-25-angular.hdf5) (121MB)            |
| GloVe                                                             |         50 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-50-angular.hdf5) (235MB)            |
| GloVe                                                             |        100 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-100-angular.hdf5) (463MB)           |
| GloVe                                                             |        200 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-200-angular.hdf5) (918MB)           |
| [Kosarak](http://fimi.uantwerpen.be/data/)                        |      27983 |     74,962 |       500 |       100 | Jaccard   | [HDF5](http://ann-benchmarks.com/kosarak-jaccard.hdf5) (2.0GB)             |
| [MNIST](http://yann.lecun.com/exdb/mnist/)                        |        784 |     60,000 |    10,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/mnist-784-euclidean.hdf5) (217MB)         |
| [NYTimes](https://archive.ics.uci.edu/ml/datasets/bag+of+words)   |        256 |    290,000 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/nytimes-256-angular.hdf5) (301MB)         |
| [SIFT](http://corpus-texmex.irisa.fr/)                           |        128 |  1,000,000 |    10,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/sift-128-euclidean.hdf5) (501MB)          |
| [Last.fm](https://github.com/erikbern/ann-benchmarks/pull/91)     |         65 |    292,385 |    50,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/lastfm-64-dot.hdf5) (135MB)               |

Results
=======

Interactive plots can be found at <http://ann-benchmarks.com>. These are all as of December 2021, running all benchmarks on a r5.4xlarge machine on AWS with `--parallelism 7`:

glove-100-angular
-----------------

![glove-100-angular](https://raw.github.com/erikbern/ann-benchmarks/master/results/glove-100-angular.png)

sift-128-euclidean
------------------

![glove-100-angular](https://raw.github.com/erikbern/ann-benchmarks/master/results/sift-128-euclidean.png)

fashion-mnist-784-euclidean
---------------------------

![fashion-mnist-784-euclidean](https://raw.github.com/erikbern/ann-benchmarks/master/results/fashion-mnist-784-euclidean.png)

lastfm-64-dot
------------------

![lastfm-64-dot](https://raw.github.com/erikbern/ann-benchmarks/master/results/lastfm-64-dot.png)

nytimes-256-angular
-------------------

![nytimes-256-angular](https://raw.github.com/erikbern/ann-benchmarks/master/results/nytimes-256-angular.png)

glove-25-angular
----------------

![glove-25-angular](https://raw.github.com/erikbern/ann-benchmarks/master/results/glove-25-angular.png)

Install
=======

The only prerequisite is Python (tested with 3.6) and Docker.

1. Clone the repo.
2. Run `pip install -r requirements.txt`.
3. Run `python install.py` to build all the libraries inside Docker containers (this can take a while, like 10-30 minutes).

Running
=======

1. Run `python run.py` (this can take an extremely long time, potentially days)
2. Run `python plot.py` or `python create_website.py` to plot results.

You can customize the algorithms and datasets if you want to:

* Check that `algos.yaml` contains the parameter settings that you want to test
* To run experiments on SIFT, invoke `python run.py --dataset glove-100-angular`. See `python run.py --help` for more information on possible settings. Note that experiments can take a long time. 
* To process the results, either use `python plot.py --dataset glove-100-angular` or `python create_website.py`. An example call: `python create_website.py --plottype recall/time --latex --scatter --outputdir website/`. 

Including your algorithm
========================

1. Add your algorithm into `ann_benchmarks/algorithms` by providing a small Python wrapper.
2. Add a Dockerfile in `install/` for it
3. Add it to `algos.yaml`
4. Add it to `.github/workflows/benchmarks.yml`

Principles
==========

* Everyone is welcome to submit pull requests with tweaks and changes to how each library is being used.
* In particular: if you are the author of any of these libraries, and you think the benchmark can be improved, consider making the improvement and submitting a pull request.
* This is meant to be an ongoing project and represent the current state.
* Make everything easy to replicate, including installing and preparing the datasets.
* Try many different values of parameters for each library and ignore the points that are not on the precision-performance frontier.
* High-dimensional datasets with approximately 100-1000 dimensions. This is challenging but also realistic. Not more than 1000 dimensions because those problems should probably be solved by doing dimensionality reduction separately.
* Single queries are used by default. ANN-Benchmarks enforces that only one CPU is saturated during experimentation, i.e., no multi-threading. A batch mode is available that provides all queries to the implementations at once. Add the flag `--batch` to `run.py` and `plot.py` to enable batch mode. 
* Avoid extremely costly index building (more than several hours).
* Focus on datasets that fit in RAM. For billion-scale benchmarks, see the related [big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks) project.
* We mainly support CPU-based ANN algorithms. GPU support exists for FAISS, but it has to be compiled with GPU support locally and experiments must be run using the flags `--local --batch`. 
* Do proper train/test set of index data and query points.
* Note that we consider that set similarity datasets are sparse and thus we pass a **sorted** array of integers to algorithms to represent the set of each user.


Authors
=======

Built by [Erik Bernhardsson](https://erikbern.com) with significant contributions from [Martin Aumüller](http://itu.dk/people/maau/) and [Alexander Faithfull](https://github.com/ale-f).

Related Publication
==================

The following publication details design principles behind the benchmarking framework: 

- M. Aumüller, E. Bernhardsson, A. Faithfull:
[ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms](https://arxiv.org/abs/1807.05614). Information Systems 2019. DOI: [10.1016/j.is.2019.02.006](https://doi.org/10.1016/j.is.2019.02.006)

Related Projects
================

- [big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks) is a benchmarking effort for billion-scale approximate nearest neighbor search as part of the [NeurIPS'21 Competition track](https://neurips.cc/Conferences/2021/CompetitionTrack).

