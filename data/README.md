Data installation utilities
---------------------------

Prepares a bunch of datasets

1. Downloads/parses datasets
2. Train/test split (if not already done)
3. Compute correct result (using brute force search)
4. Create a .hdf5 file

Available datasets
------------------

| Dataset         | Dimensions | Train size | Test size | Neighbors | Distance  | URL                                                          |
| --------------- | ---------: | ---------: | --------: | --------: | --------- | ------------------------------------------------------------ |
| Fashion-MNIST   |        784 |     60,000 |    10,000 |       100 | Euclidean | http://vectors.erikbern.com/fashion-mnist-784-euclidean.hdf5 |
| GIST            |        960 |  1,000,000 |     1,000 |       100 | Euclidean | http://vectors.erikbern.com/gist-960-euclidean.hdf5          |
| Glove           |         25 |  1,133,628 |    59,886 |       100 | Angular   | http://vectors.erikbern.com/glove-25-angular.hdf5            |
| Glove           |         50 |  1,133,628 |    59,886 |       100 | Angular   | http://vectors.erikbern.com/glove-50-angular.hdf5            |
| Glove           |        100 |  1,133,628 |    59,886 |       100 | Angular   | http://vectors.erikbern.com/glove-100-angular.hdf5           |
| Glove           |        200 |  1,133,628 |    59,886 |       100 | Angular   | http://vectors.erikbern.com/glove-200-angular.hdf5           |
| MNIST           |        784 |     60,000 |    10,000 |       100 | Euclidean | http://vectors.erikbern.com/mnist-784-euclidean.hdf5         |
| SIFT            |        128 |  1,000,000 |    10,000 |       100 | Euclidean | http://vectors.erikbern.com/sift-128-euclidean.hdf5          |
