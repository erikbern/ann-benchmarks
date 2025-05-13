export ANN_BENCHMARKS_OG_USER='YourUserName'
export ANN_BENCHMARKS_OG_PASSWORD='YourPassword'
export ANN_BENCHMARKS_OG_DBNAME='YourDBName'
export ANN_BENCHMARKS_OG_HOST='YourHost'
export ANN_BENCHMARKS_OG_PORT=YourPort


python3 run.py --algorithm openGauss-hnsw --dataset fashion-mnist-784-euclidean --local --runs 2 -k 10 --batch 
python3 run.py --algorithm openGauss-hnsw --dataset gist-960-euclidean --local --runs 2 -k 10 --batch 
python3 run.py --algorithm openGauss-hnsw --dataset glove-100-angular --local --runs 2 -k 10 --batch 
python3 run.py --algorithm openGauss-hnsw --dataset sift-128-euclidean --local --runs 2 -k 10 --batch 
python3 run.py --algorithm openGauss-hnsw --dataset deep-image-96-angular --local --runs 2 -k 10 --batch 