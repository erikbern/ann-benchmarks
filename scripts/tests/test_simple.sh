# 用一个小数据集来测试
dataset="glove-100-angular" # TEXT500k-200-angular deep-image-96-angular sift-128-euclidean TEXT1M-200-angular 

# 测试原版pgvector
# ./scripts/tests/pgvector_origin.sh $dataset

# 测试我们的pgvector
./scripts/tests/pgvector_ours.sh $dataset


# 测试cuvs
# ./scripts/tests/cuvs.sh $dataset

# TODO:添加pgvector的其他改良版本

# 画图 
python plot.py --x-scale linear --y-scale log --batch --no-pareto-frontier --dataset $dataset

# 不重新计算，只想画图用这个命令
# python plot.py --x-scale linear --y-scale log --batch --dataset "glove-100-angular"