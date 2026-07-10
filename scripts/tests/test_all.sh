# 用甲方给的三个数据集测试
# for dataset in "sift-128-euclidean" "deep-image-96-angular" "text1M-200-angular"
# for dataset in "text1M-200-angular"
./scripts/tests/compile.sh ours

for dataset in "sift-128-euclidean" "deep-image1M-96-angular" "text1M-200-angular" "sift-128-euclidean" "sift-128-euclidean" 
# for dataset in "deep-image1M-96-angular" "text1M-200-angular"

do
    # 测试原版pgvector
    # ./scripts/tests/pgvector_origin.sh $dataset

    # 测试我们的pgvector
    ./scripts/tests/pgvector_ours.sh $dataset

    # 测试cuvs
    # ./scripts/tests/cuvs.sh $dataset

    # TODO:添加pgvector的其他改良版本
    python plot.py --x-scale linear --y-scale log --batch --no-pareto-frontier --dataset $dataset
done

