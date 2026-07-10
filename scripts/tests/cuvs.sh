#!/bin/bash

# 加载cuvs环境变量
export CUDA_HOME=/usr/local/cuda-12.1 && export CUDA_PATH=/usr/local/cuda-12.1
python run.py --local --algorithm cuvs_ivfflat --dataset "$1" --batch --runs 1 --force
python run.py --local --algorithm cuvs_ivfpq --dataset "$1" --batch --runs 1 --force

echo "cuvs测试完成"






