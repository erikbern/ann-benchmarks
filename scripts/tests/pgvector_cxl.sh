#!/bin/bash

# 恢复原版pgvector扩展并测试脚本
./scripts/tests/compile.sh ours

# 步骤3: 验证原版扩展
echo "3: 验证原版扩展..."
su - postgres -c "createdb ann"
sudo -u postgres psql -d ann -c "DROP EXTENSION IF EXISTS vector CASCADE;"
sudo -u postgres psql -d ann -c "CREATE EXTENSION vector;"
sudo -u postgres psql -d ann -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"



# cp /root/cxl/pgvector/test_gpu/test_gpu_optimization.sql /tmp/test_gpu_optimization.sql
# psql -U postgres -d ann -f /tmp/test_gpu_optimization.sql

rm /tmp/performance_comparison_test.sql
cp /root/pgvector/test_ivfflat_batch/performance_comparison_test.sql /tmp/performance_comparison_test.sql
psql -U postgres -d ann -f /tmp/performance_comparison_test.sql

# cp /root/cxl/pgvector/test_ivfflat_batch/96.sql /tmp/96.sql
# psql -U postgres -d ann -f /tmp/96.sql

# 步骤4: 设置环境变量并运行原版测
# echo "4: 运行原版pgvector ann-benchmark测试..."
# export ANN_BENCHMARKS_PG_USER=postgres
# export ANN_BENCHMARKS_PG_PASSWORD=
# export ANN_BENCHMARKS_PG_DBNAME=ann
# export ANN_BENCHMARKS_PG_START_SERVICE=false

# # 测试多线程版本
# python run.py --local --algorithm pgvector_ivfflat_multi --dataset $1 --batch --runs 1 --force

# # 测试单线程版本
# # python run.py --local --algorithm pgvector_ivfflat_single --dataset $1 --batch --runs 1 --force
./scripts/tests/filter_postgres_log.sh

echo "cxl pgvector测试完成"






