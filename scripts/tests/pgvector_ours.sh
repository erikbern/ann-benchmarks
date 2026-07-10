# !/bin/bash
# ours pgvector测试脚本
echo "ours pgvector测试开始..."

# ./scripts/tests/compile.sh ours
# ./scripts/tests/compile.sh baseline

# 步骤3: 设置PostgreSQL多线程参数
su - postgres -c "createdb ann"
sudo -u postgres psql -d ann -c "DROP EXTENSION IF EXISTS vector CASCADE;"
sudo -u postgres psql -d ann -c "CREATE EXTENSION vector;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET max_parallel_workers_per_gather = 20;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET max_parallel_workers = 20;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET parallel_tuple_cost = 0.1;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET parallel_setup_cost = 1000.0;"

# 设置日志配置，日志文件在/var/log/postgresql/postgresql-16-main.log 
echo "设置日志配置..."
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET logging_collector = on;"
sudo -u postgres psql -d ann -c "ALTER SYSTEM SET log_min_messages = info;"
sudo service postgresql restart

sudo -u postgres psql -d ann -c "SHOW config_file;"
sudo -u postgres psql -d ann -c "SELECT pg_reload_conf();"

# 验证日志配置是否生效
# sudo -u postgres psql -d ann -c "SHOW log_min_messages;"
# sudo -u postgres psql -d ann -c "SHOW log_directory;"
# sudo -u postgres psql -d ann -c "SHOW log_filename;"
# sudo -u postgres psql -d ann -c "SHOW logging_collector;"
# 验证vector_batch类型是否存在
# sudo -u postgres psql -d ann -c "SELECT typname FROM pg_type WHERE typname = 'vector_batch';"

# 步骤4: 设置环境变量并运行多线程测试
echo "4: 运行多线程ann-benchmark测试..."
export ANN_BENCHMARKS_PG_USER=postgres
export ANN_BENCHMARKS_PG_PASSWORD=
export ANN_BENCHMARKS_PG_DBNAME=ann
export ANN_BENCHMARKS_PG_START_SERVICE=false

export http_proxy=http://127.0.0.1:33210
export https_proxy=http://127.0.0.1:33210
# 测试GPU版本
python run.py --local --algorithm pgvector_ivfflat_gpu --dataset $1 --force --runs 1 --batch

# 测试多线程版本
# python run.py --local --algorithm pgvector_ivfflat_multi --dataset $1 --force --runs 1 --batch

# 测试jl版本，半途而废了，不用管
# python run.py --local --algorithm pgvector_ivfjl --dataset $1 --force --runs 1 --batch
./scripts/tests/filter_postgres_log.sh

echo "ours pgvector测试完成"


