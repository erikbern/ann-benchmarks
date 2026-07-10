# 使用 AMD 官方镜像
FROM rocm/dev-ubuntu-22.04:6.2

LABEL maintainer="qingming-flat"

# 安装依赖
RUN sudo apt-get update && sudo apt-get install -y python3-pip git

# 安装 Python 库
RUN pip3 install numpy h5py pybind11

# [SABO FIX: 关键路径修正]
# 1. 先创建完整的 Python 包目录结构
WORKDIR /home/app/ann_benchmarks/algorithms/qingming

# 2. 把当前目录的所有文件(cpp, py, yml) 复制进这个深层目录
COPY . .

# 3. 原地编译
# 现在的上下文就在 .../qingming 下，hipcc 能找到文件
RUN hipcc -O3 -shared -fPIC -std=c++17 \
    --offload-arch=gfx1100 \
    -x hip qingming_hip.cpp \
    $(python3 -m pybind11 --includes) \
    -o qingming_hip$(python3-config --extension-suffix)

# 4. 回到根目录准备运行
WORKDIR /home/app

# 5. [CRITICAL] 确保 Python 能从 /home/app 找到包
ENV PYTHONPATH=/home/app

ENTRYPOINT ["python3", "-u", "run.py"]
