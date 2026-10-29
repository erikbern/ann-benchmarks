FROM ann-benchmarks

# 更新包列表并安装基础依赖
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake g++ python3 python3-setuptools python3-pip libblas-dev liblapack-dev libopenblas-dev 
# 安装 Python 依赖
RUN pip3 --default-timeout=3000 install pybind11 numpy  
RUN git clone -b main "https://github.com/wangziying1004/qpg.git"
RUN ldconfig
RUN pip3 install qpg/qpgbuild-1.0-cp310-cp310-linux_x86_64.whl
RUN pip3 install qpg/qpgsearch-1.0-cp310-cp310-linux_x86_64.whl

