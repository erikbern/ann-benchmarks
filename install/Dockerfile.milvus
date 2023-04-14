FROM milvusdb/knowhere-env:pyknowhere1.3-20230303

WORKDIR /home/app
COPY run_algorithm.py requirements.txt ./

# update python packages
RUN pip3 install -r requirements.txt
RUN pip3 install scipy==1.10.1
RUN python3 -c 'import pyknowhere'

ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
